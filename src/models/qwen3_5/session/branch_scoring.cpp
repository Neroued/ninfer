#include "models/qwen3_5/session/internal.h"
#include "models/qwen3_5/program/branch_scoring.h"

#include <algorithm>
#include <chrono>
#include <thread>

namespace ninfer::models::qwen3_5::detail {

bool SessionImpl::branch_scoring_available() const noexcept {
    return !closing && !execution && !scoring && !has_context_operation() && events.empty() &&
           !executor().has_pending_work() && !vision_window.current();
}

Result SessionImpl::wait_branch_work(runtime::ExecutionTicket ticket, bool context) {
    std::optional<Result> result;
    for (;;) {
        executor().flush();
        auto event = executor().poll();
        if (!event) {
            std::this_thread::yield();
            continue;
        }
        if (const auto* failed = std::get_if<runtime::ExecutionFailed>(&*event))
            throw RequestError(RequestErrorKind::Unavailable, failed->message);
        if (auto* ready = std::get_if<runtime::ResultReady<ExecutionTypes>>(&*event)) {
            if (ready->ticket != ticket || result)
                throw std::logic_error("decision result does not match its execution ticket");
            result.emplace(std::move(ready->result));
            if (!context) (void)executor().commit(ticket, Commit{});
        } else if (const auto* retired = std::get_if<runtime::DeviceRetired>(&*event)) {
            if (retired->ticket != ticket || !result)
                throw std::logic_error("decision retirement has no matching result");
            return std::move(*result);
        } else {
            throw std::logic_error("unexpected event during decision execution");
        }
    }
}

void SessionImpl::settle_branch_context(context::ContextTransaction& transaction) {
    if (closing || executor().failed()) {
        transaction.fail_after_drain();
        return;
    }
    try {
        transaction.prepare_actions();
        context::ContextActionCursor cursor;
        const auto& actions = transaction.actions();
        bool submitted      = false;
        do {
            auto batch = context::take_action_batch(actions, cursor);
            // Allocator maintenance can also remap chat pages. Keep those on native tables;
            // only this Session's private branch rows use the workspace-backed table view.
            const auto private_table = [&](const context::ContextAction& action) {
                const auto* mapping = std::get_if<context::MappingUpdate>(&action);
                return mapping && mapping->pool_id == kTextKvPool.index &&
                       mapping->row >= static_cast<int>(max_concurrency);
            };
            std::size_t begin = 0;
            do {
                const bool branch_tables =
                    begin < batch.operations.size() && private_table(batch.operations[begin]);
                auto end = begin;
                while (end < batch.operations.size() &&
                       private_table(batch.operations[end]) == branch_tables)
                    ++end;
                auto frame = executor().try_reserve(runtime::FrameKind::Context);
                if (!frame) throw std::logic_error("decision context frame is occupied");
                Work work{.operation = ContextWork{.branch_tables = branch_tables}};
                work.context.operations.assign(batch.operations.begin() + begin,
                                               batch.operations.begin() + end);
                const auto ticket = executor().submit(std::move(*frame), std::move(work));
                if (!submitted) {
                    transaction.mark_submitted();
                    submitted = true;
                }
                auto result = wait_branch_work(ticket, true);
                if (!std::holds_alternative<ContextResult>(result.output))
                    throw std::logic_error("decision context returned a non-context result");
                begin = end;
            } while (begin < batch.operations.size());
        } while (!cursor.complete(actions));
        transaction.mark_completed();
        transaction.publish();
    } catch (...) {
        executor().fail("decision Context transaction failed");
        try {
            executor().drain();
        } catch (...) {}
        transaction.fail_after_drain();
        throw;
    }
}

BranchResult SessionImpl::run_branch_work(BranchWork value) {
    const auto visual = value.visual;
    const auto end    = value.rows.front().frontier + value.rows.front().tokens.size();
    context::ContextTransaction address_borrow(directory);
    auto frame = executor().try_reserve(runtime::FrameKind::Compute);
    if (!frame) throw std::logic_error("decision compute frame is occupied");
    try {
        auto ticket = executor().submit(std::move(*frame), Work{.operation = std::move(value)});
        if (visual) vision_window.submitted(visual->window.id, ticket);
        auto result = wait_branch_work(ticket, false);
        if (visual) {
            vision_window.consumed(visual->window.id, end >= visual->window.plan->consume_end);
            vision_window.retired(ticket);
        }
        return std::get<BranchResult>(std::move(result.output));
    } catch (...) {
        executor().fail("decision execution failed");
        try {
            executor().drain();
        } catch (...) {}
        throw;
    }
}

void SessionImpl::release_branch_visual() {
    const auto* active = vision_window.current();
    if (!active) return;
    const auto owner = active->owner;
    const auto id    = active->id;
    // Called at an idle boundary after physical retirement, including cancelled requests.
    auto frame = executor().try_reserve(runtime::FrameKind::Context);
    if (!frame) throw std::logic_error("decision visual cleanup frame is occupied");
    auto ticket =
        executor().submit(std::move(*frame), Work{.operation = ContextWork{.release_vision = id}});
    (void)wait_branch_work(ticket, true);
    vision_window.revoke(owner);
}

void SessionImpl::release_branch_point(std::unique_ptr<BranchPoint>& point) {
    if (!point) return;
    // Sequential branch execution has retired (or drained) every physical reader here.
    point->history.reset();
    if (point->state.valid()) {
        if (!state_store->release(point->state))
            throw std::logic_error("decision State still has a reader at retirement");
    }
    point.reset();
}

bool SessionImpl::release_branch_cache() {
    if (!branches) return false;
    release_branch_point(branches->root);
    branches.reset();
    return true;
}

BranchScoreResult
SessionImpl::score_branches(std::span<const BranchScoreRow> rows, const PreparationControl& control,
                            bool reuse_prefix,
                            std::shared_ptr<const qwen::PreparedPromptData> picture) {
    if (!branch_scoring_available())
        throw std::logic_error("decision scoring requires an idle Session execution boundary");
    if (!plan.branch_batch_capacity)
        throw RequestError(RequestErrorKind::Overloaded,
                           "decision scratch does not fit the existing workspace");
    const auto started = std::chrono::steady_clock::now();
    const auto check   = [&] {
        if (control.cancellation.requested())
            throw RequestError(RequestErrorKind::Cancelled, "decision request cancelled");
        if (control.deadline != std::chrono::steady_clock::time_point{} &&
            std::chrono::steady_clock::now() >= control.deadline)
            throw RequestError(RequestErrorKind::QueueTimeout, "decision deadline exceeded");
    };
    check();
    // Use the same content/geometry/mRoPE proof as ordinary image prefix reuse. Retain
    // only compact identity metadata, never the encoded image or its patch payload.
    if (!reuse_prefix ||
        (branches &&
         (branches->has_picture != bool(picture) ||
          (picture && (branches->picture.size() != picture->token_ids.size() ||
                       !branches->picture.matches(*picture, picture->token_ids.size()))))))
        (void)release_branch_cache();
    if (!branches) branches = std::make_unique<BranchCache>();
    branches->has_picture = bool(picture);
    if (picture) branches->picture.assign(*picture);
    std::shared_ptr<const qwen::VisionExecutionInput> visual;
    if (picture) {
        if (!vision_enabled || !plan.profile->config.vision || !plan.vision_max_features)
            throw std::invalid_argument("decision images require enabled Vision");
        visual = qwen::prepare_vision_input(*picture, *plan.profile->config.vision,
                                            plan.vision_max_features);
    }
    const auto batch = plan.branch_batch_capacity;
    std::vector<std::unique_ptr<BranchPoint>> owned;
    owned.reserve(1 + 2 * batch);
    const auto release = [&](BranchPoint* point) {
        const auto found = std::find_if(owned.begin(), owned.end(),
                                        [&](const auto& owner) { return owner.get() == point; });
        if (found == owned.end()) throw std::logic_error("decision point has no owner");
        release_branch_point(*found);
        owned.erase(found);
    };
    const auto cleanup = [&] {
        for (auto& point : owned) release_branch_point(point);
        owned.clear();
        if (!executor().failed()) release_branch_visual();
    };
    const auto create = [&](BranchPoint* source, std::uint32_t length, std::uint32_t row) {
        check();
        const auto frontier = source ? source->frontier : 0U;
        std::optional<context::HistoryPrefix> prefix;
        if (source && frontier)
            prefix = text_kv_histories->retain_prefix(source->history.handle(), frontier);
        const auto history =
            text_kv_histories->binding_demand(prefix ? &*prefix : nullptr, frontier, length);
        context::ContextDemand demand;
        const auto state_id   = demand.add_state({.count = 1});
        const auto history_id = demand.add_history(kTextKvPool.index, history);
        auto reserved         = directory.reserve(demand);
        if (reserved && !reserved.reservation->ready()) {
            auto maintenance = directory.prepare_maintenance(std::move(*reserved.reservation),
                                                             reserved.assessment);
            settle_branch_context(maintenance);
            reserved.reservation.emplace(maintenance.take_reservation());
        }
        if (!reserved)
            throw RequestError(RequestErrorKind::Overloaded,
                               "decision branches exceed available shared Context capacity");
        auto state = state_store->reserve_destination(reserved.reservation->take(state_id));
        if (!state) throw std::logic_error("decision lost its reserved State destination");
        context::ContextTransaction transaction(directory);
        transaction.own_state(*state);
        if (source && frontier) {
            // Finished parents are immutable copy sources, never future write destinations.
            if (state_store->role(source->state) == context::StateRole::ActiveMutable)
                state_store->freeze(source->state);
            transaction.add_state_group(source->state, 0, state_store->image_bytes());
        } else
            transaction.clear_state();
        transaction.add_binding(kTextKvPool.index,
                                text_kv_histories->prepare_binding(
                                    prefix ? &*prefix : nullptr, frontier, length,
                                    max_concurrency + row, reserved.reservation->take(history_id)));
        auto point = std::make_unique<BranchPoint>();
        settle_branch_context(transaction);
        point->history = transaction.binding(kTextKvPool.index)->commit();
        state_store->complete_initialization(*state);
        point->state    = transaction.state_destination().release();
        point->frontier = frontier;
        auto* result    = point.get();
        owned.push_back(std::move(point));
        return result;
    };
    BranchScoreResult result;
    result.logits.resize(rows.size());
    const auto forward = [&](std::span<BranchPoint* const> points,
                             std::span<const BranchScoreRow> inputs,
                             std::span<const std::uint32_t> ends, bool prefill) {
        for (;;) {
            check();
            BranchWork work{.prefill = prefill};
            std::vector<BranchPoint*> live;
            live.reserve(points.size());
            work.rows.reserve(points.size());
            context::ContextTransaction transaction(directory);
            for (std::size_t r = 0; r < points.size(); ++r) {
                auto& point = *points[r];
                if (point.frontier == ends[r]) continue;
                const auto count = std::min(prefill ? plan.branch_prefill_chunk : kBranchWidth,
                                            ends[r] - point.frontier);
                text_kv_histories->prepare_write(point.history, point.frontier,
                                                 point.frontier + count, transaction);
                live.push_back(&point);
                BranchRow row{.state   = point.state,
                              .offsets = state_store->selectors(point.state, point.state),
                              .table =
                                  text_kv_histories->execution_row(point.history.handle()).handle(),
                              .frontier = point.frontier};
                for (std::uint32_t c = 0; c < count; ++c)
                    row.tokens.push_back(inputs[r][point.frontier + c]);
                row.rope_delta = picture ? picture->rope_delta : 0;
                if (prefill && picture) {
                    for (int axis = 0; axis < 3; ++axis)
                        for (std::uint32_t c = 0; c < count; ++c) {
                            const auto p = point.frontier + c;
                            row.rope_positions.push_back(p < picture->token_ids.size()
                                                             ? picture->position_axis(axis)[p]
                                                             : static_cast<std::int32_t>(p) +
                                                                   picture->rope_delta);
                        }
                }
                work.rows.push_back(std::move(row));
            }
            if (live.empty()) return;
            settle_branch_context(transaction);
            if (visual && prefill) {
                auto& row           = work.rows.front();
                const auto end      = row.frontier + row.tokens.size();
                const auto* current = vision_window.current();
                auto planned = current ? current->plan
                                       : vision::plan_window(visual->units, row.frontier, end,
                                                             plan.vision_max_features, 1);
                if (planned) {
                    // A window can end before the chunk when the next picture exceeds the
                    // existing feature budget. Finish this window before reserving another.
                    const auto limit = std::min<std::size_t>(end, planned->consume_end);
                    const auto count = limit - row.frontier;
                    if (count < row.tokens.size()) {
                        const auto previous = row.tokens.size();
                        std::vector<std::int32_t> positions;
                        for (int axis = 0; axis < 3; ++axis)
                            positions.insert(positions.end(),
                                             row.rope_positions.begin() + axis * previous,
                                             row.rope_positions.begin() + axis * previous + count);
                        row.rope_positions = std::move(positions);
                        row.tokens.resize(count);
                    }
                    work.visual = qwen::prepare_vision_work(
                        vision_window.reserve({}, current ? nullptr : planned), visual, 1);
                }
            }
            std::vector<std::size_t> counts;
            for (const auto& row : work.rows) counts.push_back(row.tokens.size());
            (void)run_branch_work(std::move(work));
            for (std::size_t r = 0; r < live.size(); ++r) {
                auto& point = *live[r];
                state_store->complete_write(point.state, point.state);
                point.frontier += counts[r];
                text_kv_histories->commit_frontier(point.history.handle(), point.frontier);
                result.computed_tokens += counts[r];
            }
            ++result.forward_batches;
            result.largest_batch =
                std::max(result.largest_batch, static_cast<std::uint32_t>(live.size()));
        }
    };
    bool sharing = true;
    for (;;) {
        try {
            struct Group {
                std::span<const TokenId> trunk;
                std::vector<std::size_t> rows;
            };

            std::vector<Group> groups;
            for (std::size_t i = 0; i < rows.size(); ++i) {
                const auto trunk = sharing ? rows[i].prefix.first(rows[i].trunk_tokens)
                                           : std::span<const TokenId>{};
                auto group = std::find_if(groups.begin(), groups.end(), [&](const Group& value) {
                    return std::ranges::equal(value.trunk, trunk);
                });
                if (group == groups.end())
                    groups.push_back({trunk, {i}});
                else
                    group->rows.push_back(i);
            }
            auto root_tokens = groups.front().trunk;
            for (const auto& row : rows)
                root_tokens =
                    root_tokens.first(std::min<std::size_t>(root_tokens.size(), row.root_tokens));
            for (const auto& group : groups) {
                const auto matched = std::mismatch(root_tokens.begin(), root_tokens.end(),
                                                   group.trunk.begin(), group.trunk.end());
                root_tokens        = root_tokens.first(matched.first - root_tokens.begin());
            }
            if (branches->root && !std::ranges::equal(root_tokens, branches->prefix))
                release_branch_point(branches->root);
            BranchPoint* root;
            if (branches->root) {
                root = branches->root.get();
                owned.push_back(std::move(branches->root));
            } else
                root = create(nullptr, root_tokens.size(), 0);
            branches->prefix.clear();
            const std::array root_points{root};
            const std::array root_inputs{rows.front()};
            const std::array root_ends{static_cast<std::uint32_t>(root_tokens.size())};
            forward(root_points, root_inputs, root_ends, true);
            for (std::size_t begin = 0; begin < groups.size();) {
                std::vector<BranchPoint*> trunks;
                std::vector<BranchScoreRow> inputs;
                std::vector<std::uint32_t> ends;
                std::size_t end = begin;
                for (; end < groups.size() && trunks.size() < batch; ++end) {
                    try {
                        trunks.push_back(create(root, groups[end].trunk.size(), 1 + trunks.size()));
                    } catch (const RequestError& error) {
                        if (error.kind() != RequestErrorKind::Overloaded || trunks.empty()) throw;
                        break;
                    }
                    inputs.push_back(rows[groups[end].rows.front()]);
                    ends.push_back(groups[end].trunk.size());
                }
                forward(trunks, inputs, ends, false);
                std::vector<std::pair<std::size_t, std::size_t>> pending;
                for (std::size_t rank = 0;; ++rank) {
                    const auto previous = pending.size();
                    for (std::size_t g = begin; g < end; ++g)
                        if (rank < groups[g].rows.size())
                            pending.emplace_back(groups[g].rows[rank], g - begin);
                    if (pending.size() == previous) break;
                }
                for (std::size_t offset = 0; offset < pending.size();) {
                    std::vector<BranchPoint*> leaves;
                    inputs.clear();
                    ends.clear();
                    const auto first = offset;
                    while (offset < pending.size() && leaves.size() < batch) {
                        const auto [row, group] = pending[offset];
                        try {
                            leaves.push_back(
                                create(trunks[group], rows[row].size(), 1 + batch + leaves.size()));
                        } catch (const RequestError& error) {
                            if (error.kind() != RequestErrorKind::Overloaded || leaves.empty())
                                throw;
                            break;
                        }
                        inputs.push_back(rows[row]);
                        ends.push_back(rows[row].size());
                        ++offset;
                    }
                    forward(leaves, inputs, ends, false);
                    BranchWork project;
                    for (std::size_t r = 0; r < leaves.size(); ++r) {
                        auto& point = *leaves[r];
                        project.rows.push_back(
                            {.state   = point.state,
                             .offsets = state_store->selectors(point.state, point.state),
                             .table =
                                 text_kv_histories->execution_row(point.history.handle()).handle(),
                             .frontier   = point.frontier,
                             .candidates = {inputs[r].candidates.begin(),
                                            inputs[r].candidates.end()}});
                    }
                    auto scored = run_branch_work(std::move(project));
                    check();
                    for (std::size_t r = 0; r < leaves.size(); ++r) {
                        result.logits[pending[first + r].first] = std::move(scored.logits[r]);
                        release(leaves[r]);
                    }
                }
                for (auto* trunk : trunks) release(trunk);
                begin = end;
            }
            if (!root_tokens.empty()) {
                context::ContextTransaction transaction(directory);
                text_kv_histories->deactivate(root->history.handle(), transaction);
                settle_branch_context(transaction);
                auto found = std::find_if(owned.begin(), owned.end(),
                                          [&](const auto& value) { return value.get() == root; });
                branches->root = std::move(*found);
                owned.erase(found);
                branches->prefix.assign(root_tokens.begin(), root_tokens.end());
            }
            cleanup();
            std::uint64_t logical = 0;
            for (const auto& row : rows) logical += row.size();
            result.reused_tokens =
                logical > result.computed_tokens ? logical - result.computed_tokens : 0;
            result.prefill_ms = std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - started)
                                    .count();
            if (!branches->root) branches.reset();
            return result;
        } catch (const RequestError& error) {
            cleanup();
            (void)release_branch_cache();
            if (error.kind() != RequestErrorKind::Overloaded || !sharing || executor().failed() ||
                picture)
                throw;
            sharing  = false;
            branches = std::make_unique<BranchCache>();
        } catch (...) {
            cleanup();
            (void)release_branch_cache();
            throw;
        }
    }
}

} // namespace ninfer::models::qwen3_5::detail
