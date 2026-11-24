#include "models/glm5next/session/internal.h"

#include "models/common/speculative/growth.h"

#include <algorithm>

namespace ninfer::models::glm5next {
namespace {
using Kind = detail::ComputeTransaction::Kind;

void submit(detail::SessionImpl& session, runtime::ExecutionReservation reservation, Work work) {
    auto& tx = *session.compute;
    if (work_batch(work).members.empty()) prepare_batch(work, *session.plan.batches);
    tx.context.prepare_actions();
    work.actions = session.actions(tx.context);
    tx.ticket    = session.executor->submit(std::move(reservation), std::move(work));
    for (auto h : tx.members) session.sequence(h).computing = true;
    tx.context.mark_submitted();
    if (tx.vision_id) session.vision_window.submitted(tx.vision_id, tx.ticket);
}

} // namespace

runtime::PrefillBudgetPolicy Session::prefill_budget_policy() const noexcept {
    const auto& shape = impl_->shape;
    return {.max_tokens_global           = shape.prefill_chunk,
            .max_tokens_per_placement    = shape.prefill_chunk,
            .max_sequences_per_step      = shape.max_concurrency,
            .max_sequences_per_placement = shape.max_concurrency};
}

runtime::PrefillCandidateInfo Session::prefill_candidate(SequenceHandle handle) const {
    const auto& p = *impl_;
    const auto& s = p.sequence(handle);
    if (s.phase != detail::Phase::Prefill)
        throw std::logic_error("prefill candidate is outside prefill");
    const auto begin   = s.continuation.metadata.main_frontier;
    const auto prompt  = s.request->summary.prompt_tokens;
    const auto end     = p.unit_end(s, ExecutionUnitKind::Prefill, 0);
    const bool capture = s.capture && s.captures[s.next_capture].frontier == begin;
    const bool exclusive =
        s.request->vision && begin < end &&
        std::ranges::any_of(s.request->vision->units, [&](const models::vision::Unit& unit) {
            return unit.token_begin < end && begin < unit.end();
        });
    return {.placement         = s.continuation.metadata.placement,
            .remaining_tokens  = prompt - begin,
            .unit_max_tokens   = end - begin,
            .exclusive         = exclusive,
            .zero_token_sample = begin == prompt && !capture};
}

std::optional<runtime::ExecutionTicket> Session::start_prefill(SequenceHandle handle,
                                                               bool constrained) {
    if (impl_->compute) return {};
    const runtime::PrefillGrant grant{prefill_candidate(handle).unit_max_tokens, constrained};
    return start_prefill({&handle, 1}, {&grant, 1});
}

std::optional<runtime::ExecutionTicket>
Session::start_prefill(std::span<const SequenceHandle> members,
                       std::span<const runtime::PrefillGrant> grants) {
    auto& p = *impl_;
    if (p.compute) return {};
    if (members.empty() || members.size() > p.shape.max_concurrency ||
        grants.size() != members.size())
        throw std::logic_error("invalid prefill membership");
    std::vector<std::uint32_t> ends;
    ends.reserve(members.size());
    const auto mark         = p.next_membership_mark();
    std::uint32_t remaining = p.shape.prefill_chunk;
    for (std::size_t b = 0; b < members.size(); ++b) {
        auto& s = p.sequence(members[b]);
        if (s.phase != detail::Phase::Prefill || !s.permit ||
            s.permit->kind != ExecutionUnitKind::Prefill || s.capture_pending ||
            s.membership_mark == mark)
            throw std::logic_error("prefill has no distinct executable unit");
        s.membership_mark = mark;
        const auto info   = prefill_candidate(members[b]);
        const auto begin  = s.continuation.metadata.main_frontier;
        const auto tokens = std::min(grants[b].max_tokens, info.unit_max_tokens);
        const auto end    = std::min(s.permitted_frontier, begin + tokens);
        if ((info.exclusive && members.size() > 1) || (end == begin && !info.zero_token_sample))
            throw std::logic_error("prefill grant has no executable window");
        if (end - begin > remaining)
            throw std::logic_error("prefill grants exceed the token budget");
        remaining -= end - begin;
        ends.push_back(end);
    }
    auto reservation = p.executor->try_reserve(runtime::FrameKind::Compute);
    if (!reservation) return {};
    p.compute = std::make_unique<detail::ComputeTransaction>(Kind::Prefill, p.directory);
    auto& tx  = *p.compute;
    try {
        tx.members.assign(members.begin(), members.end());
        tx.rows.reserve(members.size());
        ForwardWork forward;
        forward.kind = ForwardKind::Prefill;
        for (std::size_t b = 0; b < members.size(); ++b) {
            auto& s            = p.sequence(members[b]);
            const auto begin   = s.continuation.metadata.main_frontier;
            const auto end     = ends[b];
            const bool capture = s.capture && s.captures[s.next_capture].frontier == end;
            if (end > begin) p.prepare_write(s, end, tx.context);
            auto row              = p.work_row(s, end - begin);
            row.sample            = end == s.request->summary.prompt_tokens && !capture;
            row.constrained       = row.sample && grants[b].constrained;
            row.sampling_position = end;
            if (row.sample) forward.sample = SampleKind::FirstToken;
            tx.rows.push_back(std::move(row));
            if (auto visual = p.prepare_visual(s, begin, end)) {
                tx.vision_id   = visual->window.id;
                forward.visual = std::move(visual);
            }
        }
        forward.rows = tx.rows;
        submit(p, std::move(*reservation), {{}, std::move(forward)});
        return tx.ticket;
    } catch (const std::exception& error) {
        p.executor->fail(error.what());
        throw;
    }
}

std::optional<runtime::ExecutionTicket> Session::start_replay(SequenceHandle handle) {
    auto& p = *impl_;
    auto& s = p.sequence(handle);
    if (p.compute) return {};
    if (s.phase != detail::Phase::Replay || !s.permit || s.capture_pending)
        throw std::logic_error("replay has no executable unit");
    const auto begin = s.continuation.metadata.main_frontier;
    const auto end   = p.window_end(s, begin, std::min(s.permitted_frontier, s.replay_target));
    if (end <= begin) throw std::logic_error("replay has no missing input window");
    auto reservation = p.executor->try_reserve(runtime::FrameKind::Compute);
    if (!reservation) return {};
    p.compute = std::make_unique<detail::ComputeTransaction>(Kind::Replay, p.directory);
    auto& tx  = *p.compute;
    tx.members.push_back(handle);
    try {
        if (end > begin) p.prepare_write(s, end, tx.context);
        tx.rows.push_back(p.work_row(s, end - begin));
        ForwardWork forward;
        forward.kind   = ForwardKind::Replay;
        forward.rows   = tx.rows;
        forward.visual = p.prepare_visual(s, begin, end);
        if (forward.visual) tx.vision_id = forward.visual->window.id;
        submit(p, std::move(*reservation), {{}, std::move(forward)});
        return tx.ticket;
    } catch (const std::exception& error) {
        p.executor->fail(error.what());
        throw;
    }
}

std::optional<runtime::ExecutionTicket>
Session::start_decode(std::span<const SequenceHandle> members,
                      std::span<const runtime::RoundBudget> budgets,
                      std::span<const std::uint8_t> constrained) {
    return start_decode_round(members, {.budgets = budgets}, constrained);
}

std::optional<runtime::ExecutionTicket>
Session::start_decode_round(std::span<const SequenceHandle> members,
                            speculative::DecodeRoundRequest selection,
                            std::span<const std::uint8_t> constrained) {
    auto& p = *impl_;
    if (p.compute) return {};
    if (members.empty() || members.size() > p.shape.max_concurrency ||
        !selection.valid_membership(members.size()) ||
        (!constrained.empty() && constrained.size() != members.size()))
        throw std::logic_error("invalid decode members");
    auto reservation = p.executor->try_reserve(runtime::FrameKind::Compute);
    if (!reservation) return {};
    auto pending  = std::make_unique<detail::ComputeTransaction>(Kind::Ordinary, p.directory);
    auto& tx      = *pending;
    bool prepared = false;
    try {
        tx.members.assign(members.begin(), members.end());
        const auto& options = p.definition->options();
        const bool adaptive_length =
            p.shape.draft_tokens && selection.uses_adaptive_length(p.shape.length_policy);
        const bool measure_cost =
            p.shape.draft_tokens && (adaptive_length || selection.budgets.empty());
        std::vector<speculative::RoundMember> limits;
        std::vector<speculative::AdaptiveRowInput> adaptive;
        std::vector<std::uint32_t> frontiers;
        limits.reserve(members.size());
        if (adaptive_length) adaptive.reserve(members.size());
        if (measure_cost) frontiers.reserve(members.size());
        std::vector<speculative::RowGrowth> growth(members.size());
        const auto mark = p.next_membership_mark();
        for (std::size_t b = 0; b < members.size(); ++b) {
            auto& s      = p.sequence(members[b]);
            const auto f = s.continuation.metadata.main_frontier;
            if (s.phase != detail::Phase::Active || !s.permit ||
                s.permit->kind != ExecutionUnitKind::Decode || f >= p.shape.max_context ||
                s.permitted_frontier < f + 1 || s.ledger.size() != f + 1 ||
                s.membership_mark == mark)
                throw std::logic_error("decode row is not executable");
            s.membership_mark      = mark;
            const auto room        = p.shape.max_context - f - 1;
            const bool can_propose = options.mtp() ? f != 0 : p.shape.draft_tokens <= room;
            const auto hard =
                can_propose ? std::min({p.shape.draft_tokens, room, selection.draft_budget(b)})
                            : 0U;
            limits.push_back({s.continuation.metadata.placement.value, hard, hard});
            if (measure_cost) frontiers.push_back(f);
            if (adaptive_length) adaptive.push_back({limits.back(), f, &s.speculative_state});
            if (hard) {
                growth[b].text.emplace(p.directory.pool(0).histories(), s.histories[0].handle(), f,
                                       f + 1);
                if (options.mtp())
                    growth[b].backend.emplace(
                        p.directory.pool(1).histories(), s.histories[1].handle(),
                        *s.continuation.metadata.mtp_frontier, mtp_history_frontier(f + 1));
            }
        }
        speculative::grant_windows(p.directory, growth, limits);
        for (std::size_t b = 0; b < adaptive.size(); ++b) adaptive[b].member = limits[b];
        for (std::size_t b = 0; b < members.size(); ++b) {
            auto& s         = p.sequence(members[b]);
            auto row        = p.work_row(s, 1);
            row.constrained = !constrained.empty() && constrained[b];
            tx.rows.push_back(std::move(row));
        }
        auto batch      = plan_batch(tx.rows, *p.plan.batches, true);
        const auto path = p.plan.decode_routes ? ExecutionPath::Graph : ExecutionPath::Eager;
        tx.round =
            selection.select(p.shape.length_policy, batch.shape, path, p.plan.decode_routes.get(),
                             p.cost_model, limits, adaptive, p.shape.max_context);
        if (!tx.round) return {};
        if (p.shape.draft_tokens) {
            tx.could_speculate.reserve(limits.size());
            for (const auto& row : limits) tx.could_speculate.push_back(row.granted_limit > 0);
        }
        if (measure_cost)
            tx.cost_query = speculative::CostQuery::from_plan(p.shape.speculative, *tx.round,
                                                              frontiers, p.shape.max_context);
        tx.record_online_cost = selection.record_online_cost;
        prepared              = true;
        const auto extents    = tx.round->verify_drafts();
        for (std::size_t b = 0; b < members.size(); ++b) growth[b].prepare(extents[b]);
        for (auto& row : growth) row.adopt();
        for (std::size_t b = 0; b < members.size(); ++b) {
            auto& s = p.sequence(members[b]);
            p.prepare_write(s, s.continuation.metadata.main_frontier + 1 + extents[b], tx.context);
        }
        auto geometry =
            p.shape.draft_tokens
                ? std::make_shared<const SpeculativeExecutionGeometry>(tx.round->geometry(path))
                : nullptr;
        Work work;
        if (tx.round->route().width > 1) {
            tx.kind        = Kind::Speculative;
            work.operation = SpeculativeWork{
                tx.rows, {extents.begin(), extents.end()}, std::move(batch), std::move(geometry)};
        } else {
            work.operation = ForwardWork{ForwardKind::Ordinary, SampleKind::NextToken, tx.rows, {},
                                         std::move(batch),      std::move(geometry)};
        }
        p.compute = std::move(pending);
        submit(p, std::move(*reservation), std::move(work));
        return tx.ticket;
    } catch (const std::exception& error) {
        if (prepared) p.executor->fail(error.what());
        throw;
    }
}

std::optional<runtime::ExecutionTicket>
Session::start_control(std::span<const SequenceHandle> members, std::span<const TokenId> tokens,
                       std::uint32_t stride) {
    auto& p = *impl_;
    if (p.compute) return {};
    if (members.empty() || members.size() > p.shape.max_concurrency || !stride ||
        stride > p.definition->resources().text.control_tokens->size() ||
        tokens.size() != members.size() * stride)
        throw std::logic_error("invalid forced control window");
    const auto mark = p.next_membership_mark();
    for (std::size_t b = 0; b < members.size(); ++b) {
        auto& s = p.sequence(members[b]);
        if (s.membership_mark == mark) throw std::logic_error("duplicate control member");
        s.membership_mark = mark;
        if (s.phase != detail::Phase::Active || !s.permit ||
            s.permit->kind != ExecutionUnitKind::Control ||
            s.ledger.size() != s.continuation.metadata.main_frontier + 1 ||
            s.continuation.metadata.main_frontier + stride > s.permitted_frontier)
            throw std::logic_error("forced control row is not executable");
    }
    for (const auto id : tokens)
        if (id < 0 || std::uint32_t(id) >= p.definition->resources().text.public_token_count)
            throw std::invalid_argument("forced token is outside vocabulary");
    auto reservation = p.executor->try_reserve(runtime::FrameKind::Compute);
    if (!reservation) return {};
    p.compute = std::make_unique<detail::ComputeTransaction>(Kind::Control, p.directory);
    auto& tx  = *p.compute;
    tx.members.assign(members.begin(), members.end());
    try {
        for (std::size_t b = 0; b < members.size(); ++b) {
            auto& s = p.sequence(members[b]);
            p.prepare_write(s, s.continuation.metadata.main_frontier + stride, tx.context);
            auto row               = p.work_row(s, 1);
            const auto forced      = tokens.subspan(b * stride, stride);
            row.control_last_token = forced.back();
            row.ids.insert(row.ids.end(), forced.begin(), forced.end() - 1);
            tx.rows.push_back(std::move(row));
        }
        submit(p, std::move(*reservation),
               {{}, ForwardWork{ForwardKind::Control, SampleKind::None, tx.rows}});
        return tx.ticket;
    } catch (const std::exception& error) {
        p.executor->fail(error.what());
        throw;
    }
}

} // namespace ninfer::models::glm5next

namespace ninfer::models::glm5next {

ResourceReservation detail::SessionImpl::reserve_units(std::span<const ExecutionUnit> units) {
    auto& p = *this;
    if (p.directory.maintenance_pending())
        return {.assessment = {.status = context::ContextAvailability::Waiting}};

    struct Need {
        detail::Sequence* sequence;
        std::uint32_t end;
        std::array<std::uint32_t, 2> additional{};
        std::array<context::ContextDemandId, 2> targets;
        std::array<context::RecordReservation, 2> prepared;
    };

    std::vector<Need> needed;
    ContextDemand demand;
    const auto mark = p.next_membership_mark();
    for (const auto& unit : units) {
        auto& sequence = p.sequence(unit.sequence);
        if ((unit.kind == ExecutionUnitKind::Prefill || unit.kind == ExecutionUnitKind::Replay) &&
            p.unit_end(sequence, unit.kind, unit.tokens) ==
                sequence.continuation.metadata.main_frontier &&
            sequence.continuation.metadata.main_frontier < sequence.request->summary.prompt_tokens)
            return {.status = runtime::UnitReservationStatus::ExecutionUnavailable};
        if (sequence.membership_mark == mark)
            throw std::logic_error("execution reservation repeats a sequence");
        sequence.membership_mark = mark;
        Need need{&sequence,
                  std::max(sequence.recovery_peak, p.unit_end(sequence, unit.kind, unit.tokens))};
        if (need.end > p.shape.max_context) throw std::out_of_range("execution exceeds context");
        for (std::uint32_t pool = 0; pool < p.pool_count(); ++pool) {
            const auto& histories = p.directory.pool(pool).histories();
            const auto history    = sequence.histories[pool].handle();
            const auto begin      = pool ? sequence.continuation.metadata.mtp_frontier.value()
                                         : sequence.continuation.metadata.main_frontier;
            const auto end        = pool ? mtp_history_frontier(need.end) : need.end;
            const auto pages      = histories.write_pages(history, begin, end);
            const auto held       = histories.reserved_growth_pages(history);
            need.additional[pool] = pages > held ? pages - held : 0;
            need.targets[pool]    = demand.add_growth(histories, history, need.additional[pool]);
        }
        needed.push_back(std::move(need));
    }
    auto result = p.directory.reserve(demand);
    if (!result || !result.reservation->ready()) return {.assessment = result.assessment};
    for (auto& need : needed)
        for (std::uint32_t pool = 0; pool < p.pool_count(); ++pool) {
            need.prepared[pool] = result.reservation->take(need.targets[pool]);
            if (need.additional[pool]) {
                p.directory.pool(pool).histories().prepare_growth(
                    need.sequence->histories[pool].handle(), need.prepared[pool]);
            }
        }
    for (std::size_t i = 0; i < units.size(); ++i) {
        auto& sequence = *needed[i].sequence;
        for (std::uint32_t pool = 0; pool < p.pool_count(); ++pool)
            if (needed[i].additional[pool])
                p.directory.pool(pool).histories().adopt_growth(
                    sequence.histories[pool].handle(), std::move(needed[i].prepared[pool]));
        sequence.permit             = units[i];
        sequence.permitted_frontier = needed[i].end;
    }
    return {.status = runtime::UnitReservationStatus::Granted, .assessment = result.assessment};
}

ResourceReservation Session::reserve_units(std::span<const ExecutionUnit> units) {
    return impl_->reserve_units(units);
}

void Session::release_units(std::span<const SequenceHandle> handles) noexcept {
    try {
        for (const auto h : handles)
            if (impl_->valid(h) && !execution_pending(h)) impl_->settle(impl_->sequence(h));
    } catch (...) { std::terminate(); }
}

} // namespace ninfer::models::glm5next
