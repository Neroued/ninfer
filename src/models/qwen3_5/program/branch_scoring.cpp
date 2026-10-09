#include "models/qwen3_5/program/branch_scoring.h"
#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/execution_context.h"
#include "models/qwen3_5/execution/linear.h"
#include "core/device.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>

namespace ninfer::models::qwen3_5::detail {

BranchStorage::BranchStorage(DeviceSpan backing, const BranchLayout& layout,
                             qwen3_5::PagedKVCache& shared_kv, LogicalKVPageStore& pages)
    : kv(backing, layout.tables, shared_kv), batch_capacity(layout.batch_capacity),
      states(backing, layout.states), records(backing, layout.records),
      fold(records, states.all_layers_view()),
      addresses(pages, kv.execution_tables(), 1 + 2 * batch_capacity,
                kv.execution_tables().logical_page_capacity()),
      ids(layout.ids.bind(backing)), positions(layout.positions.bind(backing)),
      counts(layout.counts.bind(backing)), rows(layout.rows.bind(backing)),
      slots(layout.slots.bind(backing)), hidden(layout.hidden.bind(backing)),
      tails(layout.tails.bind(backing)), logits(layout.logits.bind(backing)),
      work(layout.scratch.bind(backing)) {
    if (batch_capacity < 1 || batch_capacity > kBranchBatch) {
        throw std::invalid_argument("decision batch capacity must be in [1,8]");
    }
}

BranchStorage::~BranchStorage() {
    if (cached_address) { (void)addresses.release_after_deactivate(*cached_address); }
}

void BranchStorage::release_cache() {
    if (cached_address && !addresses.release_after_deactivate(*cached_address)) {
        throw std::logic_error("branch root cache release failed");
    }
    cached_address.reset();
    cached_prefix.clear();
}

bool ProgramImpl::release_branch_cache() {
    if (!branches) { return false; }
    const bool released_pages = branches->cached_address.has_value();
    branches->release_cache();
    branches.reset();
    return released_pages;
}

BranchScoreResult ProgramImpl::score_branches(std::span<const BranchScoreRow> rows,
                                              const PreparationControl& control) {
    if (!decision_scoring || has_context_transaction() || pending_transaction_) {
        throw std::logic_error("branch scoring requires its enabled, stable Program boundary");
    }
    if (!workspace_plan.branches) {
        throw RequestError(RequestErrorKind::Overloaded,
                           "decision scratch does not fit the existing runtime workspace");
    }
    if (!branches) {
        branches = std::make_unique<BranchStorage>(
            DeviceSpan{workspace_storage.base(), workspace_plan.general_capacity},
            *workspace_plan.branches, decoder->text_kv, *text_kv_pages);
    }
    auto& b                 = *branches;
    auto& work              = b.work;
    auto& kv                = b.kv;
    const auto branch_chunk = workspace_plan.branches->prefill_chunk;
    mark_workspace_usage(workspace_plan.branches->bytes);
    const auto started = std::chrono::steady_clock::now();
    const auto check   = [&] {
        if (control.cancellation.requested()) {
            throw RequestError(RequestErrorKind::Cancelled, "decision request cancelled");
        }
        if (control.deadline != std::chrono::steady_clock::time_point{} &&
            std::chrono::steady_clock::now() >= control.deadline) {
            throw RequestError(RequestErrorKind::QueueTimeout, "decision deadline exceeded");
        }
    };
    const auto pages_for = [](std::size_t n) -> std::uint32_t {
        return static_cast<std::uint32_t>((n + kPagedKVPageSize - 1) / kPagedKVPageSize);
    };

    struct Point {
        KVAddressSpaceHandle address;
        std::uint32_t frontier = 0;
        std::int32_t slot      = 0;
    };

    std::vector<KVAddressSpaceHandle> owned;
    owned.reserve(1 + 2 * b.batch_capacity);
    const auto release = [&](Point& point) {
        if (!point.address.valid()) { return; }
        if (!b.addresses.release_after_deactivate(point.address)) {
            throw std::logic_error("branch KV release failed");
        }
        std::erase(owned, point.address);
        point.address = {};
    };
    const auto cleanup = [&] {
        for (auto address : owned) {
            if (!b.addresses.release_after_deactivate(address)) {
                throw std::logic_error("branch transaction leaked KV ownership");
            }
        }
        owned.clear();
        work.reset();
    };
    const auto copy_tail = [&](std::int32_t from, std::int32_t to) {
        auto src = b.tails.slice(1, from, 1), dst = b.tails.slice(1, to, 1);
        CUDA_CHECK(cudaMemcpyAsync(dst.data, src.data, src.bytes(), cudaMemcpyDeviceToDevice,
                                   device.stream));
    };
    const auto create = [&](std::uint32_t length, std::int32_t row, std::int32_t slot) {
        if (pages_for(length) > kv.page_pool().available_pages()) {
            throw RequestError(RequestErrorKind::Overloaded,
                               "decision request exceeds available shared KV capacity");
        }
        auto address = b.addresses.create_active(pages_for(length), row, device.stream);
        if (!address) { throw std::logic_error("branch address capacity exhausted"); }
        owned.push_back(*address);
        b.states.zero_slot(slot, device.stream);
        return Point{*address, 0, slot};
    };
    const auto fork = [&](const Point& source, std::uint32_t length, std::int32_t row,
                          std::int32_t slot) {
        if (!source.frontier) { return create(length, row, slot); }
        auto address = b.addresses.create_inactive();
        if (!address) { throw std::logic_error("branch address capacity exhausted"); }
        owned.push_back(*address);
        auto reservation =
            b.addresses.prepare_prefix_fork(source.address, *address, source.frontier,
                                            pages_for(length) - pages_for(source.frontier), row);
        if (reservation.needs_tail_copy()) {
            (void)kv.page_pool().copy_page(b.addresses.prefix_fork_tail_source(reservation),
                                           b.addresses.prefix_fork_tail_destination(reservation),
                                           device.stream);
        }
        b.addresses.commit_prefix_fork(std::move(reservation), device.stream);
        b.states.copy_slot(source.slot, slot, device.stream);
        copy_tail(source.slot, slot);
        return Point{*address, source.frontier, slot};
    };
    BranchScoreResult result;
    result.logits.resize(rows.size());

    // Execute heterogeneous suffix lengths together. Inactive/padded columns have no KV effect;
    // ReplaySSM folds only each real prefix. The saved tail is BEFORE padding, never after it.
    const auto forward = [&](std::vector<Point>& points, std::span<const BranchScoreRow> inputs,
                             std::span<const std::uint32_t> ends) {
        std::array<std::int32_t, kBranchBatch * kBranchWidth> ids{}, positions{};
        std::array<std::int32_t, kBranchBatch> lengths{}, table_rows{}, state_slots{};
        std::array<ops::GdnReplayFoldRow, kBranchBatch> folds{};
        for (;;) {
            check();
            std::vector<std::size_t> live;
            for (std::size_t r = 0; r < points.size(); ++r) {
                if (points[r].frontier < ends[r]) { live.push_back(r); }
            }
            if (live.empty()) {
                // Even zero-suffix forks may have queued KV-table uploads from pinned shadows.
                // Drain those before the caller recycles execution rows for the next level.
                device.synchronize();
                break;
            }
            const auto count = static_cast<std::int32_t>(live.size());
            auto id_view     = b.ids.slice(1, 0, count);
            auto pos_view    = b.positions.slice(1, 0, count);
            auto count_view  = b.counts.slice(0, 0, count);
            auto row_view    = b.rows.slice(0, 0, count);
            auto slot_view   = b.slots.slice(0, 0, count);
            auto hidden_view = b.hidden.slice(2, 0, count);
            for (std::int32_t r = 0; r < count; ++r) {
                const auto index = live[r];
                auto& p          = points[index];
                lengths[r]       = static_cast<std::int32_t>(
                    std::min<std::uint32_t>(kBranchWidth, ends[index] - p.frontier));
                table_rows[r]  = b.addresses.bound_row(p.address);
                state_slots[r] = p.slot;
                folds[r]       = {p.slot, p.slot, lengths[r]};
                b.addresses.ensure_mapped_to_tokens(p.address, p.frontier + lengths[r],
                                                    device.stream);
                for (std::int32_t t = 0; t < kBranchWidth; ++t) {
                    // Invalid columns use an in-range token and position, and are masked by Ops.
                    const auto offset = p.frontier + std::min(t, std::max(0, lengths[r] - 1));
                    ids[r * kBranchWidth + t] = inputs[index][offset];
                    positions[r * kBranchWidth + t] =
                        static_cast<std::int32_t>(std::min<std::uint32_t>(offset, capacity - 1));
                }
            }
            const auto upload = [&](Tensor& dst, const auto& src) {
                CUDA_CHECK(cudaMemcpyAsync(dst.data, src.data(), dst.bytes(),
                                           cudaMemcpyHostToDevice, device.stream));
            };
            upload(id_view, ids);
            upload(pos_view, positions);
            upload(count_view, lengths);
            upload(row_view, table_rows);
            upload(slot_view, state_slots);
            execution::TextContext card(device, parameters, work, {}, b.states, io, prefill_hidden,
                                        branch_chunk, 0, {}, &kv);
            card.set_gdn_state_action(execution::GdnStateAction::RecordForReplay, &b.records);
            card.branch_forward_batch(id_view, pos_view, count_view, row_view, slot_view,
                                      {1, capacity}, hidden_view);
            b.fold.execute(std::span(folds).first(count), device.stream);
            for (std::int32_t r = 0; r < count; ++r) {
                auto source = hidden_view.slice(2, r, 1).slice(1, lengths[r] - 1, 1);
                auto& point = points[live[r]];
                auto tail   = b.tails.slice(1, point.slot, 1);
                CUDA_CHECK(cudaMemcpyAsync(tail.data, source.data, tail.bytes(),
                                           cudaMemcpyDeviceToDevice, device.stream));
                point.frontier += lengths[r];
                b.addresses.commit_frontier(point.address, point.frontier);
                result.computed_tokens += lengths[r];
            }
            ++result.forward_batches;
            result.largest_batch =
                std::max(result.largest_batch, static_cast<std::uint32_t>(count));
            // Host ingress arrays are reused only after their asynchronous reads complete.
            device.synchronize();
        }
    };

    try {
        check();

        struct Group {
            std::span<const TokenId> trunk;
            std::vector<std::size_t> rows;
        };

        std::vector<Group> groups;
        for (std::size_t i = 0; i < rows.size(); ++i) {
            const auto trunk = rows[i].prefix.first(rows[i].trunk_tokens);
            auto it          = std::find_if(groups.begin(), groups.end(), [&](const Group& g) {
                return std::ranges::equal(g.trunk, trunk);
            });
            if (it == groups.end()) {
                groups.push_back({trunk, {i}});
            } else {
                it->rows.push_back(i);
            }
        }
        auto root_tokens = groups.front().trunk;
        for (const auto& row : rows) {
            root_tokens =
                root_tokens.first(std::min<std::size_t>(root_tokens.size(), row.root_tokens));
        }
        for (const auto& group : groups) {
            const auto n = std::mismatch(root_tokens.begin(), root_tokens.end(),
                                         group.trunk.begin(), group.trunk.end())
                               .first -
                           root_tokens.begin();
            root_tokens  = root_tokens.first(n);
        }
        if (b.cached_address && !std::ranges::equal(root_tokens, b.cached_prefix)) {
            b.release_cache();
        }
        const auto reusable_pages  = b.cached_address ? pages_for(b.cached_prefix.size()) : 0;
        const std::uint64_t budget = kv.page_pool().available_pages() + reusable_pages;
        for (const auto& group : groups) {
            const auto trunk_pages =
                pages_for(group.trunk.size()) - root_tokens.size() / kPagedKVPageSize;
            bool fits = true;
            for (auto row : group.rows) {
                const auto leaf_pages =
                    pages_for(rows[row].size()) - group.trunk.size() / kPagedKVPageSize;
                fits &= pages_for(root_tokens.size()) + trunk_pages + leaf_pages <= budget;
            }
            if (!fits) {
                // Reuse is optional. Near capacity, score from an empty root instead of
                // requiring spare COW pages beside a full-length predictor. The token input
                // and probability calculation are unchanged; only prefill reuse is lost.
                b.release_cache();
                groups = {{}};
                for (std::size_t i = 0; i < rows.size(); ++i) { groups.front().rows.push_back(i); }
                root_tokens = {};
                break;
            }
        }
        Point root;
        if (b.cached_address && std::ranges::equal(root_tokens, b.cached_prefix)) {
            root = {*b.cached_address, static_cast<std::uint32_t>(root_tokens.size()), 0};
            owned.push_back(root.address);
            b.cached_address.reset();
        } else {
            b.release_cache();
            root = create(static_cast<std::uint32_t>(root_tokens.size()), 0, 0);
        }
        b.cached_prefix.clear();
        while (root.frontier < root_tokens.size()) {
            check();
            const auto length =
                std::min<std::uint32_t>(branch_chunk, root_tokens.size() - root.frontier);
            b.addresses.ensure_mapped_to_tokens(root.address, root.frontier + length,
                                                device.stream);
            set_device_i32(io.text_kv_table_row, b.addresses.bound_row(root.address));
            execution::PrefillContext state{
                {device, parameters, work, b.states, nullptr, io, prefill_hidden, branch_chunk,
                 ProposalHead::Full},
                kv.execution_view(b.addresses.execution_row(root.address)),
                {},
                kv,
                nullptr,
                nullptr,
                root.frontier,
                nullptr,
                nullptr,
                0,
                0,
                0,
                0,
                nullptr};
            const auto step =
                execution::prefill_text_chunk(state, root_tokens, length, std::nullopt, false);
            if (!step.processed_tokens) {
                throw std::logic_error("branch prefix made no progress");
            }
            root.frontier += step.processed_tokens;
            b.addresses.commit_frontier(root.address, root.frontier);
            auto tail   = b.tails.slice(1, 0, 1);
            auto hidden = prefill_hidden.slice(1, step.processed_tokens - 1, 1);
            CUDA_CHECK(cudaMemcpyAsync(tail.data, hidden.data, tail.bytes(),
                                       cudaMemcpyDeviceToDevice, device.stream));
            result.computed_tokens += step.processed_tokens;
            ++result.forward_batches;
            result.largest_batch = std::max(result.largest_batch, 1U);
        }
        if (b.addresses.active(root.address)) { b.addresses.deactivate(root.address); }
        for (std::size_t begin = 0; begin < groups.size();) {
            std::size_t end         = begin;
            std::uint64_t cost      = 0;
            std::uint32_t leaf_cost = 0;
            const auto available    = kv.page_pool().available_pages();
            // Budget actual trunk growth and at least one candidate, including partial-page
            // COW. Chat allocations/reservations remain owned and cannot be borrowed.
            while (end < groups.size() &&
                   end - begin < static_cast<std::size_t>(b.batch_capacity)) {
                const auto extra =
                    pages_for(groups[end].trunk.size()) - root.frontier / kPagedKVPageSize;
                auto next_leaf = leaf_cost;
                for (auto row : groups[end].rows) {
                    next_leaf =
                        std::max(next_leaf, pages_for(rows[row].size()) -
                                                static_cast<std::uint32_t>(
                                                    groups[end].trunk.size() / kPagedKVPageSize));
                }
                if (cost + extra + next_leaf > available) {
                    if (end == begin) {
                        throw RequestError(RequestErrorKind::Overloaded,
                                           "decision branches exceed available shared KV capacity");
                    }
                    break;
                }
                cost += extra;
                leaf_cost = next_leaf;
                ++end;
            }
            std::vector<Point> trunks;
            std::vector<BranchScoreRow> trunk_inputs;
            std::vector<std::uint32_t> trunk_ends;
            for (std::size_t g = begin; g < end; ++g) {
                auto size = static_cast<std::uint32_t>(groups[g].trunk.size());
                trunks.push_back(fork(root, size, g - begin, 1 + g - begin));
                // Empty trunks have no forward columns; provide a valid padding token source.
                trunk_inputs.push_back(rows[groups[g].rows.front()]);
                trunk_ends.push_back(size);
            }
            forward(trunks, trunk_inputs, trunk_ends);
            for (auto& trunk : trunks) { b.addresses.deactivate(trunk.address); }
            std::vector<std::pair<std::size_t, std::size_t>> pending;
            // Round-robin contexts so small field sets still share a single target traversal.
            for (std::size_t rank = 0;; ++rank) {
                const auto previous = pending.size();
                for (std::size_t g = begin; g < end; ++g) {
                    if (rank < groups[g].rows.size()) {
                        pending.emplace_back(groups[g].rows[rank], g - begin);
                    }
                }
                if (pending.size() == previous) { break; }
            }
            for (std::size_t offset = 0; offset < pending.size();) {
                std::vector<Point> leaves;
                std::vector<BranchScoreRow> inputs;
                std::vector<std::uint32_t> ends;
                const auto first = offset;
                while (offset < pending.size() &&
                       leaves.size() < static_cast<std::size_t>(b.batch_capacity)) {
                    const auto [row, group] = pending[offset];
                    const auto& source      = trunks[group];
                    const auto needed =
                        pages_for(rows[row].size()) - source.frontier / kPagedKVPageSize;
                    if (needed > kv.page_pool().available_pages()) {
                        if (leaves.empty()) {
                            throw RequestError(
                                RequestErrorKind::Overloaded,
                                "decision branches exceed available shared KV capacity");
                        }
                        break;
                    }
                    const auto lane = static_cast<std::int32_t>(leaves.size());
                    leaves.push_back(
                        fork(source, rows[row].size(), lane, 1 + b.batch_capacity + lane));
                    inputs.push_back(rows[row]);
                    ends.push_back(rows[row].size());
                    ++offset;
                }
                forward(leaves, inputs, ends);
                work.reset();
                auto hidden =
                    work.alloc(DType::BF16, {dimension(parameters.model.config().text.hidden_size),
                                             static_cast<std::int32_t>(leaves.size())});
                for (std::size_t i = 0; i < leaves.size(); ++i) {
                    auto tail = b.tails.slice(1, leaves[i].slot, 1);
                    auto dst  = hidden.slice(1, i, 1);
                    CUDA_CHECK(cudaMemcpyAsync(dst.data, tail.data, tail.bytes(),
                                               cudaMemcpyDeviceToDevice, device.stream));
                }
                auto logits = b.logits.slice(1, 0, leaves.size());
                execution::project(hidden, parameters.text.output_head, logits, work,
                                   device.stream);
                std::vector<std::vector<std::uint16_t>> readback(leaves.size());
                std::vector<TokenId> lows;
                for (std::size_t i = 0; i < leaves.size(); ++i) {
                    auto [lo, hi] = std::minmax_element(inputs[i].candidates.begin(),
                                                        inputs[i].candidates.end());
                    lows.push_back(*lo);
                    readback[i].resize(*hi - *lo + 1);
                    auto source = logits.slice(1, i, 1).slice(0, *lo, readback[i].size());
                    CUDA_CHECK(cudaMemcpyAsync(readback[i].data(), source.data, source.bytes(),
                                               cudaMemcpyDeviceToHost, device.stream));
                }
                device.synchronize();
                check();
                for (std::size_t i = 0; i < leaves.size(); ++i) {
                    auto& output = result.logits[pending[first + i].first];
                    for (auto token : inputs[i].candidates) {
                        output.push_back(std::bit_cast<float>(
                            std::uint32_t(readback[i][token - lows[i]]) << 16));
                    }
                    release(leaves[i]);
                }
            }
            for (auto& trunk : trunks) { release(trunk); }
            begin = end;
        }
        if (!root_tokens.empty()) {
            b.cached_prefix.assign(root_tokens.begin(), root_tokens.end());
            b.cached_address = root.address;
            std::erase(owned, root.address);
        } else {
            release(root);
        }
        std::uint64_t logical = 0;
        for (const auto& row : rows) { logical += row.size(); }
        result.reused_tokens = logical - result.computed_tokens;
        result.prefill_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started)
                .count();
        cleanup();
        return result;
    } catch (...) {
        try {
            device.synchronize();
        } catch (...) {}
        cleanup();
        throw;
    }
}

} // namespace ninfer::models::qwen3_5::detail
