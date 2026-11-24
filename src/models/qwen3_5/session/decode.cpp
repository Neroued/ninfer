#include "models/qwen3_5/session/internal.h"
#include "models/common/speculative/growth.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {
std::optional<runtime::ExecutionTicket>
SessionImpl::start_decode_round(std::span<const SequenceHandle> members,
                                speculative::DecodeRoundRequest selection,
                                std::span<const std::uint8_t> constrained) {
    if (execution) return {};
    if (members.empty() || members.size() > max_concurrency ||
        !selection.valid_membership(members.size()) ||
        (!constrained.empty() && constrained.size() != members.size()))
        throw std::invalid_argument("decode membership is invalid");
    auto reservation = executor().try_reserve(runtime::FrameKind::Compute);
    if (!reservation) return {};
    bool prepared              = false;
    const bool adaptive_length = draft_window && selection.uses_adaptive_length(plan.length_policy);
    const bool measure_cost    = draft_window && (adaptive_length || selection.budgets.empty());
    try {
        Execution pending{.kind = OperationKind::Decode};
        pending.members.assign(members.begin(), members.end());
        pending.preparation = std::make_unique<context::ContextTransaction>(directory);
        std::vector<speculative::RoundMember> rows;
        std::vector<speculative::AdaptiveRowInput> adaptive;
        std::vector<std::uint32_t> frontiers;
        std::vector<speculative::RowGrowth> growth;
        rows.reserve(members.size());
        if (adaptive_length) adaptive.reserve(members.size());
        if (measure_cost) frontiers.reserve(members.size());
        growth.reserve(members.size());
        const auto mark = next_lane_mark();
        for (std::size_t b = 0; b < members.size(); ++b) {
            const auto l = lane(members[b]);
            if (lane_marks[l] == mark) throw std::logic_error("duplicate sequence in decode");
            lane_marks[l]       = mark;
            const auto& state   = sequences[l];
            const auto& request = requests[l];
            const auto& base    = state.continuation.metadata;
            const auto f        = base.execution_frontier;
            if (request.lifecycle != Lifecycle::Active || state.ledger.size() != f + 1 ||
                f >= capacity)
                throw std::logic_error("sequence is not decode ready");
            require_unit(l, ExecutionUnitKind::Decode, 1);
            auto hard = std::min({draft_window, selection.draft_budget(b), capacity - f - 1});
            if ((speculative_backend == SpeculativeBackend::Mtp && !f) ||
                (is_masked_draft_backend(speculative_backend) && capacity - f <= draft_window))
                hard = 0;
            rows.push_back({0, hard, hard});
            if (measure_cost) frontiers.push_back(f);
            if (adaptive_length) adaptive.push_back({rows.back(), f, &request.speculative_state});
            auto& row = growth.emplace_back();
            if (hard) {
                row.text.emplace(*text_kv_histories, state.kv->text.handle(), base.text_kv_valid,
                                 request.permit->main_frontier);
                if (state.kv->backend)
                    row.backend.emplace(*backend_kv_histories, state.kv->backend->handle(),
                                        backend_kv_valid(state), request.permit->backend_frontier);
            }
        }
        grant_windows(directory, growth, rows);
        for (std::size_t b = 0; b < adaptive.size(); ++b) adaptive[b].member = rows[b];
        const auto size  = static_cast<std::uint32_t>(members.size());
        const auto shape = plan.batch_execution->decode(size, size);
        auto round =
            selection.select(plan.length_policy, shape, plan.decode_path, plan.decode_routes.get(),
                             cost_model, rows, adaptive, capacity);
        if (!round) return {};
        const bool narrow =
            plan.split_proposal_narrowing && adaptive_length && round->route().width > 1;
        if (narrow && is_masked_draft_backend(speculative_backend) &&
            !std::ranges::equal(round->verify_drafts(), rows, [](auto depth, const auto& row) {
                return !depth || depth == row.granted_limit;
            })) {
            // Freeze the full grant for each selected proposal row before any draws. The
            // split target may narrow within that grant before executing or adopting tokens.
            std::vector<std::uint32_t> extents(size);
            for (std::size_t i = 0; i < size; ++i)
                if (round->verify_drafts()[i]) extents[i] = rows[i].granted_limit;
            round = speculative::DecodeRoundPlan::select(speculative_backend, draft_window, 1,
                                                         shape, plan.decode_path,
                                                         plan.decode_routes.get(), rows, extents);
        }
        if (measure_cost)
            pending.cost_query =
                speculative::CostQuery::from_plan(speculative_backend, *round, frontiers, capacity);
        pending.record_online_cost = selection.record_online_cost;
        DecodeWork decode(std::move(*round), plan.decode_path);
        pending.round = decode.plan;
        if (draft_window) {
            pending.could_speculate.reserve(size);
            for (const auto& row : rows) pending.could_speculate.push_back(row.granted_limit > 0);
        }
        if (narrow) {
            auto& narrowing = decode.narrowing.emplace(
                SplitProposalNarrowingInput{.cost = cost_model, .max_context = capacity});
            narrowing.rows.reserve(adaptive.size());
            for (const auto& row : adaptive)
                narrowing.rows.push_back({row.member, row.frontier, *row.state});
        }
        prepared = true;
        // Reserve all host bookkeeping before transferring any optional right to a history.
        for (std::size_t i = 0; i < size; ++i) growth[i].prepare(pending.round->verify_drafts()[i]);
        for (auto& row : growth) row.adopt();
        decode.rows.reserve(size);
        pending.bases.reserve(size);
        pending.columns.reserve(size);
        for (std::size_t b = 0; b < size; ++b) {
            auto& state         = sequences[lane(members[b])];
            const auto& request = requests[state.lane];
            const auto width    = 1 + pending.round->verify_drafts()[b];
            pending.bases.push_back(state.continuation);
            pending.columns.push_back(width);
            ensure_sequence_kv_mapped(
                state, request.permit->main_frontier + width - 1,
                state.kv->backend ? request.permit->backend_frontier + width - 1 : 0,
                *pending.preparation);
            decode.rows.push_back(
                {execution_row(state),
                 state.ledger.back(),
                 {request.sampling_host, !constrained.empty() && constrained[b]}});
        }
        return submit_execution(std::move(pending), std::move(*reservation),
                                Work{.operation = std::move(decode)});
    } catch (const std::exception& error) {
        if (prepared) executor().fail(error.what());
        throw;
    }
}

} // namespace ninfer::models::qwen3_5::detail
