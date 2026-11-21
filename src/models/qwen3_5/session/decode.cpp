#include "models/qwen3_5/session/internal.h"
#include "models/common/speculative/growth.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {
std::optional<runtime::ExecutionTicket>
SessionImpl::start_decode(std::span<const SequenceHandle> members,
                          std::span<const runtime::RoundBudget> budgets,
                          std::span<const std::uint8_t> constrained) {
    if (execution) return {};
    if (members.empty() || members.size() > max_concurrency || budgets.size() != members.size() ||
        (!constrained.empty() && constrained.size() != members.size()))
        throw std::invalid_argument("decode membership is invalid");
    auto reservation = executor().try_reserve(runtime::FrameKind::Compute);
    if (!reservation) return {};
    try {
        Execution pending{.kind = OperationKind::Decode};
        pending.members.assign(members.begin(), members.end());
        pending.preparation = std::make_unique<context::ContextTransaction>(directory);
        std::vector<speculative::RoundMember> rows;
        std::vector<speculative::RowGrowth> growth;
        rows.reserve(members.size());
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
                !budgets[b].generated_tokens_remaining)
                throw std::logic_error("sequence is not decode ready");
            require_unit(l, ExecutionUnitKind::Decode, 1);
            auto hard = std::min(
                {draft_window, budgets[b].generated_tokens_remaining - 1, capacity - f - 1});
            if ((speculative_backend == SpeculativeBackend::Mtp && !f) ||
                (is_masked_draft_backend(speculative_backend) && capacity - f <= draft_window))
                hard = 0;
            rows.push_back({0, hard, hard});
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
        const auto size  = static_cast<std::uint32_t>(members.size());
        const auto shape = plan.batch_execution->decode(size, size);
        DecodeWork decode(speculative::DecodeRoundPlan::select(speculative_backend, draft_window, 1,
                                                               shape, plan.decode_path,
                                                               plan.decode_routes.get(), rows),
                          plan.decode_path);
        pending.round = decode.plan;
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
        executor().fail(error.what());
        throw;
    }
}

} // namespace ninfer::models::qwen3_5::detail
