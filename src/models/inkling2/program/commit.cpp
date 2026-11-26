#include "models/inkling2/program/internal.h"

#include "core/transfer.h"

#include <algorithm>

namespace ninfer::models::inkling2::detail {

void RankProgramImpl::commit(runtime::ExecutionTicket ticket, runtime::CommandSequence command,
                             Commit&& commit) {
    prepare_commit(ticket, commit);
    auto& f = *compute;
    timer.resolve();
    const auto* forward = std::get_if<ForwardWork>(&f.request.work.operation);
    if (forward && forward->sample != SampleKind::None) {
        const auto& io     = frame;
        const auto members = f.sampling_members();
        const int b        = extent(members.size());
        auto* values       = static_cast<std::int32_t*>(host(io.rows.count_columns).data);
        bool any           = false;
        for (int i = 0; i < b; ++i) {
            const auto logical   = members[i];
            const auto& sampling = forward->rows[logical].sampling;
            values[i]            = (sampling.presence_penalty || sampling.frequency_penalty)
                                       ? static_cast<std::int32_t>(commit.rows[logical].accepted)
                                       : 0;
            any |= values[i] != 0;
        }
        if (any) {
            upload_region(io.rows.count_columns.region, b * sizeof(std::int32_t));
            auto counts = resources.tensor(*io.token_counts);
            ops::increment_token_counts_batch(
                resources.packed(*io.sampled_ids, {1, b}),
                resources.tensor(io.rows.count_columns).slice(0, 0, b),
                resources.tensor(io.rows.count_slots).slice(0, 0, b), counts, cuda);
        }
    }
    resources.compute_actions->submit(
        local_actions(commit.actions, parameters.profile->local.rank));
    timer.record_retirement();
    compute_done.record(cuda.stream());
    f.retirement = command;
    f.phase      = ComputePhase::Retiring;
}

void RankProgramImpl::control_counts() {
    auto& f            = *compute;
    const auto& rows   = std::get<ForwardWork>(f.request.work.operation).rows;
    const auto& io     = frame;
    const auto cap     = static_cast<std::size_t>(resources.tensor(io.main.ids).numel());
    auto* ids          = static_cast<TokenId*>(host(io.main.ids).data);
    std::uint32_t used = 0;
    while (f.control_row < rows.size() && used < cap) {
        const auto& r = rows[f.control_row];
        const auto n  = std::min<std::size_t>(cap - used, r.ids.size() - f.control_offset);
        if (n) {
            ControlTokens{r.ids, *r.control_last_token}.copy({ids + used, n}, f.control_offset);
            auto input = resources.tensor(io.main.ids).slice(0, used, n);
            copy_bytes_async({input.data, input.bytes(), cuda.device()},
                             {ids + used, input.bytes(), Device::cpu(), true}, cuda);
            auto counts =
                resources.tensor(*io.token_counts).slice(1, r.member.residency.lane.value, 1);
            ops::increment_token_counts(input, counts, cuda);
            f.control_offset += static_cast<std::uint32_t>(n);
            used += static_cast<std::uint32_t>(n);
        }
        if (f.control_offset == r.ids.size()) {
            ++f.control_row;
            f.control_offset = 0;
        }
    }
    f.counts_pending = f.control_row < rows.size();
    timer.record_result();
    compute_done.record(cuda.stream());
    f.phase = ComputePhase::AwaitResult;
}

} // namespace ninfer::models::inkling2::detail
