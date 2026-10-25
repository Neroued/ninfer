#include "models/glm5next/program/internal.h"
#include "core/transfer.h"
#include <algorithm>

namespace ninfer::models::glm5next::detail {
void RankProgramImpl::commit(runtime::ExecutionTicket ticket, runtime::CommandSequence command,
                             Commit&& commit) {
    prepare_commit(ticket, commit);
    auto& f = *compute;
    timer.resolve();
    f.release_vision    = std::any_of(commit.rows.begin(), commit.rows.end(),
                                      [](const auto& row) { return row.terminal; });
    const auto* forward = std::get_if<ForwardWork>(&f.request.work.operation);
    if (forward && forward->sample != SampleKind::None) {
        const auto& io = resources.plan.frame;
        const int b    = commit.rows.size();
        auto* values   = static_cast<std::int32_t*>(host(io.main.accepted_columns).data);
        for (int i = 0; i < b; ++i) values[i] = commit.rows[f.physical.members[i]].accepted;
        upload_region(io.main.accepted_columns.region, b * 4ULL);
        auto counts = resources.tensor(*io.token_counts);
        ops::increment_token_counts_batch(resources.packed(*io.sampled_ids, {1, b}),
                                          resources.tensor(io.main.accepted_columns).slice(0, 0, b),
                                          resources.tensor(io.main.count_slots).slice(0, 0, b),
                                          counts, cuda);
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
    const auto& io     = resources.plan.frame;
    const auto cap     = resources.tensor(io.main.ids).numel();
    auto* ids          = static_cast<TokenId*>(host(io.main.ids).data);
    std::uint32_t used = 0;
    while (f.control_row < rows.size() && used < cap) {
        const auto& row = rows[f.control_row];
        const auto n =
            std::min<std::size_t>(cap - used, row.count_tokens.size() - f.control_offset);
        if (n) {
            std::copy_n(row.count_tokens.begin() + f.control_offset, n, ids + used);
            auto input = resources.tensor(io.main.ids).slice(0, used, n);
            copy_bytes_async({input.data, input.bytes(), cuda.device()},
                             {ids + used, input.bytes(), Device::cpu(), true}, cuda);
            auto counts =
                resources.tensor(*io.token_counts).slice(1, row.member.residency.lane.value, 1);
            ops::increment_token_counts(input, counts, cuda);
            f.control_offset += n;
            used += n;
        }
        if (f.control_offset == row.count_tokens.size()) {
            ++f.control_row;
            f.control_offset = 0;
        }
    }
    f.counts_pending = f.control_row < rows.size();
    timer.record_result();
    compute_done.record(cuda.stream());
    f.phase = ComputePhase::AwaitResult;
}
} // namespace ninfer::models::glm5next::detail
