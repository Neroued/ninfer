#include "ops/sampling/acceptance/sparse_drafts/plan.h"
#include "ops/sampling/acceptance/kernel.cuh"
#include "core/device.h"

namespace ninfer::ops::sparse_drafts_detail {
using namespace sampling_detail;
using namespace common;

namespace {
__global__ __launch_bounds__(256) void speculative_accept_sparse_warp_greedy_kernel(
    const int* target_tokens, TokenRows drafts, const int* current_extents, int* licensed_tokens,
    int* licensed_counts, int k, int batch, const int* live) {
    const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= batch) return;
    if (live && !live[row]) {
        const int lane = threadIdx.x & 31;
        if (lane <= k) licensed_tokens[std::int64_t(row) * (k + 1) + lane] = 0;
        if (lane == 0) licensed_counts[row] = 0;
        return;
    }
    const int extent = min(k, max(0, current_extents[row]));
    speculative_sparse_warp_greedy(target_tokens, drafts, licensed_tokens, licensed_counts, row,
                                   extent, k);
}
} // namespace

void launch(const Tensor& target_tokens, const Tensor& logits, const Tensor& drafts,
            const Tensor& candidate_ids, const Tensor& proposal_q, const Tensor& current_extents,
            const Tensor& base_positions, Tensor& licensed_tokens, Tensor& licensed_counts,
            std::int32_t token_domain, const SamplingConfig* configs, const Tensor& live_rows,
            const Plan& plan, MemorySpan workspace, CudaExecutionView execution) {
    const auto stream        = execution.stream();
    const std::int32_t batch = drafts.ne[1];
    const std::int32_t k     = drafts.ne[0];
    const std::int32_t cols  = k + 1;
    const auto* live         = static_cast<const std::int32_t*>(live_rows.data);
    if (plan.raw_greedy) {
        const int warps = std::min(batch, 8);
        speculative_accept_sparse_warp_greedy_kernel<<<div_up(batch, warps), 32 * warps, 0,
                                                       stream>>>(
            static_cast<const int*>(target_tokens.data), token_rows(drafts),
            static_cast<const int*>(current_extents.data), static_cast<int*>(licensed_tokens.data),
            static_cast<int*>(licensed_counts.data), k, batch, live);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const auto& layout                = plan.workspace;
    const std::int32_t partial_blocks = plan.partial_blocks;
    const std::int32_t groups         = plan.groups;
    const Workspace scratch           = layout.bind(workspace);
    for (int begin = 0; begin < batch;) {
        const int rows = std::min(batch - begin, 65535);
        const SpeculativeSelectionInput input{
            .logits          = static_cast<const __nv_bfloat16*>(logits.data),
            .drafts          = token_rows(drafts),
            .current_extents = static_cast<const std::int32_t*>(current_extents.data),
            .configs         = configs,
            .live_rows       = live,
            .token_domain    = token_domain,
            .physical_rows   = logits.ne[0],
            .columns         = cols,
            .draft_count     = k,
            .workspace       = scratch,
            .row_stride      = layout.bytes,
            .first_row       = begin};
        launch_speculative_selection(input, rows, partial_blocks, plan.partial_tile, execution);

        const dim3 group_grid(static_cast<unsigned int>(groups), static_cast<unsigned int>(cols),
                              static_cast<unsigned int>(rows));

        speculative_sampling_group_finalize_kernel<Schedule, true>
            <<<group_grid, Schedule::kSamplerGroupBlock, 0, stream>>>(
                static_cast<const std::int32_t*>(target_tokens.data), token_rows(drafts),
                static_cast<const std::int32_t*>(candidate_ids.data),
                static_cast<const float*>(proposal_q.data),
                static_cast<const std::int32_t*>(current_extents.data),
                static_cast<const std::int32_t*>(base_positions.data),
                static_cast<std::int32_t*>(licensed_tokens.data),
                static_cast<std::int32_t*>(licensed_counts.data), configs, token_domain, cols,
                partial_blocks, groups, scratch, layout.bytes, live, begin);

        CUDA_CHECK(cudaGetLastError());
        begin += rows;
    }
}

} // namespace ninfer::ops::sparse_drafts_detail
