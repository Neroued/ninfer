#include "ops/sampling/acceptance/sparse_drafts/plan.h"
#include "ops/sampling/acceptance/kernel.cuh"
#include "core/device.h"

namespace ninfer::ops::sparse_drafts_detail {
using namespace sampling_detail;
using namespace common;

namespace {
__global__ __launch_bounds__(256) void speculative_accept_sparse_warp_greedy_kernel(
    const int* target_tokens, TokenRows drafts, const int* current_extents, int* licensed_tokens,
    int* licensed_counts, int k) {
    const int row    = threadIdx.x / 32;
    const int extent = min(k, max(0, current_extents[row]));
    speculative_sparse_warp_greedy(target_tokens, drafts, licensed_tokens, licensed_counts, row,
                                   extent, k);
}
} // namespace

void launch(const Tensor& target_tokens, const Tensor& logits, const Tensor& drafts,
            const Tensor& candidate_ids, const Tensor& proposal_q, const Tensor& current_extents,
            const Tensor& base_positions, Tensor& licensed_tokens, Tensor& licensed_counts,
            std::int32_t token_domain, const SamplingConfig* configs, const Plan& plan,
            MemorySpan workspace, cudaStream_t stream) {
    const std::int32_t batch = drafts.ne[1];
    const std::int32_t k     = drafts.ne[0];
    const std::int32_t cols  = k + 1;
    if (plan.raw_greedy) {
        speculative_accept_sparse_warp_greedy_kernel<<<1, 32 * batch, 0, stream>>>(
            static_cast<const int*>(target_tokens.data), token_rows(drafts),
            static_cast<const int*>(current_extents.data), static_cast<int*>(licensed_tokens.data),
            static_cast<int*>(licensed_counts.data), k);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const auto& layout                = plan.workspace;
    const std::int32_t partial_blocks = plan.partial_blocks;
    const std::int32_t groups         = plan.groups;
    const Workspace scratch           = layout.bind(workspace);
    const dim3 partial_grid(static_cast<unsigned int>(partial_blocks),
                            static_cast<unsigned int>(cols), static_cast<unsigned int>(batch));
    speculative_sampling_partial_topk_kernel<Schedule>
        <<<partial_grid, Schedule::kSamplerBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(logits.data), token_rows(drafts),
            static_cast<const std::int32_t*>(current_extents.data), configs, token_domain,
            logits.ne[0], cols, k, scratch, layout.bytes);
    CUDA_CHECK(cudaGetLastError());

    const dim3 group_grid(static_cast<unsigned int>(groups), static_cast<unsigned int>(cols),
                          static_cast<unsigned int>(batch));

    speculative_sampling_group_finalize_kernel<Schedule, true>
        <<<group_grid, Schedule::kSamplerGroupBlock, 0, stream>>>(
            static_cast<const std::int32_t*>(target_tokens.data), token_rows(drafts),
            static_cast<const std::int32_t*>(candidate_ids.data),
            static_cast<const float*>(proposal_q.data),
            static_cast<const std::int32_t*>(current_extents.data),
            static_cast<const std::int32_t*>(base_positions.data),
            static_cast<std::int32_t*>(licensed_tokens.data),
            static_cast<std::int32_t*>(licensed_counts.data), configs, token_domain, cols,
            partial_blocks, groups, scratch, layout.bytes);

    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::sparse_drafts_detail
