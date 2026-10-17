#include "ops/sampling/acceptance/greedy_drafts/plan.h"
#include "ops/sampling/acceptance/greedy_drafts/kernel.cuh"
#include "core/device.h"

namespace ninfer::ops::greedy_drafts_detail {
using namespace sampling_detail;
using namespace common;

void launch(const Tensor& target_tokens, const Tensor& logits, const Tensor& drafts,
            const Tensor& current_extents, Tensor& lengths, Tensor& anchors,
            Tensor& licensed_tokens, Tensor& licensed_counts, Tensor& accepted,
            std::int32_t token_domain, const SamplingConfig* configs, const Plan& plan,
            MemorySpan workspace, cudaStream_t stream) {
    const std::int32_t physical_rows = logits.ne[0];
    const std::int32_t cols          = drafts.ne[0] + 1;
    const std::int32_t batch         = drafts.ne[1];
    const auto& layout               = plan.workspace;
    if (layout.bytes == 0) {
        speculative_accept_greedy_drafts_kernel<RowSchedule>
            <<<batch, RowSchedule::kSamplerBlock, 0, stream>>>(
                static_cast<const std::int32_t*>(target_tokens.data),
                static_cast<const __nv_bfloat16*>(logits.data),
                static_cast<const std::int32_t*>(drafts.data),
                static_cast<const std::int32_t*>(current_extents.data),
                static_cast<std::int32_t*>(lengths.data), static_cast<std::int32_t*>(anchors.data),
                static_cast<std::int32_t*>(licensed_tokens.data),
                static_cast<std::int32_t*>(licensed_counts.data),
                static_cast<std::int32_t*>(accepted.data), configs, token_domain, physical_rows,
                drafts.ne[0]);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const std::int32_t partial_blocks = plan.partial_blocks;
    const std::int32_t groups         = plan.groups;
    const Workspace scratch           = layout.bind(workspace);
    const dim3 partial_grid(static_cast<unsigned int>(partial_blocks),
                            static_cast<unsigned int>(cols), static_cast<unsigned int>(batch));
    speculative_sampling_partial_topk_kernel<ParallelSchedule>
        <<<partial_grid, ParallelSchedule::kSamplerBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(logits.data),
            static_cast<const std::int32_t*>(drafts.data),
            static_cast<const std::int32_t*>(current_extents.data), configs, token_domain,
            physical_rows, cols, drafts.ne[0], scratch, layout.bytes);
    CUDA_CHECK(cudaGetLastError());
    const dim3 batched_group_grid(static_cast<unsigned int>(groups),
                                  static_cast<unsigned int>(cols),
                                  static_cast<unsigned int>(batch));
    speculative_sampling_group_finalize_kernel<ParallelSchedule, false>
        <<<batched_group_grid, ParallelSchedule::kSamplerGroupBlock, 0, stream>>>(
            static_cast<const std::int32_t*>(target_tokens.data),
            static_cast<const std::int32_t*>(drafts.data), nullptr, nullptr,
            static_cast<const std::int32_t*>(current_extents.data),
            static_cast<std::int32_t*>(lengths.data), static_cast<std::int32_t*>(anchors.data),
            static_cast<std::int32_t*>(licensed_tokens.data),
            static_cast<std::int32_t*>(licensed_counts.data),
            static_cast<std::int32_t*>(accepted.data), configs, token_domain, cols, partial_blocks,
            groups, scratch, layout.bytes);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::greedy_drafts_detail
