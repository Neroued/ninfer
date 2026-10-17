#include "ops/sampling/sample/plan.h"

#include "ops/common/numeric/math.h"
#include "ops/sampling/sample/kernel.cuh"
#include "core/device.h"

namespace ninfer::ops::sample_detail {
using namespace sampling_detail;
using namespace common;

void launch(const Tensor& logits, Tensor& out, std::int32_t token_domain,
            const SamplingConfig* configs, const Tensor& logical_positions, std::int32_t purpose,
            const Plan& plan, MemorySpan workspace, cudaStream_t stream) {
    const std::int32_t physical_rows = logits.ne[0];
    const std::int32_t batch         = logits.ne[1];
    const auto* positions            = static_cast<const std::int32_t*>(logical_positions.data);
    const auto& layout               = plan.workspace;
    if (layout.bytes == 0) {
        sample_row_kernel<RowSchedule>
            <<<static_cast<unsigned int>(batch), RowSchedule::kSamplerBlock, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(logits.data),
                static_cast<std::int32_t*>(out.data), configs, positions, purpose, token_domain,
                physical_rows);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const std::int32_t partial_blocks = plan.partial_blocks;
    const std::int32_t groups         = plan.groups;
    const Workspace scratch           = layout.bind(workspace);
    const dim3 partial_grid(static_cast<unsigned int>(partial_blocks),
                            static_cast<unsigned int>(batch));
    sampling_partial_topk_kernel<ParallelSchedule, NINFER_TARGET_SPARK>
        <<<partial_grid, ParallelSchedule::kSamplerBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(logits.data), configs, token_domain, physical_rows,
            scratch);
    CUDA_CHECK(cudaGetLastError());
    const dim3 group_grid(static_cast<unsigned int>(groups), static_cast<unsigned int>(batch));
    sampling_group_finalize_sample_kernel<ParallelSchedule, NINFER_TARGET_SPARK>
        <<<group_grid, ParallelSchedule::kSamplerGroupBlock, 0, stream>>>(
            static_cast<std::int32_t*>(out.data), configs, positions, purpose, token_domain,
            partial_blocks, groups, scratch);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::sample_detail
