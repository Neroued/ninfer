#include "ops/sampling/sample/plan.h"

#include "ops/common/numeric/math.h"
#include "ops/sampling/sample/kernel.cuh"
#include "core/device.h"

namespace ninfer::ops::sample_detail {
using namespace sampling_detail;
using namespace common;

void launch(const Tensor& logits, Tensor& out, std::int32_t token_domain,
            const SamplingConfig* configs, const Tensor& logical_positions, std::int32_t purpose,
            const Tensor& live_rows, const Plan& plan, MemorySpan workspace, cudaStream_t stream) {
    const std::int32_t physical_rows = logits.ne[0];
    const std::int32_t batch         = logits.ne[1];
    const auto* positions            = static_cast<const std::int32_t*>(logical_positions.data);
    const auto* live                 = static_cast<const std::int32_t*>(live_rows.data);
    const auto& layout               = plan.workspace;
    if (layout.bytes == 0) {
        sample_row_kernel<RowSchedule>
            <<<static_cast<unsigned int>(batch), RowSchedule::kSamplerBlock, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(logits.data),
                static_cast<std::int32_t*>(out.data), configs, positions, purpose, token_domain,
                physical_rows, live);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const std::int32_t partial_blocks = plan.partial_blocks;
    const std::int32_t groups         = plan.groups;
    const Workspace scratch           = layout.bind(workspace);
    for (int begin = 0; begin < batch;) {
        const int rows = std::min(batch - begin, kColumnSlab);
        const dim3 partial_grid(static_cast<unsigned int>(partial_blocks),
                                static_cast<unsigned int>(rows));
        if (plan.partial_tile == 4096) {
            sampling_greedy_partial_kernel<4096>
                <<<partial_grid, 256, 0, stream>>>(static_cast<const __nv_bfloat16*>(logits.data) +
                                                       std::int64_t(begin) * physical_rows,
                                                   configs + begin, token_domain, physical_rows,
                                                   scratch, live ? live + begin : nullptr);
            CUDA_CHECK(cudaGetLastError());
            sampling_threshold_topk_kernel<256, 16>
                <<<partial_grid, 256, 0, stream>>>(static_cast<const __nv_bfloat16*>(logits.data) +
                                                       std::int64_t(begin) * physical_rows,
                                                   configs + begin, token_domain, physical_rows,
                                                   scratch, live ? live + begin : nullptr);
        } else {
            sampling_partial_topk_kernel<ParallelSchedule, NINFER_TARGET_SPARK>
                <<<partial_grid, ParallelSchedule::kSamplerBlock, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(logits.data) +
                        std::int64_t(begin) * physical_rows,
                    configs + begin, token_domain, physical_rows, scratch,
                    live ? live + begin : nullptr);
        }
        CUDA_CHECK(cudaGetLastError());
        const dim3 group_grid(static_cast<unsigned int>(groups), static_cast<unsigned int>(rows));
        sampling_group_finalize_sample_kernel<ParallelSchedule, NINFER_TARGET_SPARK>
            <<<group_grid, ParallelSchedule::kSamplerGroupBlock, 0, stream>>>(
                static_cast<std::int32_t*>(out.data) + begin, configs + begin, positions + begin,
                purpose, token_domain, partial_blocks, groups, scratch,
                live ? live + begin : nullptr);
        CUDA_CHECK(cudaGetLastError());
        begin += rows;
    }
}

} // namespace ninfer::ops::sample_detail
