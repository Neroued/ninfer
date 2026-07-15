// Implements: include/ninfer/ops/sampling.h
// Match: validated contiguous BF16/I32 tensors and a shared-layout workspace.
// Algorithm assumptions: launcher and kernels use sampler_multiblock_ok() from
// the same layout authority, so exactly one finite route owns each shape.
#include "ops/launcher/sampling.h"

#include "ops/common/math.h"
#include "ops/kernel/sampling.cuh"
#include "core/device.h"

namespace ninfer::ops::detail {

std::size_t sampling_workspace_bytes(std::int32_t token_domain, std::int32_t columns) {
    return make_sampling_workspace_layout(token_domain, columns).bytes;
}

void sample_column_launch(const Tensor& logits, Tensor& out, std::int32_t token_domain,
                          const SamplingConfig* config, const std::int32_t* pos_base,
                          std::int32_t purpose, DeviceSpan workspace, cudaStream_t stream) {
    const std::int32_t physical_rows     = logits.ne[0];
    const std::int32_t cols              = logits.ne[1];
    const SamplingWorkspaceLayout layout = make_sampling_workspace_layout(token_domain, cols);
    if (!layout.multiblock) {
        sample_column_kernel<<<static_cast<unsigned int>(cols), kSamplerBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(logits.data), static_cast<std::int32_t*>(out.data),
            config, pos_base, purpose, token_domain, physical_rows);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const std::int32_t partial_blocks = div_up(token_domain, kSamplerPartialTileItems);
    const std::int32_t groups         = sampler_group_count(partial_blocks);
    const SamplingWorkspace scratch   = layout.bind(workspace);
    const dim3 partial_grid(static_cast<unsigned int>(partial_blocks),
                            static_cast<unsigned int>(cols));
    sampling_partial_topk_kernel<<<partial_grid, kSamplerBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits.data), config, token_domain, physical_rows,
        scratch);
    CUDA_CHECK(cudaGetLastError());
    const dim3 group_grid(static_cast<unsigned int>(groups), static_cast<unsigned int>(cols));
    sampling_group_finalize_sample_kernel<<<group_grid, kSamplerGroupBlock, 0, stream>>>(
        static_cast<std::int32_t*>(out.data), config, pos_base, purpose, token_domain,
        partial_blocks, groups, scratch);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
