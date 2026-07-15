// ninfer::ops::detail - sample launch. One block per logits column.
#include "ops/launcher/sampling.h"

#include "ops/common/math.h"
#include "ops/kernel/sampling.cuh"
#include "core/device.h"

namespace ninfer::ops::detail {

void sample_column_launch(const Tensor& logits, Tensor& out, std::int32_t token_domain,
                          const SamplingConfig* config, const std::int32_t* pos_base,
                          std::int32_t purpose, cudaStream_t stream) {
    const std::int32_t physical_rows  = logits.ne[0];
    const std::int32_t cols           = logits.ne[1];
    const std::int32_t partial_blocks = div_up(token_domain, kSamplerPartialTileItems);
    const std::int32_t groups         = sampler_group_count(partial_blocks);
    // The single-block fallback and the multi-block scratch path share
    // sampler_multiblock_ok so exactly one owns any given shape; it also bounds
    // group_count*cap to the merge tile (F1) and the scratch partial budget.
    if (!sampler_multiblock_ok(token_domain, cols, partial_blocks, groups)) {
        sample_column_kernel<<<static_cast<unsigned int>(cols), kSamplerBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(logits.data), static_cast<std::int32_t*>(out.data),
            config, pos_base, purpose, token_domain, physical_rows);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const dim3 partial_grid(static_cast<unsigned int>(partial_blocks),
                            static_cast<unsigned int>(cols));
    sampling_partial_topk_kernel<<<partial_grid, kSamplerBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits.data), config, token_domain, physical_rows);
    CUDA_CHECK(cudaGetLastError());
    const dim3 group_grid(static_cast<unsigned int>(groups), static_cast<unsigned int>(cols));
    sampling_group_finalize_sample_kernel<<<group_grid, kSamplerBlock, 0, stream>>>(
        static_cast<std::int32_t*>(out.data), config, pos_base, purpose, token_domain,
        partial_blocks, groups);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
