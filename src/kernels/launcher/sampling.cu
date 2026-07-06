// qus::kernels::detail - sample_column launch. One block per logits column.
#include "kernels/launcher/sampling.h"

#include "kernels/kernel/sampling.cuh"

namespace qus::kernels::detail {

void sample_column_launch(const Tensor& logits, Tensor& out, const SamplingConfig* config,
                          const std::int32_t* pos_base, std::int32_t purpose,
                          cudaStream_t stream) {
    const std::int32_t vocab = logits.ne[0];
    const std::int32_t cols  = logits.ne[1];
    const std::int32_t partial_blocks =
        (vocab + kSamplerPartialTileItems - 1) / kSamplerPartialTileItems;
    if (vocab <= kSamplerTileItems || cols > kSamplerScratchColumns ||
        partial_blocks > kSamplerScratchPartialBlocks) {
        sample_column_kernel<<<static_cast<unsigned int>(cols), kSamplerBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(logits.data), static_cast<std::int32_t*>(out.data),
            config, pos_base, purpose, vocab);
        return;
    }
    const dim3 partial_grid(static_cast<unsigned int>(partial_blocks),
                            static_cast<unsigned int>(cols));
    if (cols == 1) {
        const std::int32_t fused_partial_blocks =
            (vocab + kSamplerFusedPartialTileItems - 1) / kSamplerFusedPartialTileItems;
        if (fused_partial_blocks <= kSamplerScratchPartialBlocks) {
            const std::int32_t fused_groups =
                (fused_partial_blocks + kSamplerPartialsPerGroup - 1) / kSamplerPartialsPerGroup;
            const dim3 fused_grid(static_cast<unsigned int>(fused_partial_blocks),
                                  static_cast<unsigned int>(cols));
            sampling_fused_sample_kernel<<<fused_grid, kSamplerBlock, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(logits.data),
                static_cast<std::int32_t*>(out.data), config, pos_base, purpose, vocab,
                fused_partial_blocks, fused_groups);
            return;
        }
    }
    sampling_partial_topk_kernel<<<partial_grid, kSamplerBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits.data), config, vocab);
    const std::int32_t groups =
        (partial_blocks + kSamplerPartialsPerGroup - 1) / kSamplerPartialsPerGroup;
    const dim3 group_grid(static_cast<unsigned int>(groups), static_cast<unsigned int>(cols));
    sampling_group_finalize_sample_kernel<<<group_grid, kSamplerBlock, 0, stream>>>(
        static_cast<std::int32_t*>(out.data), config, pos_base, purpose, vocab, partial_blocks,
        groups);
}

} // namespace qus::kernels::detail
