// qus::kernels::detail - sample_column launch. One block per logits column.
#include "kernels/launcher/sampling.h"

#include "kernels/kernel/sampling.cuh"

namespace qus::kernels::detail {

void sample_column_launch(const Tensor& logits, Tensor& out, const SamplingConfig* config,
                          const std::int32_t* pos_base, std::int32_t purpose,
                          cudaStream_t stream) {
    const std::int32_t vocab = logits.ne[0];
    const std::int32_t cols  = logits.ne[1];
    sample_column_kernel<<<static_cast<unsigned int>(cols), kSamplerBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits.data), static_cast<std::int32_t*>(out.data),
        config, pos_base, purpose, vocab);
}

} // namespace qus::kernels::detail
