// qus::kernels - argmax launcher: grid/block/stream configuration + kernel launch.
#include "kernels/launcher/argmax.h"

#include "kernels/kernel/argmax.cuh"
#include "qus/core/device.h"  // CUDA_CHECK

#include <cstdint>

namespace qus::kernels::detail {

void argmax_launch(const Tensor& logits, Tensor& out, cudaStream_t stream) {
    constexpr int kBlock = 256;
    const std::int32_t vocab = logits.ne[0];
    const std::int32_t t_count = logits.ne[1];
    if (t_count == 0) { return; }

    argmax_kernel<<<static_cast<unsigned int>(t_count), kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits.data), static_cast<std::int32_t*>(out.data),
        vocab);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace qus::kernels::detail
