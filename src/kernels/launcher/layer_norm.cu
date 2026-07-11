#include "kernels/launcher/layer_norm.h"

#include "kernels/kernel/layer_norm.cuh"
#include "qus/core/device.h"

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace qus::kernels::detail {

void layer_norm_launch(const Tensor& x, const Tensor& weight, const Tensor& bias, float eps,
                       Tensor& out, cudaStream_t stream) {
    constexpr int block     = 256;
    const std::int64_t rows = x.numel() / x.ne[0];
    if (rows > std::numeric_limits<unsigned>::max()) {
        throw std::overflow_error("layer_norm: launch grid exceeds CUDA limit");
    }
    layer_norm_kernel<block><<<static_cast<unsigned>(rows), block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(weight.data),
        static_cast<const __nv_bfloat16*>(bias.data), static_cast<__nv_bfloat16*>(out.data),
        x.ne[0], rows, eps);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace qus::kernels::detail
