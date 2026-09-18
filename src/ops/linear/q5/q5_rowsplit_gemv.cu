#include "core/weight.h"
#include "core/device.h"
#include "ops/linear/q5/q5_launch.h"
#include "ops/linear/q5/q5_rowsplit_gemv.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

void launch_q5_gemv_r16_s2_x(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    constexpr int kK = 5120;
    if (w.n == 6144) {
        q5_rowsplit_gemv_launch_kernel<6144, kK, 16, 2, true>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.qhigh), static_cast<const std::uint8_t*>(w.scales),
            static_cast<__nv_bfloat16*>(out.data), stream);
    } else if (w.n == 7168) {
        q5_rowsplit_gemv_launch_kernel<7168, kK, 16, 2, true>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.qhigh), static_cast<const std::uint8_t*>(w.scales),
            static_cast<__nv_bfloat16*>(out.data), stream);
    } else {
        throw std::invalid_argument("q5 GEMV R16/S2/X: unsupported exact N");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
