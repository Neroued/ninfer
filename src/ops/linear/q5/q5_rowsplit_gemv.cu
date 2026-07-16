#include "ops/linear/q5/q5_rowsplit_kernels.h"

#include "core/device.h"
#include "ops/linear/q5/q5_rowsplit_gemv.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

void q5_rowsplit_gemv_r16_s2_x_launch(const Tensor& x, const Weight& w, Tensor& out,
                                      cudaStream_t stream) {
    constexpr int kRows = 6144;
    constexpr int kK    = 5120;
    q5_rowsplit_gemv_launch_kernel<kRows, kK, 16, 2, true>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.qhigh), static_cast<const std::uint8_t*>(w.scales),
        static_cast<__nv_bfloat16*>(out.data), stream);
    CUDA_CHECK(cudaGetLastError());
}

void q5_rowsplit_gemv_residual_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream) {
    const auto* xp       = static_cast<const __nv_bfloat16*>(x.data);
    const auto* codes    = static_cast<const std::uint8_t*>(w.qdata);
    const auto* high     = static_cast<const std::uint8_t*>(w.qhigh);
    const auto* scales   = static_cast<const std::uint8_t*>(w.scales);
    const auto* residual = static_cast<const __nv_bfloat16*>(residual_out.data);
    auto* out            = static_cast<__nv_bfloat16*>(residual_out.data);

    if (w.n == 5120 && w.k == 6144 && w.padded_shape[1] == 6144) {
        q5_rowsplit_gemv_residual_launch_kernel<5120, 6144, 16, 2, true>(xp, codes, high, scales,
                                                                         residual, out, stream);
    } else if (w.n == 5120 && w.k == 17408 && w.padded_shape[1] == 17408) {
        q5_rowsplit_gemv_residual_launch_kernel<5120, 17408, 16, 2, false>(xp, codes, high, scales,
                                                                           residual, out, stream);
    } else {
        throw std::invalid_argument("q5 residual GEMV: unsupported exact shape");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
