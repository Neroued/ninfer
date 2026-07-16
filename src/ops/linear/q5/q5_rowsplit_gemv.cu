#include "ops/linear/q5/q5_rowsplit_kernels.h"

#include "core/device.h"
#include "ops/linear/q5/q5_rowsplit_gemv.cuh"

#include <cuda_bf16.h>

#include <cstdint>

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

} // namespace ninfer::ops::detail
