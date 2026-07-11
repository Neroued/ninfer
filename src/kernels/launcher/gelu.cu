#include "kernels/launcher/gelu.h"

#include "kernels/common/math.h"
#include "kernels/kernel/gelu.cuh"
#include "qus/core/device.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace qus::kernels::detail {

void gelu_launch(Tensor& x, GeluMode mode, cudaStream_t stream) {
    constexpr int block  = 256;
    const std::int64_t n = x.numel();
    const int grid       = static_cast<int>(std::min<std::int64_t>(
        div_up(n, static_cast<std::int64_t>(block)), std::numeric_limits<int>::max()));
    if (mode == GeluMode::Tanh) {
        gelu_kernel<true><<<grid, block, 0, stream>>>(static_cast<__nv_bfloat16*>(x.data), n);
    } else {
        gelu_kernel<false><<<grid, block, 0, stream>>>(static_cast<__nv_bfloat16*>(x.data), n);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace qus::kernels::detail
