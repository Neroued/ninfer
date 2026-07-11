#include "kernels/launcher/add_bias.h"

#include "kernels/common/math.h"
#include "kernels/kernel/add_bias.cuh"
#include "qus/core/device.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace qus::kernels::detail {

void add_bias_launch(const Tensor& bias, Tensor& x, cudaStream_t stream) {
    constexpr int block  = 256;
    const std::int64_t n = x.numel();
    const int grid       = static_cast<int>(std::min<std::int64_t>(
        div_up(n, static_cast<std::int64_t>(block)), std::numeric_limits<int>::max()));
    add_bias_kernel<<<grid, block, 0, stream>>>(static_cast<const __nv_bfloat16*>(bias.data),
                                                static_cast<__nv_bfloat16*>(x.data), x.ne[0], n);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace qus::kernels::detail
