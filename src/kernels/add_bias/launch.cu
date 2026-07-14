#include "kernels/add_bias/launch.h"

#include "kernels/common/math.h"
#include "kernels/add_bias/kernel.cuh"
#include "core/device.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace ninfer::kernels::detail {

void add_bias_launch(const Tensor& bias, Tensor& x, cudaStream_t stream) {
    constexpr int block     = 256;
    const std::int64_t n    = x.numel();
    const std::int64_t rows = n / x.ne[0];
    const bool paired =
        (x.ne[0] % 2) == 0 &&
        ((reinterpret_cast<std::uintptr_t>(bias.data) | reinterpret_cast<std::uintptr_t>(x.data)) &
         0x3u) == 0;
    if (paired && rows <= std::numeric_limits<std::int32_t>::max()) {
        const std::int32_t pairs = x.ne[0] / 2;
        const unsigned grid_x =
            static_cast<unsigned>(div_up(pairs, block * kAddBiasPairsPerThread));
        const unsigned grid_y = static_cast<unsigned>(
            std::min<std::int64_t>(rows, std::numeric_limits<unsigned short>::max()));
        add_bias_bf16x2_kernel<block><<<dim3(grid_x, grid_y, 1u), block, 0, stream>>>(
            static_cast<const __nv_bfloat162*>(bias.data), static_cast<__nv_bfloat162*>(x.data),
            pairs, static_cast<std::int32_t>(rows));
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const int grid = static_cast<int>(std::min<std::int64_t>(
        div_up(n, static_cast<std::int64_t>(block)), std::numeric_limits<int>::max()));
    add_bias_kernel<<<grid, block, 0, stream>>>(static_cast<const __nv_bfloat16*>(bias.data),
                                                static_cast<__nv_bfloat16*>(x.data), x.ne[0], n);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::kernels::detail
