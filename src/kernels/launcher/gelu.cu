#include "kernels/launcher/gelu.h"

#include "kernels/common/math.h"
#include "kernels/kernel/gelu.cuh"
#include "ninfer/core/device.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace ninfer::kernels::detail {

void gelu_launch(Tensor& x, GeluMode mode, cudaStream_t stream) {
    constexpr int block  = 256;
    const std::int64_t n = x.numel();
    const bool paired    = (reinterpret_cast<std::uintptr_t>(x.data) & 0x3u) == 0;
    if (paired && n >= 2) {
        const std::int64_t pairs = n / 2;
        const int grid           = static_cast<int>(std::min<std::int64_t>(
            div_up(pairs, static_cast<std::int64_t>(block * kGeluPairsPerThread)),
            std::numeric_limits<int>::max()));
        auto* x2                 = static_cast<__nv_bfloat162*>(x.data);
        auto* tail               = static_cast<__nv_bfloat16*>(x.data) + pairs * 2;
        if (mode == GeluMode::Tanh) {
            gelu_bf16x2_kernel<true, block>
                <<<grid, block, 0, stream>>>(x2, pairs, tail, (n & 1) != 0);
        } else {
            gelu_bf16x2_kernel<false, block>
                <<<grid, block, 0, stream>>>(x2, pairs, tail, (n & 1) != 0);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const int grid = static_cast<int>(std::min<std::int64_t>(
        div_up(n, static_cast<std::int64_t>(block)), std::numeric_limits<int>::max()));
    if (mode == GeluMode::Tanh) {
        gelu_kernel<true><<<grid, block, 0, stream>>>(static_cast<__nv_bfloat16*>(x.data), n);
    } else {
        gelu_kernel<false><<<grid, block, 0, stream>>>(static_cast<__nv_bfloat16*>(x.data), n);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::kernels::detail
