#include "kernels/launcher/layer_norm.h"

#include "kernels/kernel/layer_norm.cuh"
#include "ninfer/core/device.h"

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::kernels::detail {

void layer_norm_launch(const Tensor& x, const Tensor& weight, const Tensor& bias, float eps,
                       Tensor& out, cudaStream_t stream) {
    constexpr int block        = 256;
    constexpr int vision_block = 128;
    const std::int64_t rows    = x.numel() / x.ne[0];
    if (rows > std::numeric_limits<unsigned>::max()) {
        throw std::overflow_error("layer_norm: launch grid exceeds CUDA limit");
    }
    const bool paired = x.ne[0] == 1152 && ((reinterpret_cast<std::uintptr_t>(x.data) |
                                             reinterpret_cast<std::uintptr_t>(weight.data) |
                                             reinterpret_cast<std::uintptr_t>(bias.data) |
                                             reinterpret_cast<std::uintptr_t>(out.data)) &
                                            0x3u) == 0;
    if (paired) {
        constexpr int warps = vision_block / 32;
        const unsigned grid = static_cast<unsigned>((rows + warps - 1) / warps);
        layer_norm_d1152_warp_kernel<vision_block>
            <<<grid, vision_block, 0, stream>>>(static_cast<const __nv_bfloat162*>(x.data),
                                                static_cast<const __nv_bfloat162*>(weight.data),
                                                static_cast<const __nv_bfloat162*>(bias.data),
                                                static_cast<__nv_bfloat162*>(out.data), rows, eps);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    layer_norm_kernel<block><<<static_cast<unsigned>(rows), block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(weight.data),
        static_cast<const __nv_bfloat16*>(bias.data), static_cast<__nv_bfloat16*>(out.data),
        x.ne[0], rows, eps);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::kernels::detail
