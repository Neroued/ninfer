#include "kernels/vision_pos_embed/launch.h"

#include "kernels/common/math.h"
#include "kernels/vision_pos_embed/kernel.cuh"
#include "core/device.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace ninfer::kernels::detail {

void vision_pos_embed_add_launch(const Tensor& table, const Tensor& indices, const Tensor& weights,
                                 Tensor& x, cudaStream_t stream) {
    constexpr int block = 128;
    const bool paired =
        x.ne[0] == 1152 &&
        ((reinterpret_cast<std::uintptr_t>(table.data) | reinterpret_cast<std::uintptr_t>(x.data)) &
         0x3u) == 0;
    if (paired) {
        vision_pos_embed_add_d1152_kernel<block>
            <<<static_cast<unsigned>(x.ne[1]), block, 0, stream>>>(
                static_cast<const __nv_bfloat162*>(table.data),
                static_cast<const std::int32_t*>(indices.data),
                static_cast<const float*>(weights.data), static_cast<__nv_bfloat162*>(x.data),
                x.ne[1], table.ne[1]);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const std::int64_t n = x.numel();
    const int grid       = static_cast<int>(std::min<std::int64_t>(
        div_up(n, static_cast<std::int64_t>(block)), std::numeric_limits<int>::max()));
    vision_pos_embed_add_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(table.data),
        static_cast<const std::int32_t*>(indices.data), static_cast<const float*>(weights.data),
        static_cast<__nv_bfloat16*>(x.data), x.ne[0], x.ne[1], table.ne[1], n);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::kernels::detail
