#include "kernels/launcher/vision_pos_embed.h"

#include "kernels/common/math.h"
#include "kernels/kernel/vision_pos_embed.cuh"
#include "qus/core/device.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace qus::kernels::detail {

void vision_pos_embed_add_launch(const Tensor& table, const Tensor& indices, const Tensor& weights,
                                 Tensor& x, cudaStream_t stream) {
    constexpr int block  = 256;
    const std::int64_t n = x.numel();
    const int grid       = static_cast<int>(std::min<std::int64_t>(
        div_up(n, static_cast<std::int64_t>(block)), std::numeric_limits<int>::max()));
    vision_pos_embed_add_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(table.data),
        static_cast<const std::int32_t*>(indices.data), static_cast<const float*>(weights.data),
        static_cast<__nv_bfloat16*>(x.data), x.ne[0], x.ne[1], table.ne[1], n);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace qus::kernels::detail
