#include "kernels/launcher/scatter.h"

#include "kernels/kernel/scatter.cuh"
#include "ninfer/core/device.h"

#include <cstdint>

namespace ninfer::kernels::detail {

void scatter_launch(const Tensor& src, const Tensor& indices, Tensor& dst, cudaStream_t stream) {
    constexpr int block = 256;
    const int d         = src.ne[0];
    const int vision    = src.ne[1];
    if ((d & 1) == 0) {
        scatter_bf16x2_kernel<<<vision, block, 0, stream>>>(
            static_cast<const __nv_bfloat162*>(src.data),
            static_cast<const std::int32_t*>(indices.data),
            static_cast<__nv_bfloat162*>(dst.data), d / 2, dst.ne[1]);
    } else {
        scatter_scalar_kernel<<<vision, block, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(src.data),
            static_cast<const std::int32_t*>(indices.data),
            static_cast<__nv_bfloat16*>(dst.data), d, dst.ne[1]);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::kernels::detail
