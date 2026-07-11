#include "kernels/launcher/scatter.h"

#include "kernels/common/math.h"
#include "kernels/kernel/scatter.cuh"
#include "qus/core/device.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace qus::kernels::detail {

void scatter_launch(const Tensor& src, const Tensor& indices, Tensor& dst, cudaStream_t stream) {
    constexpr int block  = 256;
    const std::int64_t n = src.numel();
    const int grid       = static_cast<int>(std::min<std::int64_t>(
        div_up(n, static_cast<std::int64_t>(block)), std::numeric_limits<int>::max()));
    scatter_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(src.data), static_cast<const std::int32_t*>(indices.data),
        static_cast<__nv_bfloat16*>(dst.data), src.ne[0], dst.ne[1], n);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace qus::kernels::detail
