#include "ops/launcher/cast.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/kernel/cast.cuh"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace ninfer::ops::detail {

void cast_fp32_to_bf16_launch(const Tensor& source, Tensor& destination, cudaStream_t stream) {
    constexpr int block      = 256;
    const std::int64_t count = source.numel();
    const int grid           = static_cast<int>(std::min<std::int64_t>(
        div_up(count, static_cast<std::int64_t>(block)), std::numeric_limits<int>::max()));
    cast_fp32_to_bf16_kernel<<<grid, block, 0, stream>>>(
        static_cast<const float*>(source.data), static_cast<__nv_bfloat16*>(destination.data),
        count);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
