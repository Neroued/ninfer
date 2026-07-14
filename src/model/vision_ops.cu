#include "model/vision_ops.h"

#include "kernels/common/math.h"
#include "ninfer/core/device.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::model::detail {
namespace {

__global__ void vision_f32_to_bf16_kernel(const float* src, __nv_bfloat16* dst, std::int64_t n) {
    const std::int64_t start  = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < n; i += stride) { dst[i] = __float2bfloat16_rn(src[i]); }
}

} // namespace

void vision_f32_to_bf16(const Tensor& src, Tensor& dst, cudaStream_t stream) {
    if (src.dtype != DType::FP32 || dst.dtype != DType::BF16) {
        throw std::invalid_argument("vision_f32_to_bf16: src must be FP32 and dst BF16");
    }
    for (int dim = 0; dim < 4; ++dim) {
        if (src.ne[dim] != dst.ne[dim]) {
            throw std::invalid_argument("vision_f32_to_bf16: shapes must match");
        }
    }
    if (!src.is_contiguous() || !dst.is_contiguous() || src.data == nullptr ||
        dst.data == nullptr) {
        throw std::invalid_argument("vision_f32_to_bf16: tensors must be contiguous and non-null");
    }
    constexpr int block  = 256;
    const std::int64_t n = src.numel();
    const int grid       = static_cast<int>(std::min<std::int64_t>(
        kernels::div_up(n, static_cast<std::int64_t>(block)), std::numeric_limits<int>::max()));
    vision_f32_to_bf16_kernel<<<grid, block, 0, stream>>>(static_cast<const float*>(src.data),
                                                          static_cast<__nv_bfloat16*>(dst.data), n);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::model::detail
