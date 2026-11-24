#include "ninfer/ops/cast.h"
#include "core/device.h"
#include "ops/common/host/operands.h"
#include "ops/common/numeric/convert.cuh"
#include "ops/common/numeric/math.h"
#include <cuda_bf16.h>
#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {
namespace {

__global__ void cast_fp32_to_bf16_x4_kernel(const float4* source, common::Bf16x4Pack* destination,
                                            std::int64_t vectors) {
    const std::int64_t start  = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < vectors; i += stride) {
        const float4 value = source[i];
        destination[i]     = {
            {__floats2bfloat162_rn(value.x, value.y), __floats2bfloat162_rn(value.z, value.w)}};
    }
}

__global__ void cast_fp32_to_bf16_x2_kernel(const float2* source, __nv_bfloat162* destination,
                                            std::int64_t pairs) {
    const std::int64_t start  = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < pairs; i += stride) {
        const float2 value = source[i];
        destination[i]     = __floats2bfloat162_rn(value.x, value.y);
    }
}

__global__ void cast_fp32_to_bf16_scalar_kernel(const float* source, __nv_bfloat16* destination,
                                                std::int64_t count) {
    const std::int64_t start  = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < count; i += stride) {
        destination[i] = __float2bfloat16_rn(source[i]);
    }
}

constexpr int kBlock   = 256;
constexpr int kGridCap = 4096;

int cast_grid(std::int64_t work_items) {
    return static_cast<int>(std::max<std::int64_t>(
        1, std::min<std::int64_t>(common::div_up(work_items, static_cast<std::int64_t>(kBlock)),
                                  kGridCap)));
}

using CastLaunch = void (*)(const Tensor&, Tensor&, cudaStream_t);

template <int Width>
void launch_cast(const Tensor& source, Tensor& destination, cudaStream_t stream) {
    const auto count = source.numel() / Width;
    const int grid   = cast_grid(count);
    if constexpr (Width == 4)
        cast_fp32_to_bf16_x4_kernel<<<grid, kBlock, 0, stream>>>(
            static_cast<const float4*>(source.data),
            static_cast<common::Bf16x4Pack*>(destination.data), count);
    else if constexpr (Width == 2)
        cast_fp32_to_bf16_x2_kernel<<<grid, kBlock, 0, stream>>>(
            static_cast<const float2*>(source.data), static_cast<__nv_bfloat162*>(destination.data),
            count);
    else
        cast_fp32_to_bf16_scalar_kernel<<<grid, kBlock, 0, stream>>>(
            static_cast<const float*>(source.data), static_cast<__nv_bfloat16*>(destination.data),
            count);
}

CastLaunch select_cast(const Tensor& source, const Tensor& destination) {
    const auto count = source.numel();
    if (count % 4 == 0 && common::operand::aligned(source.data, 16) &&
        common::operand::aligned(destination.data, 8))
        return launch_cast<4>;
    if (count % 2 == 0 && common::operand::aligned(source.data, 8) &&
        common::operand::aligned(destination.data, 4))
        return launch_cast<2>;
    return launch_cast<1>;
}

} // namespace

void cast_fp32_to_bf16(const Tensor& source, Tensor& destination, CudaExecutionView execution) {
    if (source.device != execution.device() || destination.device != execution.device()) {
        throw std::invalid_argument(
            "cast_fp32_to_bf16: operands must belong to the execution device");
    }
    const auto stream = execution.stream();
    if (source.dtype != DType::FP32 || destination.dtype != DType::BF16) {
        throw std::invalid_argument("cast_fp32_to_bf16: source must be FP32 and destination BF16");
    }
    for (int dim = 0; dim < 4; ++dim) {
        if (source.ne[dim] != destination.ne[dim]) {
            throw std::invalid_argument("cast_fp32_to_bf16: shapes must match");
        }
    }
    if (!source.is_contiguous() || !destination.is_contiguous() || source.data == nullptr ||
        destination.data == nullptr) {
        throw std::invalid_argument("cast_fp32_to_bf16: tensors must be contiguous and non-null");
    }
    if (source.data == destination.data) {
        throw std::invalid_argument("cast_fp32_to_bf16: source and destination must not alias");
    }
    select_cast(source, destination)(source, destination, stream);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
