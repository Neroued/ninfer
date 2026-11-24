#include "ninfer/ops/embedding.h"
#include "core/device.h"
#include "ops/common/host/operands.h"
#include "ops/common/numeric/math.h"
#include <cuda_bf16.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::ops {
namespace {
__global__ void embedding_interpolate_add_d1152_warp_kernel(const __nv_bfloat162* table,
                                                            const std::int32_t* indices,
                                                            const float* weights, __nv_bfloat162* x,
                                                            std::int32_t patches,
                                                            std::int32_t tiles_per_patch) {
    constexpr int pairs = 1152 / 2;
    const int lane      = static_cast<int>(threadIdx.x);
    const int patch     = static_cast<int>(blockIdx.x) / tiles_per_patch;
    const int tile      = static_cast<int>(blockIdx.x) - patch * tiles_per_patch;
    if (patch >= patches) { return; }

    const std::int64_t control = static_cast<std::int64_t>(patch) * 4 + lane;
    const int lane_index       = lane < 4 ? indices[control] : 0;
    const float lane_weight    = lane < 4 ? weights[control] : 0.0f;
    int corner_indices[4];
    float corner_weights[4];
#pragma unroll
    for (int corner = 0; corner < 4; ++corner) {
        corner_indices[corner] = __shfl_sync(0xffffffffu, lane_index, corner);
        corner_weights[corner] = __shfl_sync(0xffffffffu, lane_weight, corner);
    }

    const std::int64_t x_base = static_cast<std::int64_t>(patch) * pairs;
    for (int pair = tile * 32 + lane; pair < pairs; pair += tiles_per_patch * 32) {
        float lo = 0.0f;
        float hi = 0.0f;
#pragma unroll
        for (int corner = 0; corner < 4; ++corner) {
            const float2 value = __bfloat1622float2(
                table[static_cast<std::int64_t>(corner_indices[corner]) * pairs + pair]);
            lo += value.x * corner_weights[corner];
            hi += value.y * corner_weights[corner];
        }
        const __nv_bfloat162 residual = x[x_base + pair];
        x[x_base + pair] =
            __floats2bfloat162_rn(__low2float(residual) + lo, __high2float(residual) + hi);
    }
}

template <int Block>
__launch_bounds__(Block) __global__
    void embedding_interpolate_add_d1152_kernel(const __nv_bfloat162* table,
                                                const std::int32_t* indices, const float* weights,
                                                __nv_bfloat162* x, std::int32_t patches) {
    constexpr int d     = 1152;
    constexpr int pairs = d / 2;
    const int patch     = static_cast<int>(blockIdx.x);
    if (patch >= patches) { return; }
    __shared__ std::int32_t corner_indices[4];
    __shared__ float corner_weights[4];
    if (threadIdx.x < 4) {
        const std::int64_t control  = static_cast<std::int64_t>(patch) * 4 + threadIdx.x;
        corner_indices[threadIdx.x] = indices[control];
        corner_weights[threadIdx.x] = weights[control];
    }
    __syncthreads();

    const std::int64_t x_base = static_cast<std::int64_t>(patch) * pairs;
    for (int pair = static_cast<int>(threadIdx.x); pair < pairs; pair += Block) {
        float lo = 0.0f;
        float hi = 0.0f;
#pragma unroll
        for (int corner = 0; corner < 4; ++corner) {
            const float2 value = __bfloat1622float2(
                table[static_cast<std::int64_t>(corner_indices[corner]) * pairs + pair]);
            lo += value.x * corner_weights[corner];
            hi += value.y * corner_weights[corner];
        }
        const __nv_bfloat162 residual = x[x_base + pair];
        x[x_base + pair] =
            __floats2bfloat162_rn(__low2float(residual) + lo, __high2float(residual) + hi);
    }
}

__global__ void embedding_interpolate_add_kernel(const __nv_bfloat16* table,
                                                 const std::int32_t* indices, const float* weights,
                                                 __nv_bfloat16* x, std::int32_t d,
                                                 std::int32_t patches, std::int32_t table_rows,
                                                 std::int64_t n) {
    const std::int64_t start  = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t linear = start; linear < n; linear += stride) {
        const std::int32_t row   = static_cast<std::int32_t>(linear % d);
        const std::int32_t patch = static_cast<std::int32_t>(linear / d);
        float position           = 0.0f;
#pragma unroll
        for (int corner = 0; corner < 4; ++corner) {
            const std::int64_t control = static_cast<std::int64_t>(patch) * 4 + corner;
            const std::int32_t index   = indices[control];
            if (index >= 0 && index < table_rows) {
                position += __bfloat162float(table[static_cast<std::int64_t>(index) * d + row]) *
                            weights[control];
            }
        }
        x[linear] = __float2bfloat16_rn(__bfloat162float(x[linear]) + position);
    }
}

constexpr int kD1152Pairs         = 1152 / 2;
constexpr int kSmallPTargetBlocks = 4096;
constexpr int kWarpPairs          = 32;

int small_p_tiles(int patches) {
    return std::min(common::div_up(kD1152Pairs, kWarpPairs),
                    std::max(1, common::div_up(kSmallPTargetBlocks, patches)));
}

using InterpolateLaunch = void (*)(const Tensor&, const Tensor&, const Tensor&, Tensor&,
                                   cudaStream_t);

void launch_interpolate_warp(const Tensor& table, const Tensor& indices, const Tensor& weights,
                             Tensor& x, cudaStream_t stream) {
    const int tiles = small_p_tiles(x.ne[1]);
    embedding_interpolate_add_d1152_warp_kernel<<<x.ne[1] * tiles, 32, 0, stream>>>(
        static_cast<const __nv_bfloat162*>(table.data),
        static_cast<const std::int32_t*>(indices.data), static_cast<const float*>(weights.data),
        static_cast<__nv_bfloat162*>(x.data), x.ne[1], tiles);
}

void launch_interpolate_cta(const Tensor& table, const Tensor& indices, const Tensor& weights,
                            Tensor& x, cudaStream_t stream) {
    constexpr int block = 128;
    embedding_interpolate_add_d1152_kernel<block><<<x.ne[1], block, 0, stream>>>(
        static_cast<const __nv_bfloat162*>(table.data),
        static_cast<const std::int32_t*>(indices.data), static_cast<const float*>(weights.data),
        static_cast<__nv_bfloat162*>(x.data), x.ne[1]);
}

void launch_interpolate_scalar(const Tensor& table, const Tensor& indices, const Tensor& weights,
                               Tensor& x, cudaStream_t stream) {
    constexpr int block = 128;
    const auto count    = x.numel();
    const int grid      = static_cast<int>(std::min<std::int64_t>(
        common::div_up(count, std::int64_t{block}), std::numeric_limits<int>::max()));
    embedding_interpolate_add_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(table.data),
        static_cast<const std::int32_t*>(indices.data), static_cast<const float*>(weights.data),
        static_cast<__nv_bfloat16*>(x.data), x.ne[0], x.ne[1], table.ne[1], count);
}

InterpolateLaunch select_interpolate_add(const Tensor& table, const Tensor& x) {
    if (x.ne[0] == 1152 && common::operand::aligned(table.data, 4) &&
        common::operand::aligned(x.data, 4))
        return x.ne[1] < 1024 ? launch_interpolate_warp : launch_interpolate_cta;
    return launch_interpolate_scalar;
}

} // namespace

void embedding_interpolate_add(const Tensor& table, const Tensor& indices, const Tensor& weights,
                               Tensor& x, CudaExecutionView execution) {
    const auto stream = execution.stream();
    if (table.dtype != DType::BF16 || x.dtype != DType::BF16 || indices.dtype != DType::I32 ||
        weights.dtype != DType::FP32) {
        throw std::invalid_argument(
            "embedding_interpolate_add: table/x must be BF16, indices I32, weights FP32");
    }
    if (table.ne[2] != 1 || table.ne[3] != 1 || x.ne[2] != 1 || x.ne[3] != 1 ||
        indices.ne[0] != 4 || indices.ne[1] != x.ne[1] || indices.ne[2] != 1 ||
        indices.ne[3] != 1 || weights.ne[0] != 4 || weights.ne[1] != x.ne[1] ||
        weights.ne[2] != 1 || weights.ne[3] != 1 || table.ne[0] != x.ne[0]) {
        throw std::invalid_argument(
            "embedding_interpolate_add: expected table [D,R], controls [4,P], x [D,P]");
    }
    if (x.ne[0] < 0 || x.ne[1] < 0 || table.ne[1] <= 0) {
        throw std::invalid_argument("embedding_interpolate_add: invalid dimensions");
    }
    if (x.ne[0] == 0 || x.ne[1] == 0) { return; }
    if (table.device != execution.device() || indices.device != execution.device() ||
        weights.device != execution.device() || x.device != execution.device()) {
        throw std::invalid_argument(
            "embedding_interpolate_add: operands must belong to the execution device");
    }
    if (!table.is_contiguous() || !indices.is_contiguous() || !weights.is_contiguous() ||
        !x.is_contiguous()) {
        throw std::invalid_argument("embedding_interpolate_add: tensors must be contiguous");
    }
    if (table.data == nullptr || indices.data == nullptr || weights.data == nullptr ||
        x.data == nullptr) {
        throw std::invalid_argument("embedding_interpolate_add: tensor data must be non-null");
    }
    select_interpolate_add(table, x)(table, indices, weights, x, stream);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
