#include "ninfer/ops/l2norm.h"
#include "core/device.h"
#include "ops/common/host/operands.h"
#include "ops/common/numeric/math.h"
#include "ops/common/cuda/reduce.cuh"
#include <cuda_bf16.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::ops {
namespace {
// Fast domain: D in {64, 128, 192, 256}. One warp owns one row and keeps the input in registers.
template <int Block>
__launch_bounds__(Block) __global__
    void l2norm_warp_bf16x2_kernel(const __nv_bfloat162* x, __nv_bfloat162* out, std::int32_t d,
                                   std::int64_t rows, float eps) {
    static_assert(Block % common::kWarpSize == 0);
    constexpr int kWarpsPerBlock   = Block / common::kWarpSize;
    constexpr int kMaxPairsPerLane = 4;
    const int lane                 = static_cast<int>(threadIdx.x) & (common::kWarpSize - 1);
    const int warp                 = static_cast<int>(threadIdx.x) / common::kWarpSize;
    const std::int64_t row         = static_cast<std::int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
    if (row >= rows) { return; }

    const int pairs             = d / 2;
    const std::int64_t row_base = row * static_cast<std::int64_t>(pairs);
    __nv_bfloat162 values[kMaxPairsPerLane];
    float sum = 0.0f;

#pragma unroll
    for (int k = 0; k < kMaxPairsPerLane; ++k) {
        const int pair = lane + k * common::kWarpSize;
        if (pair < pairs) {
            values[k]       = x[row_base + pair];
            const float2 xf = __bfloat1622float2(values[k]);
            sum += xf.x * xf.x + xf.y * xf.y;
        }
    }

    sum       = common::warp_reduce_sum(sum);
    float inv = lane == 0 ? rsqrtf(sum + eps) : 0.0f;
    inv       = __shfl_sync(common::kFullWarpMask, inv, 0);

#pragma unroll
    for (int k = 0; k < kMaxPairsPerLane; ++k) {
        const int pair = lane + k * common::kWarpSize;
        if (pair < pairs) {
            const float2 xf      = __bfloat1622float2(values[k]);
            out[row_base + pair] = __floats2bfloat162_rn(xf.x * inv, xf.y * inv);
        }
    }
}

// Functional fallback for unaligned data and widths outside the aligned fast domain.
__launch_bounds__(512) __global__
    void l2norm_generic_kernel(const __nv_bfloat16* x, __nv_bfloat16* out, std::int32_t d,
                               std::int64_t rows, float eps) {
    constexpr int kRowsPerBlock = 512 / common::kWarpSize;
    const int lane              = static_cast<int>(threadIdx.x) & (common::kWarpSize - 1);
    const int warp              = static_cast<int>(threadIdx.x) / common::kWarpSize;
    const std::int64_t row      = static_cast<std::int64_t>(blockIdx.x) * kRowsPerBlock + warp;
    if (row >= rows) { return; }

    const std::int64_t base = row * static_cast<std::int64_t>(d);
    float sum               = 0.0f;
    for (std::int64_t i = lane; i < static_cast<std::int64_t>(d); i += common::kWarpSize) {
        const float value = __bfloat162float(x[base + i]);
        sum += value * value;
    }

    sum       = common::warp_reduce_sum(sum);
    float inv = lane == 0 ? rsqrtf(sum + eps) : 0.0f;
    inv       = __shfl_sync(common::kFullWarpMask, inv, 0);
    for (std::int64_t i = lane; i < static_cast<std::int64_t>(d); i += common::kWarpSize) {
        const std::int64_t index = base + i;
        out[index]               = __float2bfloat16_rn(__bfloat162float(x[index]) * inv);
    }
}

using L2NormLaunch = void (*)(const Tensor&, float, Tensor&, cudaStream_t);

template <bool Paired>
void launch_l2norm(const Tensor& x, float eps, Tensor& out, cudaStream_t stream) {
    constexpr int block = 512;
    const auto rows     = x.numel() / x.ne[0];
    const auto grid =
        static_cast<unsigned>(common::div_up(rows, std::int64_t{block / common::kWarpSize}));
    if constexpr (Paired)
        l2norm_warp_bf16x2_kernel<block><<<grid, block, 0, stream>>>(
            static_cast<const __nv_bfloat162*>(x.data), static_cast<__nv_bfloat162*>(out.data),
            x.ne[0], rows, eps);
    else
        l2norm_generic_kernel<<<grid, block, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                                          static_cast<__nv_bfloat16*>(out.data),
                                                          x.ne[0], rows, eps);
}

L2NormLaunch select_l2norm(const Tensor& x, const Tensor& out) {
    if (x.ne[0] >= 64 && x.ne[0] <= 256 && x.ne[0] % 64 == 0 &&
        common::operand::aligned(x.data, 4) && common::operand::aligned(out.data, 4))
        return launch_l2norm<true>;
    return launch_l2norm<false>;
}

void require_same_shape(const Tensor& a, const Tensor& b) {
    for (int d = 0; d < 4; ++d) {
        if (a.ne[d] != b.ne[d]) { throw std::invalid_argument("l2norm: x/out shapes must match"); }
    }
}

} // namespace

void l2norm(const Tensor& x, float eps, Tensor& out, CudaExecutionView execution) {
    const auto stream = execution.stream();
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("l2norm: x/out must be BF16");
    }
    if (!(eps > 0.0f) || !std::isfinite(eps)) {
        throw std::invalid_argument("l2norm: eps must be positive and finite");
    }

    const std::int64_t n = common::operand::numel(x, "l2norm");
    (void)common::operand::numel(out, "l2norm");
    require_same_shape(x, out);
    if (n == 0) { return; }
    if (x.device != execution.device() || out.device != execution.device()) {
        throw std::invalid_argument("l2norm: operands must belong to the execution device");
    }

    const std::int64_t rows = n / x.ne[0];
    if (rows > std::numeric_limits<int>::max()) {
        throw std::overflow_error("l2norm: row count exceeds CUDA grid limit");
    }

    if (!x.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("l2norm: x/out must be contiguous");
    }
    if (x.data == nullptr || out.data == nullptr) {
        throw std::invalid_argument("l2norm: x/out data must be non-null");
    }

    select_l2norm(x, out)(x, eps, out, stream);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
