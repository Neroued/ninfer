#include "ninfer/ops/layer_norm.h"
#include "core/device.h"
#include "core/platform.h"
#include "ops/common/host/operands.h"
#include "ops/common/cuda/reduce.cuh"
#include <cuda_bf16.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::ops {
namespace {
template <int Block>
__launch_bounds__(Block) __global__
    void layer_norm_d1152_warp_kernel(const __nv_bfloat162* x, const __nv_bfloat162* weight,
                                      const __nv_bfloat162* bias, __nv_bfloat162* out,
                                      std::int64_t rows, float eps) {
    constexpr int d        = 1152;
    constexpr int pairs    = d / 2;
    constexpr int warps    = Block / 32;
    const int lane         = static_cast<int>(threadIdx.x) & 31;
    const int warp         = static_cast<int>(threadIdx.x) >> 5;
    const std::int64_t row = static_cast<std::int64_t>(blockIdx.x) * warps + warp;
    if (row >= rows) { return; }
    const std::int64_t base = row * pairs;

    common::WelfordMoments local{0.0f, 0.0f, 0};
    for (int pair = lane; pair < pairs; pair += 32) {
        const float2 values = __bfloat1622float2(x[base + pair]);
        common::welford_update(local, values.x);
        common::welford_update(local, values.y);
    }
    local                   = common::warp_reduce_moments(local);
    constexpr unsigned mask = 0xffffffffu;
    const float mean        = __shfl_sync(mask, local.mean, 0);
    const float m2          = __shfl_sync(mask, local.m2, 0);
    const float inv         = rsqrtf(m2 / static_cast<float>(d) + eps);

    for (int pair = lane; pair < pairs; pair += 32) {
        const float2 values = __bfloat1622float2(x[base + pair]);
        const float2 scales = __bfloat1622float2(weight[pair]);
        const float2 shifts = __bfloat1622float2(bias[pair]);
        out[base + pair]    = __floats2bfloat162_rn((values.x - mean) * inv * scales.x + shifts.x,
                                                    (values.y - mean) * inv * scales.y + shifts.y);
    }
}

template <int Block>
__global__ void layer_norm_kernel(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                                  const __nv_bfloat16* bias, __nv_bfloat16* out, std::int32_t d,
                                  std::int64_t rows, float eps) {
    const std::int64_t row = static_cast<std::int64_t>(blockIdx.x);
    if (row >= rows) { return; }
    const std::int64_t base = row * d;

    common::WelfordMoments local{0.0f, 0.0f, 0};
    for (std::int32_t i = static_cast<std::int32_t>(threadIdx.x); i < d; i += Block) {
        const float value = __bfloat162float(x[base + i]);
        common::welford_update(local, value);
    }

    __shared__ float means[Block];
    __shared__ float m2s[Block];
    __shared__ int counts[Block];
    means[threadIdx.x]  = local.mean;
    m2s[threadIdx.x]    = local.m2;
    counts[threadIdx.x] = local.count;
    __syncthreads();

    for (int offset = Block / 2; offset > 0; offset >>= 1) {
        if (static_cast<int>(threadIdx.x) < offset) {
            const common::WelfordMoments a{means[threadIdx.x], m2s[threadIdx.x],
                                           counts[threadIdx.x]};
            const common::WelfordMoments b{means[threadIdx.x + offset], m2s[threadIdx.x + offset],
                                           counts[threadIdx.x + offset]};
            const common::WelfordMoments merged = common::welford_merge(a, b);
            means[threadIdx.x]                  = merged.mean;
            m2s[threadIdx.x]                    = merged.m2;
            counts[threadIdx.x]                 = merged.count;
        }
        __syncthreads();
    }

    const float mean = means[0];
    const float inv  = rsqrtf(m2s[0] / static_cast<float>(d) + eps);
    for (std::int32_t i = static_cast<std::int32_t>(threadIdx.x); i < d; i += Block) {
        const float value = (__bfloat162float(x[base + i]) - mean) * inv;
        out[base + i] =
            __float2bfloat16_rn(value * __bfloat162float(weight[i]) + __bfloat162float(bias[i]));
    }
}

template <int Block>
__global__ void layer_norm_d1152_centered_kernel(const __nv_bfloat162* x,
                                                 const __nv_bfloat162* weight,
                                                 const __nv_bfloat162* bias, __nv_bfloat162* out,
                                                 std::int64_t rows, float eps) {
    constexpr int Pairs = 576;
    constexpr int Items = (Pairs + Block - 1) / Block;
    const int lane      = threadIdx.x;
    const auto row      = std::int64_t(blockIdx.x);
    if (row >= rows) return;
    const auto base = row * Pairs;
    // Center around a represented input before reducing, including large constant rows.
    const float anchor = __low2float(x[base]);
    float2 values[Items];
    float sum = 0;
#pragma unroll
    for (int i = 0; i < Items; ++i) {
        const int pair = lane + i * Block;
        if (pair < Pairs) {
            const auto v = __bfloat1622float2(x[base + pair]);
            values[i]    = {v.x - anchor, v.y - anchor};
            sum += values[i].x + values[i].y;
        }
    }
    __shared__ float sums[Block / 32];
    __shared__ float scalar;
    const auto reduce = [&](float value) {
        value = common::block_reduce_sum<Block>(value, sums);
        if (threadIdx.x == 0) scalar = value;
        __syncthreads();
        return scalar;
    };
    const float mean = reduce(sum) / 1152.f;
    float squares    = 0;
#pragma unroll
    for (int i = 0; i < Items; ++i) {
        if (lane + i * Block < Pairs) {
            values[i].x -= mean;
            values[i].y -= mean;
            squares = fmaf(values[i].x, values[i].x, squares);
            squares = fmaf(values[i].y, values[i].y, squares);
        }
    }
    const float inverse = rsqrtf(reduce(squares) / 1152.f + eps);
#pragma unroll
    for (int i = 0; i < Items; ++i) {
        const int pair = lane + i * Block;
        if (pair < Pairs) {
            const auto w     = __bfloat1622float2(weight[pair]);
            const auto b     = __bfloat1622float2(bias[pair]);
            out[base + pair] = __floats2bfloat162_rn(values[i].x * inverse * w.x + b.x,
                                                     values[i].y * inverse * w.y + b.y);
        }
    }
}

#if NINFER_TARGET_SPARK
void launch_layer_norm_centered(const Tensor& x, const Tensor& weight, const Tensor& bias,
                                float eps, Tensor& out, cudaStream_t stream) {
    const auto rows = x.numel() / 1152;
    layer_norm_d1152_centered_kernel<128><<<static_cast<unsigned>(rows), 128, 0, stream>>>(
        static_cast<const __nv_bfloat162*>(x.data), static_cast<const __nv_bfloat162*>(weight.data),
        static_cast<const __nv_bfloat162*>(bias.data), static_cast<__nv_bfloat162*>(out.data), rows,
        eps);
}
#endif

using LayerNormLaunch = void (*)(const Tensor&, const Tensor&, const Tensor&, float, Tensor&,
                                 cudaStream_t);

template <bool Paired>
void launch_layer_norm(const Tensor& x, const Tensor& weight, const Tensor& bias, float eps,
                       Tensor& out, cudaStream_t stream) {
    const auto rows = x.numel() / x.ne[0];
    if constexpr (Paired) {
        constexpr int block = 128;
        const auto grid     = static_cast<unsigned>((rows + block / 32 - 1) / (block / 32));
        layer_norm_d1152_warp_kernel<block>
            <<<grid, block, 0, stream>>>(static_cast<const __nv_bfloat162*>(x.data),
                                         static_cast<const __nv_bfloat162*>(weight.data),
                                         static_cast<const __nv_bfloat162*>(bias.data),
                                         static_cast<__nv_bfloat162*>(out.data), rows, eps);
    } else {
        constexpr int block = 256;
        layer_norm_kernel<block><<<static_cast<unsigned>(rows), block, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const __nv_bfloat16*>(weight.data),
            static_cast<const __nv_bfloat16*>(bias.data), static_cast<__nv_bfloat16*>(out.data),
            x.ne[0], rows, eps);
    }
}

LayerNormLaunch select_layer_norm(const Tensor& x, const Tensor& weight, const Tensor& bias,
                                  const Tensor& out) {
    if (x.ne[0] == 1152 && common::operand::aligned(x.data, 4) &&
        common::operand::aligned(weight.data, 4) && common::operand::aligned(bias.data, 4) &&
        common::operand::aligned(out.data, 4)) {
#if NINFER_TARGET_SPARK
        return launch_layer_norm_centered;
#else
        return launch_layer_norm<true>;
#endif
    }
    return launch_layer_norm<false>;
}

bool vector_of(const Tensor& t, std::int32_t d) {
    return t.ne[0] == d && t.ne[1] == 1 && t.ne[2] == 1 && t.ne[3] == 1;
}

} // namespace

void layer_norm(const Tensor& x, const Tensor& weight, const Tensor& bias, float eps, Tensor& out,
                CudaExecutionView execution) {
    const auto stream = execution.stream();
    if (x.dtype != DType::BF16 || weight.dtype != DType::BF16 || bias.dtype != DType::BF16 ||
        out.dtype != DType::BF16) {
        throw std::invalid_argument("layer_norm: x/weight/bias/out must be BF16");
    }
    if (!(eps > 0.0f) || !std::isfinite(eps)) {
        throw std::invalid_argument("layer_norm: eps must be positive and finite");
    }
    const std::int64_t n = common::operand::numel(x, "layer_norm");
    (void)common::operand::numel(weight, "layer_norm");
    (void)common::operand::numel(bias, "layer_norm");
    (void)common::operand::numel(out, "layer_norm");
    for (int i = 0; i < 4; ++i) {
        if (out.ne[i] != x.ne[i]) {
            throw std::invalid_argument("layer_norm: x/out shapes must match");
        }
    }
    if (!vector_of(weight, x.ne[0]) || !vector_of(bias, x.ne[0])) {
        throw std::invalid_argument("layer_norm: weight/bias must have shape [x.ne[0]]");
    }
    if (n == 0) { return; }
    if (x.device != execution.device() || weight.device != execution.device() ||
        bias.device != execution.device() || out.device != execution.device()) {
        throw std::invalid_argument("layer_norm: operands must belong to the execution device");
    }
    if (!x.is_contiguous() || !weight.is_contiguous() || !bias.is_contiguous() ||
        !out.is_contiguous()) {
        throw std::invalid_argument("layer_norm: tensors must be contiguous");
    }
    if (x.data == nullptr || weight.data == nullptr || bias.data == nullptr ||
        out.data == nullptr) {
        throw std::invalid_argument("layer_norm: tensor data must be non-null");
    }
    if (n / x.ne[0] > std::numeric_limits<unsigned>::max())
        throw std::overflow_error("layer_norm: launch grid exceeds CUDA limit");
    select_layer_norm(x, weight, bias, out)(x, weight, bias, eps, out, stream);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
