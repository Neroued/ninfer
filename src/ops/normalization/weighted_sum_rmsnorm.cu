#include "ninfer/ops/weighted_sum_rmsnorm.h"

#include "ops/common/cuda/reduce.cuh"
#include "ops/common/host/operands.h"
#include "ops/common/numeric/convert.cuh"

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {
namespace {

using Bf16        = __nv_bfloat16;
namespace operand = common::operand;

template <class Alpha>
__device__ __forceinline__ float alpha_at(const Alpha* alpha, int k) {
    return common::load_state_scalar(alpha + k);
}

template <int Block, bool Maximum = false>
__device__ __forceinline__ float row_reduce(float value, float* warp_values) {
    if constexpr (Block == 32) {
        if constexpr (Maximum)
            return common::warp_max(value);
        else
            return common::warp_sum(value);
    } else {
        constexpr int warps = Block / 32;
        const int lane      = threadIdx.x % 32;
        if constexpr (Maximum)
            value = common::warp_max(value);
        else
            value = common::warp_reduce_sum(value);
        if (lane == 0) warp_values[threadIdx.x / 32] = value;
        __syncthreads();
        value = warp_values[lane % warps];
        if constexpr (Maximum)
            return common::warp_max<warps>(value);
        else
            return common::warp_sum<warps>(value);
    }
}

// Each product rounds before the sum so opposite represented products cancel exactly.
__device__ __forceinline__ float accumulate_tap(float mixed, float alpha, float x) {
    return __fadd_rn(mixed, __fmul_rn(alpha, x));
}

template <int Block, class Visit>
__device__ __forceinline__ float inverse_rms(float total_sq, float inverse_d, float eps,
                                             const Visit& visit, float* warp_sums,
                                             float* warp_maxima) {
    const float denominator = fmaf(total_sq, inverse_d, eps);
    if (isfinite(denominator)) return rsqrtf(denominator);

    float max_abs = 0.f;
    visit([&](float value) { max_abs = fmaxf(max_abs, fabsf(value)); });
    max_abs = row_reduce<Block, true>(max_abs, warp_maxima);
    if (!isfinite(max_abs) || max_abs == 0.f) return rsqrtf(denominator);

    const float scale = __frcp_rn(max_abs);
    float scaled_sq   = 0.f;
    visit([&](float value) {
        const float scaled = value * scale;
        scaled_sq          = fmaf(scaled, scaled, scaled_sq);
    });
    return scale *
           rsqrtf(fmaf(row_reduce<Block>(scaled_sq, warp_sums), inverse_d, (eps * scale) * scale));
}

template <int Block, int Items, int FixedTaps, class Alpha, int Width = 8>
__launch_bounds__(Block) __global__
    void weighted_rmsnorm_tile(const Bf16* __restrict__ inputs, const Alpha* __restrict__ alphas,
                               const Bf16* __restrict__ weight, Bf16* __restrict__ out, int packs,
                               int taps, float inverse_d, float eps) {
    const int lane         = threadIdx.x;
    const auto row         = std::int64_t(blockIdx.x);
    const int count        = FixedTaps ? FixedTaps : taps;
    const auto row_stride  = std::int64_t(packs) * Width;
    const Bf16* row_inputs = inputs + row * count * row_stride;
    Bf16* row_out          = out + row * row_stride;
    float mixed[Items][Width]{};

#pragma unroll(FixedTaps ? FixedTaps : 1)
    for (int tap = 0; tap < count; ++tap) {
        const float alpha     = alpha_at(alphas, tap);
        const Bf16* tap_input = row_inputs + std::int64_t(tap) * row_stride;
#pragma unroll
        for (int item = 0; item < Items; ++item) {
            const int pack = lane + item * Block;
            if (pack < packs) {
                if constexpr (Width == 1) {
                    mixed[item][0] =
                        accumulate_tap(mixed[item][0], alpha, __bfloat162float(tap_input[pack]));
                } else {
                    const common::Bf16x8Bits input{
                        .raw = common::load_vec<uint4>(tap_input + pack * Width)};
#pragma unroll
                    for (int pair = 0; pair < 4; ++pair) {
                        const float2 x        = __bfloat1622float2(input.pair[pair]);
                        mixed[item][2 * pair] = accumulate_tap(mixed[item][2 * pair], alpha, x.x);
                        mixed[item][2 * pair + 1] =
                            accumulate_tap(mixed[item][2 * pair + 1], alpha, x.y);
                    }
                }
            }
        }
    }

    float sum = 0.f, odd = 0.f;
#pragma unroll
    for (int item = 0; item < Items; ++item) {
#pragma unroll
        for (int i = 0; i < Width; ++i) {
            if (i % 2 == 0)
                sum = fmaf(mixed[item][i], mixed[item][i], sum);
            else
                odd = fmaf(mixed[item][i], mixed[item][i], odd);
        }
    }
    const auto visit = [&](auto consume) {
#pragma unroll
        for (int item = 0; item < Items; ++item) {
#pragma unroll
            for (int i = 0; i < Width; ++i) consume(mixed[item][i]);
        }
    };
    __shared__ float warp_sums[Block / 32], warp_maxima[Block / 32];
    const float total_sq = row_reduce<Block>(sum + odd, warp_sums);
    const float inverse =
        inverse_rms<Block>(total_sq, inverse_d, eps, visit, warp_sums, warp_maxima);
#pragma unroll
    for (int item = 0; item < Items; ++item) {
        const int pack = lane + item * Block;
        if (pack < packs) {
            if constexpr (Width == 1) {
                row_out[pack] =
                    __float2bfloat16_rn(mixed[item][0] * inverse * __bfloat162float(weight[pack]));
            } else {
                const common::Bf16x8Bits gain{.raw =
                                                  common::load_vec<uint4>(weight + pack * Width)};
                common::Bf16x8Bits result{.raw = {}};
#pragma unroll
                for (int pair = 0; pair < 4; ++pair) {
                    const float2 w = __bfloat1622float2(gain.pair[pair]);
                    result.pair[pair] =
                        __floats2bfloat162_rn(mixed[item][2 * pair] * inverse * w.x,
                                              mixed[item][2 * pair + 1] * inverse * w.y);
                }
                common::store_vec(row_out + pack * Width, result.raw);
            }
        }
    }
}

template <class Alpha>
__device__ __forceinline__ float mix_channel(const Bf16* input, const Alpha* alphas, int d,
                                             int taps) {
    float mixed = 0.f;
    for (int k = 0; k < taps; ++k)
        mixed = accumulate_tap(mixed, alpha_at(alphas, k),
                               __bfloat162float(input[std::int64_t(k) * d]));
    return mixed;
}

template <int Block, class Alpha>
__launch_bounds__(Block) __global__
    void weighted_rmsnorm_scalar(const Bf16* __restrict__ inputs, const Alpha* __restrict__ alphas,
                                 const Bf16* __restrict__ weight, Bf16* __restrict__ out, int d,
                                 int taps, float inverse_d, float eps) {
    const auto row         = std::int64_t(blockIdx.x);
    const Bf16* row_inputs = inputs + row * taps * d;
    Bf16* row_out          = out + row * d;

    float sum = 0.f;
    for (std::int64_t i = threadIdx.x; i < d; i += Block) {
        const float mix = mix_channel(row_inputs + i, alphas, d, taps);
        sum             = fmaf(mix, mix, sum);
    }
    const auto visit = [&](auto consume) {
        for (std::int64_t i = threadIdx.x; i < d; i += Block)
            consume(mix_channel(row_inputs + i, alphas, d, taps));
    };
    __shared__ float warp_sums[Block / 32], warp_maxima[Block / 32];
    const float total_sq = row_reduce<Block>(sum, warp_sums);
    const float inverse =
        inverse_rms<Block>(total_sq, inverse_d, eps, visit, warp_sums, warp_maxima);

    // Recompute the mix for rows wider than a register tile.
    for (std::int64_t i = threadIdx.x; i < d; i += Block) {
        const float mix = mix_channel(row_inputs + i, alphas, d, taps);
        row_out[i]      = __float2bfloat16_rn(mix * inverse * __bfloat162float(weight[i]));
    }
}

template <int Block, int Items, class Alpha>
void launch_vector(const Bf16* inputs, const Alpha* alphas, const Bf16* weight, Bf16* out, int d,
                   int taps, int m, float inverse_d, float eps, cudaStream_t stream) {
    const auto launch = [&]<int FixedTaps> {
        weighted_rmsnorm_tile<Block, Items, FixedTaps>
            <<<m, Block, 0, stream>>>(inputs, alphas, weight, out, d / 8, taps, inverse_d, eps);
    };
    switch (taps) {
    case 1:
        launch.template operator()<1>();
        break;
    case 2:
        launch.template operator()<2>();
        break;
    case 3:
        launch.template operator()<3>();
        break;
    case 4:
        launch.template operator()<4>();
        break;
    case 5:
        launch.template operator()<5>();
        break;
    case 6:
        launch.template operator()<6>();
        break;
    case 7:
        launch.template operator()<7>();
        break;
    case 8:
        launch.template operator()<8>();
        break;
    default:
        launch.template operator()<0>();
        break;
    }
}

template <class Alpha>
void dispatch(const Tensor& inputs, const Tensor& alphas, const Tensor& weight, Tensor& out,
              float eps, cudaStream_t stream) {
    const int d = inputs.ne[0], taps = inputs.ne[1], m = inputs.ne[2];
    const auto* x         = static_cast<const Bf16*>(inputs.data);
    const auto* a         = static_cast<const Alpha*>(alphas.data);
    const auto* w         = static_cast<const Bf16*>(weight.data);
    auto* y               = static_cast<Bf16*>(out.data);
    const float inverse_d = 1.f / static_cast<float>(d);
    if (d % 8 == 0 && d <= 4096 && operand::aligned(x, 16) && operand::aligned(w, 16) &&
        operand::aligned(y, 16)) {
        if (d <= 256)
            launch_vector<32, 1>(x, a, w, y, d, taps, m, inverse_d, eps, stream);
        else if (d <= 512)
            launch_vector<64, 1>(x, a, w, y, d, taps, m, inverse_d, eps, stream);
        else if (d <= 1024)
            launch_vector<128, 1>(x, a, w, y, d, taps, m, inverse_d, eps, stream);
        else if (d <= 2048)
            launch_vector<128, 2>(x, a, w, y, d, taps, m, inverse_d, eps, stream);
        else
            // A wider CTA keeps one pack per lane across rows spanning 257–512 packs.
            launch_vector<512, 1>(x, a, w, y, d, taps, m, inverse_d, eps, stream);
    } else if (d <= 4096) {
        const auto launch = [&]<int Block, int Items> {
            weighted_rmsnorm_tile<Block, Items, 0, Alpha, 1>
                <<<m, Block, 0, stream>>>(x, a, w, y, d, taps, inverse_d, eps);
        };
        if (d <= 32)
            launch.template operator()<32, 1>();
        else if (d <= 128)
            launch.template operator()<128, 1>();
        else if (d <= 512)
            launch.template operator()<128, 4>();
        else if (d <= 1024)
            launch.template operator()<128, 8>();
        else if (d <= 2048)
            launch.template operator()<128, 16>();
        else
            launch.template operator()<128, 32>();
    } else {
        constexpr int block = 128;
        weighted_rmsnorm_scalar<block>
            <<<m, block, 0, stream>>>(x, a, w, y, d, taps, inverse_d, eps);
    }
}

} // namespace

void weighted_sum_rmsnorm(const Tensor& inputs, const Tensor& alphas, const Tensor& weight,
                          float eps, Tensor& out, CudaExecutionView execution) {
    constexpr char op[] = "weighted_sum_rmsnorm";
    const int d = inputs.ne[0], taps = inputs.ne[1], m = inputs.ne[2];
    if (d <= 0 || taps <= 0 || m < 0 || !(eps > 0.f) || !std::isfinite(eps))
        throw std::invalid_argument("weighted_sum_rmsnorm: invalid extents or epsilon");
    operand::shape(inputs, {d, taps, m, 1}, op, "inputs");
    operand::shape(out, {d, m, 1, 1}, op, "out");
    operand::dtype(inputs, DType::BF16, op, "inputs must be BF16");
    operand::dtype(out, DType::BF16, op, "out must be BF16");
    if (alphas.dtype != DType::FP32 && alphas.dtype != DType::BF16)
        throw std::invalid_argument("weighted_sum_rmsnorm: alphas must be FP32 or BF16");
    const auto device = execution.device();
    operand::tensor(alphas, alphas.dtype, {taps}, dtype_size(alphas.dtype), op, "alphas", device);
    operand::tensor(weight, DType::BF16, {d}, 2, op, "weight", device);
    if (m == 0) return;

    operand::tensor(inputs, DType::BF16, {d, taps, m}, 2, op, "inputs", device);
    operand::tensor(out, DType::BF16, {d, m}, 2, op, "out", device);
    const auto output = operand::range(out, op);
    for (const auto* input : {&inputs, &alphas, &weight})
        operand::disjoint(operand::range(*input, op), output, op);

    if (alphas.dtype == DType::FP32)
        dispatch<float>(inputs, alphas, weight, out, eps, execution.stream());
    else
        dispatch<Bf16>(inputs, alphas, weight, out, eps, execution.stream());
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
