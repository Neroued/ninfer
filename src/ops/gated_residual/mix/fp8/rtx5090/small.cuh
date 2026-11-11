#pragma once

#include "ops/gated_residual/mix/fp8/kernel/output.cuh"
#include "ops/linear/kernels/fp8/fp8_a8_mma_common.cuh"
#include "ops/common/cuda/pipeline.cuh"

namespace ninfer::ops::detail::hc::mix::fp8 {

inline constexpr int kSegmentThreads = 64, kSegmentK = kTotal / kDownPartitions;
using SegmentedDown =
    Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<kSmallColumns, 32, 128, 1, 2, 3, 2>, 340,
                           kDownPartitions, 1>;
static_assert(kSegmentK % SegmentedDown::kBlockK == 0);

inline int down_control_elements(int columns, bool injection) {
    return columns * kDownPartitions * (injection ? kStreams + 1 : 1);
}

// Each K partition owns its activation scale and four original-BF16 injection dots.
// Those scales are applied to the corresponding raw MMA partials before K reduction.
template <bool Injection>
__global__ __launch_bounds__(kSegmentThreads) void encode_segments(const __nv_bfloat16* input,
                                                                   const __nv_bfloat16* weight,
                                                                   std::uint8_t* codes,
                                                                   float* control, int columns) {
    constexpr int Slots = kSegmentK / (2 * kSegmentThreads);
    constexpr int Warps = kSegmentThreads / common::kWarpSize;
    static_assert(kSegmentK % (2 * kSegmentThreads) == 0);
    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    const int token = blockIdx.x, part = blockIdx.y;
    __nv_bfloat162 values[Slots];
    float dots[4]{}, maximum = 0;
#pragma unroll
    for (int i = 0; i < Slots; ++i) {
        const int k = part * kSegmentK + 2 * (tid + i * kSegmentThreads);
        values[i] =
            *reinterpret_cast<const __nv_bfloat162*>(input + std::int64_t(token) * kTotal + k);
        const auto value = __bfloat1622float2(values[i]);
        maximum          = fmaxf(maximum, fmaxf(fabsf(value.x), fabsf(value.y)));
        if constexpr (Injection) {
#pragma unroll
            for (int stream = 0; stream < kStreams; ++stream) {
                const auto w = __bfloat1622float2(
                    *reinterpret_cast<const __nv_bfloat162*>(weight + stream * kTotal + k));
                dots[stream] = fmaf(value.x, w.x, dots[stream]);
                dots[stream] = fmaf(value.y, w.y, dots[stream]);
            }
        }
    }
    __shared__ float maxima[Warps], sums[kStreams][Warps];
    maximum = common::warp_max(maximum);
    if (lane == 0) maxima[warp] = maximum;
    if constexpr (Injection) {
#pragma unroll
        for (int stream = 0; stream < kStreams; ++stream) {
            const float value = common::warp_sum(dots[stream]);
            if (lane == 0) sums[stream][warp] = value;
        }
    }
    __syncthreads();
    maximum = maxima[0];
#pragma unroll
    for (int i = 1; i < Warps; ++i) maximum = fmaxf(maximum, maxima[i]);
    const float scale   = maximum > 0 ? maximum / 448.F : 0.F;
    const float inverse = scale > 0 ? 1.F / scale : 0.F;
#pragma unroll
    for (int i = 0; i < Slots; ++i) {
        const int k      = part * kSegmentK + 2 * (tid + i * kSegmentThreads);
        const auto value = __bfloat1622float2(values[i]);
        *reinterpret_cast<std::uint16_t*>(codes + std::int64_t(token) * kTotal + k) =
            __nv_cvt_float2_to_fp8x2(make_float2(value.x * inverse, value.y * inverse),
                                     __NV_SATFINITE, __NV_E4M3);
    }
    if (tid == 0) control[token * kDownPartitions + part] = scale;
    if constexpr (Injection) {
        if (tid < kStreams) {
            float sum = sums[tid][0];
#pragma unroll
            for (int i = 1; i < Warps; ++i) sum += sums[tid][i];
            control[columns * kDownPartitions + (token * kDownPartitions + part) * kStreams + tid] =
                sum;
        }
    }
}

// TMA partials have [row tile, partition, token, row] order. The SIMT route
// uses [partition, token, row]. Both finish at the same public FP32 nonlinearities.
template <bool Segmented, bool EncodeUp, bool Injection>
__global__ __launch_bounds__(kUpGroup) void finish_small_down(
    const float* partial, const __nv_bfloat16* weights, const float* control, __nv_bfloat16* u,
    std::uint8_t* codes, float* scales, float* injection, int columns) {
    const int tid = threadIdx.x, token = blockIdx.y, group = blockIdx.x;
    const int row = group * kUpGroup + tid;
    float sum     = 0;
#pragma unroll
    for (int part = 0; part < kDownPartitions; ++part) {
        if constexpr (Segmented) {
            constexpr int Rows = SegmentedDown::kBlockRows, Tokens = SegmentedDown::kBlockTokens;
            const float value = partial[((row / Rows) * kDownPartitions + part) * Tokens * Rows +
                                        token * Rows + row % Rows];
            sum               = fmaf(value, control[token * kDownPartitions + part], sum);
        } else {
            sum += partial[(std::int64_t(part) * columns + token) * (kRank + kStreams) + row];
        }
    }
    const float value = common::silu(sum * __bfloat162float(weights[row]) * .25F);
    if constexpr (EncodeUp) {
        const float maximum = common::block_max_broadcast<kUpGroup>(fabsf(value));
        const float scale   = maximum > 0 ? maximum / 448.F : 0.F;
        const float inverse = scale > 0 ? 1.F / scale : 0.F;
        codes[token * kRank + row] =
            __nv_cvt_float_to_fp8(value * inverse, __NV_SATFINITE, __NV_E4M3);
        if (tid == 0) scales[token * (kRank / kUpGroup) + group] = scale;
    } else {
        u[token * kRank + row] = __float2bfloat16_rn(value);
    }
    if constexpr (Injection) {
        if (group == 0 && tid < kStreams) {
            float dot = 0;
#pragma unroll
            for (int part = 0; part < kDownPartitions; ++part) {
                if constexpr (Segmented)
                    dot += control[columns * kDownPartitions +
                                   (token * kDownPartitions + part) * kStreams + tid];
                else
                    dot += partial[(std::int64_t(part) * columns + token) * (kRank + kStreams) +
                                   kRank + tid];
            }
            injection[token * kStreams + tid] = 2.F * common::sigmoid(dot * .25F);
        }
    }
}

template <int Tokens>
using GroupedUpSchedule =
    Fp8A8MmaSchedule<Tokens, 32, kUpGroup, Tokens / 16, 64 / Tokens, 2, 2, Cache::cg, Cache::ca,
                     Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;

// The five K groups are accumulated in FP32 with their own activation scales.
// Four stream gates stay in one CTA and share the existing fused output epilogue.
template <int Tokens>
__global__ __launch_bounds__(128, 2) void grouped_up_kernel(const std::uint8_t* input,
                                                            const float* input_scales,
                                                            const std::uint8_t* weights,
                                                            const __nv_bfloat16* weight_scales,
                                                            UpOutput output) {
    using S            = GroupedUpSchedule<Tokens>;
    constexpr int Rows = S::kBlockRows, Group = kUpGroup, Groups = kRank / Group;
    constexpr int Threads = S::kThreads;
    auto* shared          = common::shared_storage<S::kSharedBytes>();
    auto* x               = shared;
    auto* w               = x + 2 * Tokens * Group;
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int row_begin = blockIdx.x * Rows;
    const auto stage    = [&](int slot, int group) {
        for (int task = tid; task < (Tokens + Rows) * (Group / 16); task += Threads) {
            int row     = task / (Group / 16);
            const int k = task % (Group / 16) * 16;
            if (row < Tokens) {
                const bool valid = row < output.columns;
                common::cp_async_zfill<16, Cache::ca>(
                    x + (slot * Tokens + row) * Group + fp8_mma_shared_byte<S>(row, k),
                    input + (valid ? row : 0) * kRank + group * Group + k, valid ? 16 : 0);
            } else {
                row -= Tokens;
                const int parent = row / (Rows / 4) * kWidth + row_begin / 4 + row % (Rows / 4);
                common::cp_async<16, Cache::cg>(w + (slot * Rows + row) * Group +
                                                       fp8_mma_shared_byte<S>(row, k),
                                                   weights + parent * kRank + group * Group + k);
            }
        }
        common::cp_commit();
    };
    stage(0, 0);
    stage(1, 1);
    float sum[S::kMmaTokens][S::kMmaRows][4]{};
#pragma unroll
    for (int group = 0; group < Groups; ++group) {
        const int slot = group % 2;
        if (group + 2 <= Groups)
            common::cp_wait<1>();
        else
            common::cp_wait<0>();
        __syncthreads();
        float value[S::kMmaTokens][S::kMmaRows][4]{};
        fp8_mma_compute_stage<S>(x + slot * Tokens * Group, w + slot * Rows * Group, value, warp,
                                 lane);
#pragma unroll
        for (int mt = 0; mt < S::kMmaTokens; ++mt) {
            const int t    = warp / S::kWarpsRows * S::kWarpTokens + mt * 16 + lane / 4;
            const float s0 = t < output.columns ? input_scales[t * Groups + group] : 0;
            const float s1 = t + 8 < output.columns ? input_scales[(t + 8) * Groups + group] : 0;
#pragma unroll
            for (int mr = 0; mr < S::kMmaRows; ++mr) {
                sum[mt][mr][0] = fmaf(value[mt][mr][0], s0, sum[mt][mr][0]);
                sum[mt][mr][1] = fmaf(value[mt][mr][1], s0, sum[mt][mr][1]);
                sum[mt][mr][2] = fmaf(value[mt][mr][2], s1, sum[mt][mr][2]);
                sum[mt][mr][3] = fmaf(value[mt][mr][3], s1, sum[mt][mr][3]);
            }
        }
        __syncthreads();
        if (group + 2 < Groups) stage(slot, group + 2);
    }
#pragma unroll
    for (int mt = 0; mt < S::kMmaTokens; ++mt) {
#pragma unroll
        for (int mr = 0; mr < S::kMmaRows; ++mr) {
            const int row     = warp % S::kWarpsRows * S::kWarpRows + mr * 8 + 2 * (lane % 4);
            const int parent  = row / (Rows / 4) * kWidth + row_begin / 4 + row % (Rows / 4);
            const auto scales = __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162*>(weight_scales + parent));
            sum[mt][mr][0] *= scales.x;
            sum[mt][mr][1] *= scales.y;
            sum[mt][mr][2] *= scales.x;
            sum[mt][mr][3] *= scales.y;
        }
    }
    __syncthreads();
    A8UpEpilogue{}.template finish_tile<S, false>(output, shared, sum, row_begin, 0, kTotal,
                                                  output.columns);
}
} // namespace ninfer::ops::detail::hc::mix::fp8
