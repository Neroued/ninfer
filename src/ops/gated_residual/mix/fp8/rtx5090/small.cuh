#pragma once

#include "ops/gated_residual/mix/fp8/kernel/output.cuh"
#include "ops/linear/kernels/fp8/fp8_a8_tma_mma.cuh"
#include "ops/common/cuda/pipeline.cuh"

namespace ninfer::ops::detail::hc::mix::fp8 {

inline constexpr int kSegmentThreads = 64, kSegmentK = kTotal / kDownPartitions;

template <int Columns>
struct SmallDownSchedule : Fp8A8TmaMmaSchedule<Columns, 32, 128, Columns / 16, 2, 2, 1> {
    static constexpr std::size_t kPartialBytes =
        std::size_t(kRank) * kDownPartitions * Columns * sizeof(float);
};

// Every CTA owns one fixed row/K partition and publishes raw FP32 partials.
// Partition scaling and nonlinearities belong to finish_small_down.
template <int Columns>
__global__ __launch_bounds__(SmallDownSchedule<Columns>::kThreads,
                             1) void down_small(const __grid_constant__ Fp8TmaDescriptors maps,
                                                float* partials) {
    using S            = SmallDownSchedule<Columns>;
    constexpr int Rows = S::kBlockRows, K = S::kBlockK, Stages = S::kStages;
    constexpr int Steps = kSegmentK / K;
    static_assert(kSegmentK % K == 0 && kRank % Rows == 0);
    extern __shared__ __align__(128) unsigned char storage[];
    auto* x           = storage;
    auto* w           = x + Stages * Columns * K;
    auto* full        = reinterpret_cast<std::uint64_t*>(storage + S::kStorageBytes);
    auto* empty       = full + Stages;
    const int row     = blockIdx.x / kDownPartitions * Rows;
    const int first_k = blockIdx.x % kDownPartitions * Steps;
    if (threadIdx.x == 0) {
#pragma unroll
        for (int stage = 0; stage < Stages; ++stage) {
            common::cta_mbarrier_init(full + stage, 1);
            common::cta_mbarrier_init(empty + stage, S::kConsumerWarps);
        }
        common::cta_mbarrier_fence_init();
    }
    __syncthreads();
    if (threadIdx.x < S::kProducerThreads) {
        if (threadIdx.x == 0) {
#pragma unroll
            for (int kt = 0; kt < Steps; ++kt) {
                const int stage = kt % Stages;
                common::cta_mbarrier_wait(empty + stage, 1U ^ ((kt / Stages) & 1U));
                common::cta_mbarrier_arrive_expect_tx(full + stage, (Columns + Rows) * K);
                common::tma_load_2d(x + stage * Columns * K, &maps.activation, (first_k + kt) * K,
                                    0, full + stage);
                common::tma_load_2d(w + stage * Rows * K, &maps.weight, (first_k + kt) * K, row,
                                    full + stage);
            }
        }
        return;
    }
    const int warp = (threadIdx.x - S::kProducerThreads) / 32, lane = threadIdx.x % 32;
    float sum[S::kMmaTokens][S::kMmaRows][4]{};
#pragma unroll
    for (int kt = 0; kt < Steps; ++kt) {
        const int stage = kt % Stages;
        common::cta_mbarrier_wait(full + stage, (kt / Stages) & 1U);
        fp8_mma_compute_stage<S>(x + stage * Columns * K, w + stage * Rows * K, sum, warp, lane);
        __syncwarp();
        if (lane == 0) common::cta_mbarrier_arrive(empty + stage);
    }
#pragma unroll
    for (int mt = 0; mt < S::kMmaTokens; ++mt) {
        const int token = warp / S::kWarpsRows * S::kWarpTokens + mt * 16 + lane / 4;
#pragma unroll
        for (int mr = 0; mr < S::kMmaRows; ++mr) {
            const int r = warp % S::kWarpsRows * S::kWarpRows + mr * 8 + 2 * (lane % 4);
            auto* out   = partials + blockIdx.x * Columns * Rows + token * Rows + r;
            *reinterpret_cast<float2*>(out) = make_float2(sum[mt][mr][0], sum[mt][mr][1]);
            *reinterpret_cast<float2*>(out + 8 * Rows) =
                make_float2(sum[mt][mr][2], sum[mt][mr][3]);
        }
    }
}

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
template <bool Segmented, bool EncodeUp, bool Injection, int PartialColumns = kSmallColumns>
__global__ __launch_bounds__(kUpGroup) void finish_small_down(
    const float* partial, const __nv_bfloat16* weights, const float* control, __nv_bfloat16* u,
    std::uint8_t* codes, float* scales, float* injection, int columns) {
    const int tid = threadIdx.x, token = blockIdx.y, group = blockIdx.x;
    const int row = group * kUpGroup + tid;
    float sum     = 0;
#pragma unroll
    for (int part = 0; part < kDownPartitions; ++part) {
        if constexpr (Segmented) {
            constexpr int Rows   = SmallDownSchedule<PartialColumns>::kBlockRows;
            constexpr int Tokens = PartialColumns;
            const float value    = partial[((row / Rows) * kDownPartitions + part) * Tokens * Rows +
                                        token * Rows + row % Rows];
            sum                  = fmaf(value, control[token * kDownPartitions + part], sum);
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
struct GroupedUpSchedule
    : Fp8A8MmaSchedule<Tokens, 32, kUpGroup, Tokens / 16, 64 / Tokens, 2, 2, Cache::cg, Cache::ca,
                       Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast> {
    static constexpr int kScaleOffset       = 2 * (Tokens + 32) * kUpGroup;
    static constexpr int kWeightScaleOffset = kScaleOffset + Tokens * (kRank / kUpGroup) * 4;
    static constexpr int kNormalizedOffset  = kWeightScaleOffset + 32 * 2;
    static constexpr int kSharedBytes       = kNormalizedOffset + Tokens * 32 * 2;
};

// Scales and normalized values enter shared memory with the first K group.
// Their consumers then stay off the global-memory dependency path.
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
    auto* w               = x + S::kStages * Tokens * Group;
    auto* scales          = reinterpret_cast<float*>(shared + S::kScaleOffset);
    auto* row_scales      = reinterpret_cast<__nv_bfloat16*>(shared + S::kWeightScaleOffset);
    auto* normalized      = reinterpret_cast<__nv_bfloat16*>(shared + S::kNormalizedOffset);
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int row_begin = blockIdx.x * Rows;
    const auto stage    = [&](int slot, int group) {
#pragma unroll
        for (int i = 0; i < Rows * (Group / 16) / Threads; ++i) {
            const int row    = tid / (Group / 16) + i * Threads / (Group / 16);
            const int k      = tid % (Group / 16) * 16;
            const int parent = row / (Rows / 4) * kWidth + row_begin / 4 + row % (Rows / 4);
            common::cp_async<16, Cache::cg>(w + (slot * Rows + row) * Group +
                                                fp8_mma_shared_byte<S>(row, k),
                                            weights + parent * kRank + group * Group + k);
        }
#pragma unroll
        for (int i = 0; i < Tokens * (Group / 16) / Threads; ++i) {
            const int row    = tid / (Group / 16) + i * Threads / (Group / 16);
            const int k      = tid % (Group / 16) * 16;
            const bool valid = row < output.columns;
            common::cp_async_zfill<16, Cache::ca>(
                x + (slot * Tokens + row) * Group + fp8_mma_shared_byte<S>(row, k),
                input + (valid ? row : 0) * kRank + group * Group + k, valid ? 16 : 0);
        }
        if (group == 0) {
            if (tid < Tokens * Groups / 4) {
                const int bytes = min(16, max(0, (output.columns * Groups - tid * 4) * 4));
                common::cp_async_zfill<16, Cache::ca>(scales + tid * 4,
                                                      input_scales + (bytes ? tid * 4 : 0), bytes);
            }
            if (tid < kStreams)
                common::cp_async<16, Cache::ca>(row_scales + tid * (Rows / kStreams),
                                                weight_scales + tid * kWidth +
                                                    row_begin / kStreams);
            if (tid < Tokens * kStreams) {
                const int token = tid / kStreams, stream = tid % kStreams;
                const bool valid = token < output.columns;
                common::cp_async_zfill<16, Cache::ca>(
                    normalized + A8UpEpilogue::at<Tokens>(stream * (Rows / kStreams), token),
                    output.z + std::int64_t(valid ? token : 0) * kTotal + stream * kWidth +
                        row_begin / kStreams,
                    valid ? 16 : 0);
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
            const int token = warp / S::kWarpsRows * S::kWarpTokens + mt * 16 + lane / 4;
            const float s0  = scales[token * Groups + group];
            const float s1  = scales[(token + 8) * Groups + group];
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
    const auto load = [&](int row, int token, int) {
        return A8UpEpilogue::GateInput{
            __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(
                normalized + A8UpEpilogue::at<Tokens>(row, token))),
            __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(row_scales + row))};
    };
    A8UpEpilogue::finish<S, true>(output, shared, sum, row_begin, 0, load);
}
} // namespace ninfer::ops::detail::hc::mix::fp8
