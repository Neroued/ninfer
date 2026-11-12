#pragma once

#include "ops/gated_residual/mix/fp8/kernel/output.cuh"
#include "ops/common/cuda/pipeline.cuh"
#include "ops/common/cuda/reduce.cuh"
#include "ops/common/numeric/convert.cuh"
#include "ops/common/numeric/math.cuh"
#include "ops/common/quant/fp8.cuh"

namespace ninfer::ops::detail::hc::mix::fp8 {

template <int Columns>
struct VectorDownSchedule {
    static constexpr int kRows         = Columns == 2 ? 4 : 2;
    static constexpr int kValues       = Columns == 2 ? 4 : 8;
    static constexpr int kAccumulators = Columns == 2 ? 1 : 2;

    struct Chunk {
        unsigned input[Columns][kValues / 2];
        unsigned weight[kRows][kValues / 4];
    };
};

// Packed inputs survive only until the next K chunk. The injection CTA has a
// separate BF16 loop, so projection warps have no per-row representation branch.
template <int Columns>
__global__ __launch_bounds__(128, 2) void down_vector(const __nv_bfloat16* input,
                                                      const std::uint8_t* weights,
                                                      const __nv_bfloat16* injection,
                                                      float* partials, int columns) {
    using S            = VectorDownSchedule<Columns>;
    constexpr int Rows = S::kRows, Values = S::kValues, Accumulators = S::kAccumulators;
    constexpr int Step = 32 * Values, Chunks = kTotal / Step;
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, part = blockIdx.z;
    const int first = part * Chunks / kDownPartitions;
    const int last  = (part + 1) * Chunks / kDownPartitions;
    if (blockIdx.x == kRank / (4 * Rows)) {
        float sum[Columns]{};
        for (int chunk = first; chunk < last; ++chunk) {
            const int k = chunk * Step + lane * Values;
#pragma unroll
            for (int pair = 0; pair < Values / 2; ++pair) {
                const auto w = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(
                    injection + warp * kTotal + k + 2 * pair));
#pragma unroll
                for (int t = 0; t < Columns; ++t) {
                    if (t < columns) {
                        const auto x = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(
                            input + std::int64_t(t) * kTotal + k + 2 * pair));
                        sum[t]       = fmaf(w.x, x.x, sum[t]);
                        sum[t]       = fmaf(w.y, x.y, sum[t]);
                    }
                }
            }
        }
#pragma unroll
        for (int t = 0; t < Columns; ++t) {
            const float value = common::warp_sum(sum[t]);
            if (lane == 0 && t < columns)
                partials[(part * columns + t) * (kRank + kStreams) + kRank + warp] = value;
        }
        return;
    }

    const int row   = (blockIdx.x * 4 + warp) * Rows;
    const auto load = [&](int chunk) {
        typename S::Chunk values{};
        if (chunk < last) {
            const int k = chunk * Step + lane * Values;
#pragma unroll
            for (int t = 0; t < Columns; ++t) {
                if (t < columns) {
                    const auto* x = input + std::int64_t(t) * kTotal + k;
                    if constexpr (Values == 8)
                        common::store_vec(values.input[t], common::load_ldg<uint4>(x));
                    else
                        common::store_vec(values.input[t], common::load_ldg<uint2>(x));
                }
            }
#pragma unroll
            for (int r = 0; r < Rows; ++r) {
                const auto* w = weights + std::int64_t(row + r) * kTotal + k;
                if constexpr (Values == 8)
                    common::store_vec(values.weight[r], common::load_ldg<uint2>(w));
                else
                    values.weight[r][0] = common::load_ldg<unsigned>(w);
            }
        }
        return values;
    };

    auto next = load(first);
    float sum[Rows][Columns][Accumulators]{};
#pragma unroll 1
    for (int chunk = first; chunk < last; ++chunk) {
        const auto values = next;
        next              = load(chunk + 1);
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
#pragma unroll
            for (int pair = 0; pair < Values / 2; ++pair) {
                const unsigned codes   = values.weight[r][pair / 2] >> ((pair % 2) * 16);
                const unsigned decoded = common::fp8_e4m3x2_to_bf16x2_bits(codes);
                unsigned low, high;
                common::unpack_bf16x2_to_fp32_bits(decoded, low, high);
                const float wx = __uint_as_float(low), wy = __uint_as_float(high);
#pragma unroll
                for (int t = 0; t < Columns; ++t) {
                    common::unpack_bf16x2_to_fp32_bits(values.input[t][pair], low, high);
                    auto& accumulator = sum[r][t][pair % Accumulators];
                    accumulator       = fmaf(wx, __uint_as_float(low), accumulator);
                    accumulator       = fmaf(wy, __uint_as_float(high), accumulator);
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
#pragma unroll
        for (int t = 0; t < Columns; ++t) {
            float value = sum[r][t][0];
#pragma unroll
            for (int a = 1; a < Accumulators; ++a) value += sum[r][t][a];
            value = common::warp_sum(value);
            if (lane == 0 && t < columns)
                partials[(part * columns + t) * (kRank + kStreams) + row + r] = value;
        }
    }
}

// One warp owns a feature and all four streams. The stream dot products run
// concurrently; their final reduction stays in registers.
template <int Columns>
__global__ __launch_bounds__(128, 2) void up_vector(Fp8A16Operands weights, UpOutput output) {
    constexpr int Group = 8, Values = 8, Chunks = kRank / (Group * Values);
    constexpr int Accumulators = Columns == 2 ? 2 : 1;
    const int tid = threadIdx.x, lane = tid % Group, stream = tid / Group % kStreams;
    const int feature = blockIdx.x * 4 + tid / 32, parent = stream * kWidth + feature;
    __shared__ __align__(16) __nv_bfloat16 input[Columns * kRank];

    // Issue the epilogue reads before the dot-product pipeline.
    unsigned short row_scale, normalized[Columns]{};
    asm volatile("ld.global.nc.u16 %0, [%1];" : "=h"(row_scale) : "l"(weights.scales + parent));
#pragma unroll
    for (int t = 0; t < Columns; ++t) {
        if (t < output.columns)
            asm volatile("ld.global.nc.u16 %0, [%1];"
                         : "=h"(normalized[t])
                         : "l"(output.z + std::int64_t(t) * kTotal + parent));
    }
    const auto load = [&](int chunk) {
        return chunk < Chunks
                   ? common::load_ldg<uint2>(weights.codes + std::int64_t(parent) * kRank +
                                             chunk * Group * Values + lane * Values)
                   : make_uint2(0, 0);
    };
    uint2 current = load(0), next{};
    if constexpr (Columns == 2) next = load(1);
    for (int index = tid * 8; index < Columns * kRank; index += 128 * 8) {
        const bool valid = index / kRank < output.columns;
        common::cp_async_zfill<16, Cache::ca>(input + index, weights.x + (valid ? index : 0),
                                              valid ? 16 : 0);
    }
    common::cp_commit();
    common::cp_wait<0>();
    __syncthreads();
    float sum[Columns][Accumulators]{};
    const auto step = [&](int chunk) {
        const auto codes = current;
        if constexpr (Columns == 2) {
            current = next;
            next    = load(chunk + 2);
        } else {
            current = load(chunk + 1);
        }
        const unsigned words[2]{codes.x, codes.y};
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const unsigned packed = words[pair / 2] >> ((pair % 2) * 16);
            float2 w;
            if constexpr (Columns == 2) {
                unsigned low, high;
                common::unpack_bf16x2_to_fp32_bits(common::fp8_e4m3x2_to_bf16x2_bits(packed), low,
                                                   high);
                w = make_float2(__uint_as_float(low), __uint_as_float(high));
            } else {
                w = common::fp8_code2_to_float2(static_cast<std::uint16_t>(packed));
            }
#pragma unroll
            for (int t = 0; t < Columns; ++t) {
                const auto x      = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(
                    input + t * kRank + chunk * Group * Values + lane * Values + pair * 2));
                auto& accumulator = sum[t][pair % Accumulators];
                accumulator       = fmaf(w.x, x.x, accumulator);
                accumulator       = fmaf(w.y, x.y, accumulator);
            }
        }
    };
    if constexpr (Columns == 2) {
#pragma unroll
        for (int chunk = 0; chunk < Chunks; ++chunk) step(chunk);
    } else {
#pragma unroll 1
        for (int chunk = 0; chunk < Chunks; ++chunk) step(chunk);
    }
#pragma unroll
    for (int t = 0; t < Columns; ++t) {
        float value = sum[t][0];
#pragma unroll
        for (int a = 1; a < Accumulators; ++a) value += sum[t][a];
        value = common::warp_reduce_sum<Group>(value);
        const float gate =
            lane == 0 ? common::sigmoid(value * __uint_as_float(unsigned(row_scale) << 16)) *
                            __uint_as_float(unsigned(normalized[t]) << 16)
                      : 0;
        float result = gate;
        result += __shfl_sync(0xffffffff, gate, 8);
        result += __shfl_sync(0xffffffff, gate, 16);
        result += __shfl_sync(0xffffffff, gate, 24);
        if (stream == 0 && lane == 0 && t < output.columns)
            output.x[t * kWidth + feature] = __float2bfloat16_rn(result * .25F);
    }
}
} // namespace ninfer::ops::detail::hc::mix::fp8
