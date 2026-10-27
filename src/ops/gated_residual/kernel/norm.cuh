#pragma once
#include "ops/gated_residual/kernel/combine.cuh"
#include "ops/common/cuda/reduce.cuh"
#include "ops/common/cuda/memory.cuh"

namespace ninfer::ops::detail::hc {
// A CTA owns every read and write of one (stream,column), including exact in-place updates.
template <int Width, Form Kind, int Block>
__launch_bounds__(Block) __global__ void norm_kernel(Operands o, std::int64_t first) {
    constexpr int kPairs  = Width / 2;
    constexpr int kValues = kPairs / Block;
    static_assert(kPairs % Block == 0);
    const std::int64_t row = first + blockIdx.x;
    const auto m           = row / kStreams;
    const int s            = int(row % kStreams);
    const auto base        = row * kPairs;
    const auto* input      = static_cast<const __nv_bfloat162*>(o.input);
    const auto* block      = static_cast<const __nv_bfloat162*>(o.block);
    const auto* weight     = static_cast<const __nv_bfloat162*>(o.weight) + s * kPairs;
    auto* residual         = static_cast<__nv_bfloat162*>(o.residual);
    auto* normalized       = static_cast<__nv_bfloat162*>(o.normalized);
    const float j          = Kind == Form::CombineNorm ? o.injection[row] : 1.0f;
    __nv_bfloat162 values[kValues];
    __nv_bfloat162 weights[kValues];
    float sum = 0;
#pragma unroll
    for (int v = 0; v < kValues; ++v) {
        const int d = threadIdx.x + v * Block;
        auto value  = input[(Kind == Form::ExpandNorm ? m * kPairs : base) + d];
        if constexpr (Kind == Form::CombineNorm || Kind == Form::AddNorm)
            value = update<Kind>(value, block[m * kPairs + d], j);
        values[v]    = value;
        weights[v]   = weight[d];
        const auto x = __bfloat1622float2(value);
        sum += x.x * x.x + x.y * x.y;
    }
    __shared__ float warp_sums[Block / common::kWarpSize];
    __shared__ float inverse;
    const float total = common::block_reduce_sum<Block>(sum, warp_sums);
    if (threadIdx.x == 0) inverse = rsqrtf(total * (1.0f / Width) + o.epsilon);
    __syncthreads();
#pragma unroll
    for (int v = 0; v < kValues; ++v) {
        const int d  = threadIdx.x + v * Block;
        const auto x = __bfloat1622float2(values[v]);
        const auto w = __bfloat1622float2(weights[v]);
        if constexpr (Kind != Form::GroupedNorm) residual[base + d] = values[v];
        normalized[base + d] =
            __floats2bfloat162_rn(x.x * inverse * (1.0f + w.x), x.y * inverse * (1.0f + w.y));
    }
}

// One CTA owns a complete stream/column. Vector slots stay in registers across
// reduction; tail slots contribute zero and never touch another row.
template <int Width, Form Kind, int Block, bool StreamingInput>
__launch_bounds__(Block) __global__ void norm_vector_kernel(Operands o, std::int64_t first) {
    constexpr int kVectors = Width / 8, kWarps = Block / common::kWarpSize;
    constexpr int kSlots = (kVectors + Block - 1) / Block;
    static_assert(Width % 8 == 0 && Block % common::kWarpSize == 0);
    const auto row = first + blockIdx.x, m = row / kStreams;
    const int s = int(row % kStreams);
    const auto* input =
        static_cast<const uint4*>(o.input) + (Kind == Form::ExpandNorm ? m : row) * kVectors;
    common::Bf16x8Pack values[kSlots];
    float sum = 0;
#pragma unroll
    for (int v = 0; v < kSlots; ++v) {
        const int d = threadIdx.x + v * Block;
        if (d < kVectors) {
            const uint4 bits = StreamingInput ? common::streaming_load(input + d) : input[d];
            auto value       = common::load_vec<common::Bf16x8Pack>(&bits);
            if constexpr (Kind == Form::CombineNorm || Kind == Form::AddNorm) {
                const auto block_bits = static_cast<const uint4*>(o.block)[m * kVectors + d];
                const auto block      = common::load_vec<common::Bf16x8Pack>(&block_bits);
                const float j         = Kind == Form::CombineNorm ? o.injection[row] : 1.F;
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    value.pair[i] = update<Kind>(value.pair[i], block.pair[i], j);
            }
            values[v] = value;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const auto x = __bfloat1622float2(value.pair[i]);
                sum += x.x * x.x + x.y * x.y;
            }
        }
    }
    __shared__ float sums[kWarps], inverse;
    sum = common::warp_reduce_sum(sum);
    if ((threadIdx.x & (common::kWarpSize - 1)) == 0) sums[threadIdx.x / common::kWarpSize] = sum;
    __syncthreads();
    if (threadIdx.x < common::kWarpSize) {
        sum = threadIdx.x < kWarps ? sums[threadIdx.x] : 0.F;
        sum = common::warp_reduce_sum(sum);
        if (threadIdx.x == 0) inverse = rsqrtf(sum * (1.F / Width) + o.epsilon);
    }
    __syncthreads();
#pragma unroll
    for (int v = 0; v < kSlots; ++v) {
        const int d = threadIdx.x + v * Block;
        if (d < kVectors) {
            const auto weight_bits = static_cast<const uint4*>(o.weight)[s * kVectors + d];
            const auto weight      = common::load_vec<common::Bf16x8Pack>(&weight_bits);
            common::Bf16x8Pack normalized;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const auto x = __bfloat1622float2(values[v].pair[i]);
                const auto w = __bfloat1622float2(weight.pair[i]);
                normalized.pair[i] =
                    __floats2bfloat162_rn(x.x * inverse * (1.F + w.x), x.y * inverse * (1.F + w.y));
            }
            if constexpr (Kind != Form::GroupedNorm)
                static_cast<uint4*>(o.residual)[row * kVectors + d] =
                    common::load_vec<uint4>(&values[v]);
            static_cast<uint4*>(o.normalized)[row * kVectors + d] =
                common::load_vec<uint4>(&normalized);
        }
    }
}
} // namespace ninfer::ops::detail::hc
