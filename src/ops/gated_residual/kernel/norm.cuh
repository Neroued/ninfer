#pragma once
#include "ops/gated_residual/kernel/combine.cuh"
#include "ops/common/cuda/reduce.cuh"
#include "ops/common/cuda/memory.cuh"

namespace ninfer::ops::detail::hc {
// A CTA owns every read and write of one (stream,column), including exact in-place updates.
template <Form Kind, int Block>
__launch_bounds__(Block) __global__ void norm_kernel(Operands o, std::int64_t first) {
    constexpr int kPairs  = kWidth / 2;
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
    if (threadIdx.x == 0) inverse = rsqrtf(total * (1.0f / kWidth) + o.epsilon);
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

// 320 threads cover a row with one 16-byte vector per thread. Streaming input loads
// retain less consumed input in cache; they are selected only in the measured domains.
template <Form Kind, bool StreamingInput>
__launch_bounds__(320) __global__ void norm_vector_kernel(Operands o, std::int64_t first) {
    constexpr int kVectors = kWidth / 8, kWarps = kVectors / common::kWarpSize;
    const auto row = first + blockIdx.x, m = row / kStreams;
    const int s = int(row % kStreams), d = threadIdx.x;
    const auto* input =
        static_cast<const uint4*>(o.input) + (Kind == Form::ExpandNorm ? m : row) * kVectors + d;
    const uint4 input_bits = StreamingInput ? common::streaming_load(input) : *input;
    auto value             = common::load_vec<common::Bf16x8Pack>(&input_bits);
    if constexpr (Kind == Form::CombineNorm || Kind == Form::AddNorm) {
        const auto block_bits = static_cast<const uint4*>(o.block)[m * kVectors + d];
        const auto block      = common::load_vec<common::Bf16x8Pack>(&block_bits);
        const float j         = Kind == Form::CombineNorm ? o.injection[row] : 1.F;
#pragma unroll
        for (int i = 0; i < 4; ++i) value.pair[i] = update<Kind>(value.pair[i], block.pair[i], j);
    }
    float sum = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const auto x = __bfloat1622float2(value.pair[i]);
        sum += x.x * x.x + x.y * x.y;
    }
    __shared__ float sums[kWarps], inverse;
    sum = common::warp_reduce_sum(sum);
    if ((threadIdx.x & (common::kWarpSize - 1)) == 0) sums[threadIdx.x / common::kWarpSize] = sum;
    __syncthreads();
    // Ten warp totals use a full warp reduction with zero-filled inactive entries.
    if (threadIdx.x < common::kWarpSize) {
        sum = threadIdx.x < kWarps ? sums[threadIdx.x] : 0.F;
        sum = common::warp_reduce_sum(sum);
        if (threadIdx.x == 0) inverse = rsqrtf(sum * (1.F / kWidth) + o.epsilon);
    }
    __syncthreads();
    const auto weight_bits = static_cast<const uint4*>(o.weight)[s * kVectors + d];
    const auto weight      = common::load_vec<common::Bf16x8Pack>(&weight_bits);
    common::Bf16x8Pack normalized;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const auto x = __bfloat1622float2(value.pair[i]);
        const auto w = __bfloat1622float2(weight.pair[i]);
        normalized.pair[i] =
            __floats2bfloat162_rn(x.x * inverse * (1.F + w.x), x.y * inverse * (1.F + w.y));
    }
    if constexpr (Kind != Form::GroupedNorm)
        static_cast<uint4*>(o.residual)[row * kVectors + d] = common::load_vec<uint4>(&value);
    static_cast<uint4*>(o.normalized)[row * kVectors + d] = common::load_vec<uint4>(&normalized);
}

} // namespace ninfer::ops::detail::hc
