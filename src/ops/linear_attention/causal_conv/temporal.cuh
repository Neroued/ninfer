#pragma once

#include "ops/linear_attention/kernels/causal_conv.cuh"

namespace ninfer::ops::detail {

// One thread owns a channel pair across a token tile. State publication requires a
// disjoint destination, or ownership of that channel pair's entire sequence.
template <int Channels, int TokensPerTile, int Block, bool Publish, class Output>
__global__ __launch_bounds__(Block) void causal_conv_temporal_kernel(
    const __nv_bfloat162* __restrict__ input, const __nv_bfloat162* __restrict__ weight,
    const __nv_bfloat162* state, __nv_bfloat162* final_state, Output output, int tokens) {
    constexpr int kPairs         = Channels / 2;
    constexpr int kChannelBlocks = (kPairs + Block - 1) / Block;
    const std::int64_t first     = std::int64_t(blockIdx.x / kChannelBlocks) * TokensPerTile;
    const int channel            = (blockIdx.x % kChannelBlocks) * Block + threadIdx.x;
    if (channel >= kPairs) return;

    const auto w0 = __bfloat1622float2(weight[channel]);
    const auto w1 = __bfloat1622float2(weight[kPairs + channel]);
    const auto w2 = __bfloat1622float2(weight[2 * kPairs + channel]);
    const auto w3 = __bfloat1622float2(weight[3 * kPairs + channel]);
    auto r0 = first >= 3 ? input[(first - 3) * kPairs + channel] : state[first * kPairs + channel];
    auto r1 =
        first >= 2 ? input[(first - 2) * kPairs + channel] : state[(first + 1) * kPairs + channel];
    auto r2 =
        first >= 1 ? input[(first - 1) * kPairs + channel] : state[(first + 2) * kPairs + channel];
    auto h0                = __bfloat1622float2(r0);
    auto h1                = __bfloat1622float2(r1);
    auto h2                = __bfloat1622float2(r2);
    const auto destination = output.column(channel);
#pragma unroll
    for (int column = 0; column < TokensPerTile; ++column) {
        const auto t = first + column;
        if (t < tokens) {
            const auto raw = input[t * kPairs + channel];
            const auto x   = __bfloat1622float2(raw);
            const float a  = fmaf(w3.x, x.x, fmaf(w2.x, h2.x, fmaf(w1.x, h1.x, w0.x * h0.x)));
            const float b  = fmaf(w3.y, x.y, fmaf(w2.y, h2.y, fmaf(w1.y, h1.y, w0.y * h0.y)));
            *destination.at(static_cast<std::int32_t>(t)) =
                __floats2bfloat162_rn(common::silu(a), common::silu(b));
            h0 = h1;
            h1 = h2;
            h2 = x;
            if constexpr (Publish) {
                r0 = r1;
                r1 = r2;
                r2 = raw;
            }
        }
    }
    if constexpr (Publish) {
        if (first + TokensPerTile >= tokens) {
            final_state[channel]              = r0;
            final_state[kPairs + channel]     = r1;
            final_state[2 * kPairs + channel] = r2;
        }
    }
}
} // namespace ninfer::ops::detail
