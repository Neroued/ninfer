#pragma once

#include "ops/linear/kernels/fp8_block/a16_tail_plan.h"
#include "ops/linear/kernels/fp8_block/a16_tma.cuh"

namespace ninfer::ops::detail {

template <class S, bool Full, class O, class E>
__global__ __launch_bounds__(S::kThreads, S::kMinBlocksPerSm) void fp8_block_a16_tail_kernel(
    const __grid_constant__ Fp8TmaDescriptors maps, Fp8BlockA16Operands<> p, O output, E epilogue,
    float* partials, int token_offset, int full_tiles) {
    using P        = typename S::TailPlan;
    const int tail = int(blockIdx.x) % P::kTailTiles, rank = int(blockIdx.x) / P::kTailTiles;
    const int groups = p.k / 128;
    const int first  = groups * rank / P::kParts;
    const int last   = groups * (rank + 1) / P::kParts;
    fp8_block_a16_tma_body<S, Full, true>(
        maps, p, output, epilogue, token_offset, full_tiles + tail, first * (128 / S::kBlockK),
        (last - first) * (128 / S::kBlockK), partials + int(blockIdx.x) * P::kTileElements);
}

// One CTA per accumulator fragment exposes enough parallelism for the small merge.
// No BF16 rounding occurs until all three FP32 K parts have been added.
template <class S, bool Full, class O, class E>
__global__ void fp8_block_a16_tail_reduce(const float* partials, O output, E epilogue, int rows,
                                          int tokens, int token_offset, int full_tiles) {
    using P                 = typename S::TailPlan;
    constexpr int fragments = S::kMmaRows * S::kMmaTokens;
    const int fragment = int(blockIdx.x) % fragments, tail = int(blockIdx.x) / fragments;
    const int tid = int(threadIdx.x), lane = tid & 31, warp = tid >> 5;
    const int wr = warp / S::kWarpsTokens, wt = warp % S::kWarpsTokens;
    const int r = fragment / S::kMmaTokens, t = fragment % S::kMmaTokens;
    float value[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int index = (fragment * 4 + j) * S::kConsumerThreads + tid;
        value[j]        = partials[tail * P::kTileElements + index];
#pragma unroll
        for (int part = 1; part < P::kParts; ++part)
            value[j] += partials[(part * P::kTailTiles + tail) * P::kTileElements + index];
    }
    int ri, ti;
    native_fp8_tile<S>(full_tiles + tail, rows, tokens, ri, ti);
    const int row = ri * S::kBlockRows + wr * S::kWarpRows + r * 16 + (lane >> 2);
    const int token =
        token_offset + ti * S::kBlockTokens + wt * S::kWarpTokens + t * 8 + 2 * (lane & 3);
    linear_finish_mma_fragment<Full, O, E, true>(
        output, epilogue, row, token, make_float4(value[0], value[1], value[2], value[3]), rows,
        token_offset + tokens);
}

template <class S, class O, class E>
void launch_fp8_block_a16_split_tail(const Fp8BlockA16Operands<>& p, O output, E epilogue,
                                     float* partials, cudaStream_t stream) {
    using P = typename S::TailPlan;
    static_assert(S::kBlockRows == P::kRows && S::kBlockTokens == P::kTokens);
    constexpr int bytes = fp8_block_a16_tma_bytes<S, E>;
    static_assert(bytes <= 99 * 1024);
    const auto maps = fp8_block_a16_tma_descriptors<S>(p);
    if (!partials || reinterpret_cast<std::uintptr_t>(partials) % P::kAlignment)
        throw std::invalid_argument("FP8 block split tail requires aligned FP32 partials");
    common::for_each_token_slice(p.tokens, S::kBlockTokens, [&](int offset, int count) {
        auto slice      = p;
        slice.tokens    = count;
        const auto plan = fp8_block_a16_tail_plan<P>(p.rows, count);
        if (plan.tiles > 2147483647LL)
            throw std::invalid_argument("FP8 block TMA grid exceeds CUDA grid.x");
        const auto run = [&]<bool Full>() {
            constexpr auto prefix = fp8_block_a16_tma_kernel<S, Full, O, E>;
            common::prepare_shared<bytes, prefix, true>();
            const auto prefix_tiles = plan.split_tail() ? plan.full_tiles() : plan.tiles;
            if (prefix_tiles) {
                prefix<<<unsigned(prefix_tiles), S::kThreads, bytes, stream>>>(maps, slice, output,
                                                                               epilogue, offset);
                CUDA_CHECK(cudaGetLastError());
            }
            if (!plan.split_tail()) return;
            constexpr auto tail = fp8_block_a16_tail_kernel<S, Full, O, E>;
            common::prepare_shared<bytes, tail, true>();
            tail<<<P::kCtas, S::kThreads, bytes, stream>>>(maps, slice, output, epilogue, partials,
                                                           offset, int(prefix_tiles));
            CUDA_CHECK(cudaGetLastError());
            fp8_block_a16_tail_reduce<S, Full>
                <<<P::kTailTiles * S::kMmaRows * S::kMmaTokens, S::kConsumerThreads, 0, stream>>>(
                    partials, output, epilogue, p.rows, count, offset, int(prefix_tiles));
            CUDA_CHECK(cudaGetLastError());
        };
        if (count % S::kBlockTokens == 0)
            run.template operator()<true>();
        else
            run.template operator()<false>();
    });
}

} // namespace ninfer::ops::detail
