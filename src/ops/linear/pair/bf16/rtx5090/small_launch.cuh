#pragma once

#include "ops/linear/pair/bf16/launch.cuh"

namespace ninfer::ops::detail {

template <class S>
__global__ __launch_bounds__(S::kThreads, S::kMinBlocksPerSm) void bf16_pair_gemv_kernel(
    Bf16A16Operands a, Bf16A16Operands b, LinearBf16Output ya, LinearBf16Output yb) {
    static_assert(S::kActivationAccess == Bf16ActivationAccess::Direct && S::kWarpsPerRow == 1);
    const auto& p     = blockIdx.y ? b : a;
    const auto output = blockIdx.y ? yb : ya;
    const int lane    = threadIdx.x % common::kWarpSize;
    const int warp    = threadIdx.x / common::kWarpSize;
    const int row     = int(blockIdx.x) * S::kBlockRows + warp * S::kRowsPerWarp;
    float accum[S::kRowsPerWarp][S::kAccumulatorChains]{};
    compute_bf16_gemv_rows<S>(p.x, p.weight, row, 0, lane, accum, p.k);

#pragma unroll
    for (int r = 0; r < S::kRowsPerWarp; ++r) {
        float sum = 0.0F;
#pragma unroll
        for (int chain = 0; chain < S::kAccumulatorChains; ++chain) sum += accum[r][chain];
        sum = common::warp_reduce_sum(sum);
        if (lane == 0) output.store(row + r, 0, sum);
    }
}

template <class S>
void launch_bf16_pair_gemv(Bf16A16Operands a, Bf16A16Operands b, LinearBf16Output ya,
                           LinearBf16Output yb, cudaStream_t stream) {
    const dim3 grid(a.rows / S::kBlockRows, 2);
    bf16_pair_gemv_kernel<S><<<grid, S::kThreads, 0, stream>>>(a, b, ya, yb);
    CUDA_CHECK(cudaGetLastError());
}

template <class S>
__global__ __launch_bounds__(S::kThreads, S::kMinBlocksPerSm) void bf16_pair_simt_kernel(
    Bf16A16Operands a, Bf16A16Operands b, LinearBf16Output ya, LinearBf16Output yb, int offset) {
    const auto& p = blockIdx.y ? b : a;
    __shared__ Bf16SimtSharedStorage<S, S::kBlockTokens> shared;
    const int row_tiles = p.rows / S::kBlockRows;
    const int row_tile = int(blockIdx.x) % row_tiles, token_tile = int(blockIdx.x) / row_tiles;
    bf16_a16_simt_body<S>(p.x, p.weight, blockIdx.y ? yb : ya, LinearIdentityEpilogue{}, p.k,
                          p.tokens, row_tile, offset + token_tile * S::kBlockTokens, shared);
}

template <class S>
void launch_bf16_pair_simt(Bf16A16Operands a, Bf16A16Operands b, LinearBf16Output ya,
                           LinearBf16Output yb, cudaStream_t stream) {
    common::for_each_token_slice(a.tokens, S::kBlockTokens, [&](int offset, int count) {
        const dim3 grid(a.rows / S::kBlockRows * common::div_up(count, S::kBlockTokens), 2);
        bf16_pair_simt_kernel<S><<<grid, S::kThreads, 0, stream>>>(a, b, ya, yb, offset);
        CUDA_CHECK(cudaGetLastError());
    });
}

template <class S>
__global__ __launch_bounds__(S::kThreads, S::kMinBlocksPerSm) void bf16_pair_sliced_kernel(
    Bf16A16Operands a, Bf16A16Operands b, LinearBf16Output ya, LinearBf16Output yb, int offset,
    int count) {
    const auto& p = blockIdx.y ? b : a;
    extern __shared__ __align__(16) unsigned char shared[];
    int row_tile, token_tile;
    bf16_mma_tile_coordinates<S>(blockIdx.x, p.rows / S::kBlockRows,
                                 common::div_up(count, S::kBlockTokens), row_tile, token_tile);
    bf16_a16_sliced_k_mma_body<S>(p.x, p.weight, blockIdx.y ? yb : ya, LinearIdentityEpilogue{},
                                  p.rows, p.k, p.tokens, row_tile,
                                  offset + token_tile * S::kBlockTokens, shared);
}

template <class S>
void launch_bf16_pair_sliced(Bf16A16Operands a, Bf16A16Operands b, LinearBf16Output ya,
                             LinearBf16Output yb, cudaStream_t stream) {
    constexpr auto kernel = bf16_pair_sliced_kernel<S>;
    const int bytes       = common::prepare_shared<S::kSharedBytes, kernel, true>();
    common::for_each_token_slice(a.tokens, S::kBlockTokens, [&](int offset, int count) {
        const dim3 grid(a.rows / S::kBlockRows * common::div_up(count, S::kBlockTokens), 2);
        kernel<<<grid, S::kThreads, bytes, stream>>>(a, b, ya, yb, offset, count);
        CUDA_CHECK(cudaGetLastError());
    });
}

} // namespace ninfer::ops::detail
