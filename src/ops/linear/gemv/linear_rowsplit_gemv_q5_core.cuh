#pragma once

// Shared Q5 row-split decode (T=1) GEMV core.
//
// Memory-bound design for the narrow Qwen3.6 decode shapes (N ~= 5-7K):
//   - A block owns kRowsPerBlock output rows; one warp computes one full row.
//   - Each warp streams its row's 16-group weight tiles through shared memory with
//     a cp.async pipeline kStages buffers deep: tiles j+1 .. j+kStages-1 are in
//     flight while tile j is unpacked/accumulated, so DRAM latency is hidden
//     without relying on very high occupancy (these matrices cannot fill the GPU
//     with one warp per row otherwise).
//   - The activation x (full K vector, reused by every row) is optionally staged
//     into shared once per block (kStageX); disable it when K is so large that x
//     would not fit alongside the weight buffers (e.g. mlp_down, K=17408).
//   - Each warp does a warp-shuffle reduction and writes one bf16 output; there is
//     no block barrier on the hot path.
// A tile is 16 groups: 512 nibble bytes (32 uint4, one per lane), 128 high bytes
// (8 uint4), 32 scale bytes (2 uint4). All weight reads are fully coalesced 128-bit
// loads, so the kernel runs DRAM-bound instead of L1/LSU- or latency-bound.

#include "ops/common/math.h"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Issue the cp.async copies for one 16-group tile into a shared buffer slot, then
// commit them as one pipeline group. Every lane commits (even lanes that issue
// fewer copies) so a uniform pipe_wait works warp-wide.
__device__ __forceinline__ void
q5_issue_tile(uint4* __restrict__ s_nib, uint4* __restrict__ s_hi, uint4* __restrict__ s_sc,
              const std::uint8_t* __restrict__ code_row, const std::uint8_t* __restrict__ high_row,
              const std::uint8_t* __restrict__ scale_row, int tile, int lane) {
    constexpr int kGroupsPerTile = 16;
    const int g0                 = tile * kGroupsPerTile;
    pipe_copy<16>(&s_nib[lane], reinterpret_cast<const uint4*>(code_row + g0 * 32) + lane);
    if (lane < 8) {
        pipe_copy<16>(&s_hi[lane], reinterpret_cast<const uint4*>(high_row + g0 * 8) + lane);
    }
    if (lane < 2) {
        pipe_copy<16>(&s_sc[lane], reinterpret_cast<const uint4*>(scale_row + g0 * 2) + lane);
    }
    pipe_commit();
}

// Unpack + accumulate one staged 16-group tile. x is read from x2 (shared or global).
__device__ __forceinline__ float q5_consume_tile(const __nv_bfloat162* __restrict__ x2,
                                                 const uint4* __restrict__ s_nib,
                                                 const uint4* __restrict__ s_hi,
                                                 const uint4* __restrict__ s_sc, int tile, int lane,
                                                 float acc) {
    constexpr int kGroupK              = 64;
    constexpr int kGroupsPerTile       = 16;
    constexpr int kNibbleBytesPerGroup = 32;
    constexpr int kHighBytesPerGroup   = 8;
    const auto* tn                     = reinterpret_cast<const std::uint8_t*>(s_nib);
    const auto* th                     = reinterpret_cast<const std::uint8_t*>(s_hi);
    const auto* tsc                    = reinterpret_cast<const std::uint16_t*>(s_sc);
    const int g0                       = tile * kGroupsPerTile;
#pragma unroll
    for (int tg = 0; tg < kGroupsPerTile; ++tg) {
        const float scale       = __half2float(__ushort_as_half(tsc[tg]));
        const std::uint8_t low  = tn[tg * kNibbleBytesPerGroup + lane];
        const std::uint8_t high = th[tg * kHighBytesPerGroup + (lane >> 2)] >> ((lane & 3) * 2);
        const int q0 = sign_extend<5>(static_cast<int>((low & 0x0fu) | ((high & 0x01u) << 4)));
        const int q1 = sign_extend<5>(static_cast<int>((low >> 4) | ((high & 0x02u) << 3)));

        const int k0    = (g0 + tg) * kGroupK + lane * 2;
        const float2 xv = __bfloat1622float2(x2[k0 >> 1]);
        acc             = fmaf(static_cast<float>(q0) * scale, xv.x, acc);
        acc             = fmaf(static_cast<float>(q1) * scale, xv.y, acc);
    }
    return acc;
}

// kN  : output rows, kK : reduction dim (multiple of 1024).
// kRowsPerBlock : rows (= warps) per block.
// kStages : cp.async pipeline depth (shared buffers; >=2 to overlap).
// kStageX : stage the activation vector into shared (false when x is too large).
template <int kN, int kK, int kRowsPerBlock, int kStages, bool kStageX, bool kResidual>
__global__ void linear_rowsplit_gemv_q5_kernel(const __nv_bfloat16* __restrict__ x,
                                               const std::uint8_t* __restrict__ codes,
                                               const std::uint8_t* __restrict__ high_bits,
                                               const std::uint8_t* __restrict__ scales,
                                               const __nv_bfloat16* __restrict__ residual,
                                               __nv_bfloat16* __restrict__ out) {
    constexpr int kGroupK              = 64;
    constexpr int kGroups              = kK / kGroupK;
    constexpr int kGroupsPerTile       = 16;
    constexpr int kTiles               = kGroups / kGroupsPerTile;
    constexpr int kNibbleBytesPerGroup = 32;
    constexpr int kHighBytesPerGroup   = 8;
    constexpr int kXVecs               = kK / 8; // x as uint4 (8 bf16 each)
    constexpr int kPrefetch            = kStages - 1;
    static_assert(kGroups % kGroupsPerTile == 0, "K must be a multiple of 16 groups (1024)");
    static_assert(kN % kRowsPerBlock == 0, "N must be a multiple of kRowsPerBlock");
    static_assert(kStages >= 2, "need at least double buffering");

    // __align__(16) so the uint4 staging below is well-defined by construction.
    __shared__ __align__(16) __nv_bfloat16 x_sh[kStageX ? kK : 1];
    __shared__ uint4 s_nib[kRowsPerBlock][kStages][32];
    __shared__ uint4 s_hi[kRowsPerBlock][kStages][8];
    __shared__ uint4 s_sc[kRowsPerBlock][kStages][2];

    if constexpr (kStageX) {
        // x must be 16-byte aligned for the uint4 staging (guaranteed: activations
        // come from the 256-byte-aligned workspace arena).
        auto* x_sh_v    = reinterpret_cast<uint4*>(x_sh);
        const auto* x_g = reinterpret_cast<const uint4*>(x);
        for (int i = static_cast<int>(threadIdx.x); i < kXVecs; i += static_cast<int>(blockDim.x)) {
            x_sh_v[i] = x_g[i];
        }
        __syncthreads();
    }

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int row  = static_cast<int>(blockIdx.x) * kRowsPerBlock + warp;

    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(row) * kGroups * kNibbleBytesPerGroup;
    const std::uint8_t* high_row =
        high_bits + static_cast<std::int64_t>(row) * kGroups * kHighBytesPerGroup;
    const std::uint8_t* scale_row = scales + static_cast<std::int64_t>(row) * kGroups * 2;
    const auto* x2                = kStageX ? reinterpret_cast<const __nv_bfloat162*>(x_sh)
                                            : reinterpret_cast<const __nv_bfloat162*>(x);

    // Prime up to kPrefetch tiles; empty commits keep the group count uniform so the
    // constant pipe_wait<kPrefetch>() in the loop is always valid.
#pragma unroll
    for (int p = 0; p < kPrefetch; ++p) {
        if (p < kTiles) {
            q5_issue_tile(s_nib[warp][p], s_hi[warp][p], s_sc[warp][p], code_row, high_row,
                          scale_row, p, lane);
        } else {
            pipe_commit();
        }
    }

    float acc = 0.0f;
#pragma unroll 1
    for (int tile = 0; tile < kTiles; ++tile) {
        const int fetch = tile + kPrefetch;
        if (fetch < kTiles) {
            const int buf = fetch % kStages;
            q5_issue_tile(s_nib[warp][buf], s_hi[warp][buf], s_sc[warp][buf], code_row, high_row,
                          scale_row, fetch, lane);
        } else {
            pipe_commit();
        }
        pipe_wait<kPrefetch>();
        __syncwarp();

        const int buf = tile % kStages;
        acc = q5_consume_tile(x2, s_nib[warp][buf], s_hi[warp][buf], s_sc[warp][buf], tile, lane,
                              acc);
        __syncwarp();
    }

    acc = warp_reduce_sum(acc);
    if constexpr (kResidual) {
        if (lane == 0) {
            const __nv_bfloat16 y = __float2bfloat16_rn(acc);
            acc                   = __bfloat162float(residual[row]) + __bfloat162float(y);
        }
    }
    if (lane == 0) { out[row] = __float2bfloat16_rn(acc); }
}

template <int kN, int kK, int kRowsPerBlock, int kStages, bool kStageX>
__global__ void linear_rowsplit_gemv_q5_dual_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes0,
    const std::uint8_t* __restrict__ high_bits0, const std::uint8_t* __restrict__ scales0,
    const std::uint8_t* __restrict__ codes1, const std::uint8_t* __restrict__ high_bits1,
    const std::uint8_t* __restrict__ scales1, __nv_bfloat16* __restrict__ out0,
    __nv_bfloat16* __restrict__ out1) {
    constexpr int kGroupK              = 64;
    constexpr int kGroups              = kK / kGroupK;
    constexpr int kGroupsPerTile       = 16;
    constexpr int kTiles               = kGroups / kGroupsPerTile;
    constexpr int kNibbleBytesPerGroup = 32;
    constexpr int kHighBytesPerGroup   = 8;
    constexpr int kXVecs               = kK / 8;
    constexpr int kPrefetch            = kStages - 1;
    static_assert(kGroups % kGroupsPerTile == 0, "K must be a multiple of 16 groups (1024)");
    static_assert(kN % kRowsPerBlock == 0, "N must be a multiple of kRowsPerBlock");
    static_assert(kStages >= 2, "need at least double buffering");

    __shared__ __align__(16) __nv_bfloat16 x_sh[kStageX ? kK : 1];
    __shared__ uint4 s_nib[kRowsPerBlock][kStages][32];
    __shared__ uint4 s_hi[kRowsPerBlock][kStages][8];
    __shared__ uint4 s_sc[kRowsPerBlock][kStages][2];

    if constexpr (kStageX) {
        auto* x_sh_v    = reinterpret_cast<uint4*>(x_sh);
        const auto* x_g = reinterpret_cast<const uint4*>(x);
        for (int i = static_cast<int>(threadIdx.x); i < kXVecs; i += static_cast<int>(blockDim.x)) {
            x_sh_v[i] = x_g[i];
        }
        __syncthreads();
    }

    const int lane       = static_cast<int>(threadIdx.x) & 31;
    const int warp       = static_cast<int>(threadIdx.x) >> 5;
    const int global_row = static_cast<int>(blockIdx.x) * kRowsPerBlock + warp;
    const bool second    = global_row >= kN;
    const int row        = second ? global_row - kN : global_row;

    const std::uint8_t* codes     = second ? codes1 : codes0;
    const std::uint8_t* high_bits = second ? high_bits1 : high_bits0;
    const std::uint8_t* scales    = second ? scales1 : scales0;
    __nv_bfloat16* out            = second ? out1 : out0;

    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(row) * kGroups * kNibbleBytesPerGroup;
    const std::uint8_t* high_row =
        high_bits + static_cast<std::int64_t>(row) * kGroups * kHighBytesPerGroup;
    const std::uint8_t* scale_row = scales + static_cast<std::int64_t>(row) * kGroups * 2;
    const auto* x2                = kStageX ? reinterpret_cast<const __nv_bfloat162*>(x_sh)
                                            : reinterpret_cast<const __nv_bfloat162*>(x);

#pragma unroll
    for (int p = 0; p < kPrefetch; ++p) {
        if (p < kTiles) {
            q5_issue_tile(s_nib[warp][p], s_hi[warp][p], s_sc[warp][p], code_row, high_row,
                          scale_row, p, lane);
        } else {
            pipe_commit();
        }
    }

    float acc = 0.0f;
#pragma unroll 1
    for (int tile = 0; tile < kTiles; ++tile) {
        const int fetch = tile + kPrefetch;
        if (fetch < kTiles) {
            const int buf = fetch % kStages;
            q5_issue_tile(s_nib[warp][buf], s_hi[warp][buf], s_sc[warp][buf], code_row, high_row,
                          scale_row, fetch, lane);
        } else {
            pipe_commit();
        }
        pipe_wait<kPrefetch>();
        __syncwarp();

        const int buf = tile % kStages;
        acc = q5_consume_tile(x2, s_nib[warp][buf], s_hi[warp][buf], s_sc[warp][buf], tile, lane,
                              acc);
        __syncwarp();
    }

    acc = warp_reduce_sum(acc);
    if (lane == 0) { out[row] = __float2bfloat16(acc); }
}

// One block per kRowsPerBlock rows; kRowsPerBlock warps per block.
template <int kN, int kK, int kRowsPerBlock, int kStages = 2, bool kStageX = true>
inline void launch_q5_rowsplit_gemv(const __nv_bfloat16* x, const std::uint8_t* codes,
                                    const std::uint8_t* high_bits, const std::uint8_t* scales,
                                    __nv_bfloat16* out, cudaStream_t stream) {
    constexpr int kBlockThreads = kRowsPerBlock * 32;
    const int grid              = kN / kRowsPerBlock;
    linear_rowsplit_gemv_q5_kernel<kN, kK, kRowsPerBlock, kStages, kStageX, false>
        <<<grid, kBlockThreads, 0, stream>>>(x, codes, high_bits, scales, nullptr, out);
}

template <int kN, int kK, int kRowsPerBlock, int kStages = 2, bool kStageX = true>
inline void launch_q5_rowsplit_gemv_residual(const __nv_bfloat16* x, const std::uint8_t* codes,
                                             const std::uint8_t* high_bits,
                                             const std::uint8_t* scales,
                                             const __nv_bfloat16* residual, __nv_bfloat16* out,
                                             cudaStream_t stream) {
    constexpr int kBlockThreads = kRowsPerBlock * 32;
    const int grid              = kN / kRowsPerBlock;
    linear_rowsplit_gemv_q5_kernel<kN, kK, kRowsPerBlock, kStages, kStageX, true>
        <<<grid, kBlockThreads, 0, stream>>>(x, codes, high_bits, scales, residual, out);
}

template <int kN, int kK, int kRowsPerBlock, int kStages = 2, bool kStageX = true>
inline void launch_q5_rowsplit_gemv_dual(const __nv_bfloat16* x, const std::uint8_t* codes0,
                                         const std::uint8_t* high_bits0,
                                         const std::uint8_t* scales0, const std::uint8_t* codes1,
                                         const std::uint8_t* high_bits1,
                                         const std::uint8_t* scales1, __nv_bfloat16* out0,
                                         __nv_bfloat16* out1, cudaStream_t stream) {
    constexpr int kBlockThreads = kRowsPerBlock * 32;
    const int grid              = (2 * kN) / kRowsPerBlock;
    linear_rowsplit_gemv_q5_dual_kernel<kN, kK, kRowsPerBlock, kStages, kStageX>
        <<<grid, kBlockThreads, 0, stream>>>(x, codes0, high_bits0, scales0, codes1, high_bits1,
                                             scales1, out0, out1);
}

} // namespace ninfer::ops::detail
