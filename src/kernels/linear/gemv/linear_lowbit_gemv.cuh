#pragma once

// qus::kernels::detail - tuned low-bit decode GEMV (T==1), tile-centric.
//
// The q5090 TILE_N64_K64 layout stores, per K64 group, 64 fp16 scales then 64
// rows x bpr code bytes, with rows contiguous. A warp-per-row mapping reads one
// row's groups scattered across tiles (kTileBytes apart) -> small uncoalesced
// transactions, pipe-bound well below the DRAM roofline.
//
// This kernel is tile-centric instead: each CTA owns a 32-row stripe of one N64
// tile; KSPLIT warps split the K groups; per group a warp cooperatively loads the
// contiguous 32-row code block (+ 32 scales) into shared memory with fully
// coalesced word loads, then lane == row decodes its own row from SMEM (branchless
// unpack) and accumulates the dot product with x (broadcast across the 32 rows).
// Partials are reduced across the KSPLIT warps in SMEM. Externally workspace-free.

#include "kernels/linear/codec/linear_codec.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace qus::kernels::detail {

// Decode one row's 64 codes from its packed words (SMEM or global, 4-byte aligned)
// and accumulate the dot product with x_group[0..valid). Branchless unpack.
template <class Codec>
__device__ __forceinline__ float accumulate_row(const std::uint32_t* w, float scale,
                                                const __nv_bfloat16* x_group, int valid, float acc);

template <>
__device__ __forceinline__ float accumulate_row<Q4Codec>(const std::uint32_t* w, float scale,
                                                         const __nv_bfloat16* x_group, int valid,
                                                         float acc) {
#pragma unroll
    for (int j = 0; j < Q4Codec::kGroupK; ++j) {
        if (j >= valid) { break; }
        const int byte           = j >> 1;
        const std::uint32_t bits = (w[byte >> 2] >> ((byte & 3) * 8)) & 0xffu;
        const std::uint32_t nib  = (j & 1) ? (bits >> 4) : (bits & 0x0fu);
        const int s              = static_cast<int>(nib << 28) >> 28;
        acc = fmaf(static_cast<float>(s) * scale, __bfloat162float(x_group[j]), acc);
    }
    return acc;
}

template <>
__device__ __forceinline__ float accumulate_row<Q5Codec>(const std::uint32_t* w, float scale,
                                                         const __nv_bfloat16* x_group, int valid,
                                                         float acc) {
#pragma unroll
    for (int j = 0; j < Q5Codec::kGroupK; ++j) {
        if (j >= valid) { break; }
        const int bitpos         = j * Q5Codec::kBits;
        const int wi             = bitpos >> 5;
        const int sh             = bitpos & 31;
        const std::uint32_t hi   = wi < 9 ? w[wi + 1] : 0u;       // 40 bytes = 10 words
        const std::uint32_t bits = __funnelshift_r(w[wi], hi, sh);
        const int s              = static_cast<int>(bits << 27) >> 27;  // 5-bit sign-extend
        acc = fmaf(static_cast<float>(s) * scale, __bfloat162float(x_group[j]), acc);
    }
    return acc;
}

template <>
__device__ __forceinline__ float accumulate_row<Q6Codec>(const std::uint32_t* w, float scale,
                                                         const __nv_bfloat16* x_group, int valid,
                                                         float acc) {
#pragma unroll
    for (int j = 0; j < Q6Codec::kGroupK; ++j) {
        if (j >= valid) { break; }
        const int bitpos         = j * Q6Codec::kBits;
        const int wi             = bitpos >> 5;
        const int sh             = bitpos & 31;
        const std::uint32_t hi   = wi < 11 ? w[wi + 1] : 0u;      // 48 bytes = 12 words
        const std::uint32_t bits = __funnelshift_r(w[wi], hi, sh);
        const int s              = static_cast<int>(bits << 26) >> 26;  // 6-bit sign-extend
        acc = fmaf(static_cast<float>(s) * scale, __bfloat162float(x_group[j]), acc);
    }
    return acc;
}

// ROWS rows per CTA (ROWS <= 32, lane == row; lanes >= ROWS only help the load).
// KSPLIT warps split the K groups for the same ROWS rows. blockDim.x == 32*KSPLIT.
// Smaller ROWS -> more CTAs -> higher occupancy for few-row (small-N) shapes.
// Dynamic SMEM (uint32-aligned): codes[KSPLIT][ROWS*wpr] scales[KSPLIT][ROWS] partials[KSPLIT][ROWS]
template <class Codec, int ROWS, int KSPLIT>
__global__ void linear_tile_lowbit_gemv_kernel(const __nv_bfloat16* x, const std::uint8_t* payload,
                                               __nv_bfloat16* out, std::int32_t n, std::int32_t k,
                                               std::int32_t padded_k) {
    constexpr int kWordsPerRow = Codec::kBytesPerRowPerGroup / 4;  // Q4=8 Q5=10 Q6=12
    static_assert(Codec::kBytesPerRowPerGroup % 4 == 0, "bpr must be 4-byte aligned");
    static_assert(Codec::kGroupK == 64, "group size 64");
    static_assert(ROWS <= 32 && 64 % ROWS == 0, "ROWS must divide 64 and fit a warp");

    const int lane      = threadIdx.x & 31;
    const int warp      = threadIdx.x >> 5;  // 0 .. KSPLIT-1
    const int base_row  = static_cast<int>(blockIdx.x) * ROWS;
    const int row       = base_row + lane;
    const int group_cnt = padded_k / Codec::kGroupK;
    const int tile      = base_row / 64;
    const int rit0      = base_row - tile * 64;  // multiple of ROWS, within [0,64)

    extern __shared__ std::uint32_t smem[];
    std::uint32_t* codes   = smem;                                              // KSPLIT*ROWS*wpr
    float* scales          = reinterpret_cast<float*>(codes + KSPLIT * ROWS * kWordsPerRow);
    float* partials        = scales + KSPLIT * ROWS;
    std::uint32_t* w_codes = codes + warp * (ROWS * kWordsPerRow);
    float* w_scales        = scales + warp * ROWS;

    float acc = 0.0f;
    for (int g = warp; g < group_cnt; g += KSPLIT) {
        const std::int64_t tile_off =
            (static_cast<std::int64_t>(tile) * group_cnt + g) * Codec::kTileBytes;
        const std::uint32_t* g_codes = reinterpret_cast<const std::uint32_t*>(
            payload + tile_off + 64 * 2 +
            static_cast<std::int64_t>(rit0) * Codec::kBytesPerRowPerGroup);
        const std::uint16_t* g_scales =
            reinterpret_cast<const std::uint16_t*>(payload + tile_off + rit0 * 2);

        // Coalesced cooperative load of the ROWS-row block (+ROWS scales) into SMEM.
#pragma unroll
        for (int idx = lane; idx < ROWS * kWordsPerRow; idx += 32) { w_codes[idx] = g_codes[idx]; }
        if (lane < ROWS) { w_scales[lane] = __half2float(__ushort_as_half(g_scales[lane])); }
        __syncwarp();

        const int base_k = g * Codec::kGroupK;
        if (lane < ROWS && base_k < k) {
            const int valid = (k - base_k) < Codec::kGroupK ? (k - base_k) : Codec::kGroupK;
            acc = accumulate_row<Codec>(w_codes + lane * kWordsPerRow, w_scales[lane], x + base_k,
                                        valid, acc);
        }
        __syncwarp();
    }

    if (lane < ROWS) { partials[warp * ROWS + lane] = acc; }
    __syncthreads();

    if (warp == 0 && lane < ROWS) {
        float sum = 0.0f;
#pragma unroll
        for (int wsplit = 0; wsplit < KSPLIT; ++wsplit) { sum += partials[wsplit * ROWS + lane]; }
        if (row < n) { out[row] = __float2bfloat16(sum); }
    }
}

template <class Codec, int ROWS, int KSPLIT>
constexpr std::size_t tile_lowbit_gemv_smem_bytes() {
    constexpr int kWordsPerRow = Codec::kBytesPerRowPerGroup / 4;
    return static_cast<std::size_t>(KSPLIT) * ROWS * kWordsPerRow * sizeof(std::uint32_t) +
           static_cast<std::size_t>(KSPLIT) * ROWS * sizeof(float) * 2;  // scales + partials
}

// ---------------------------------------------------------------------------
// Q4 keeps a dedicated direct kernel. Q4's byte-per-lane layout (one row's 32
// packed bytes = 32 contiguous bytes) is already coalesced without SMEM staging,
// so the tile-centric SMEM round-trip is pure overhead for it (measured: Q4
// mlp.gate/up regresses from ~86 ms to ~100 ms end-to-end under the tile kernel).
// Two warps per row, 4-group unrolled fast path with shuffled scale broadcast.
// ---------------------------------------------------------------------------

__device__ __forceinline__ std::uint32_t q4_load_scale_bits(const std::uint8_t* payload,
                                                            std::int64_t off, std::int32_t rit,
                                                            int lane) {
    std::uint32_t scale_bits = 0u;
    if (lane == 0) {
        scale_bits = static_cast<std::uint16_t>(payload[off + rit * 2]) |
                     static_cast<std::uint32_t>(
                         static_cast<std::uint16_t>(payload[off + rit * 2 + 1]) << 8);
    }
    return __shfl_sync(0xffffffffu, scale_bits, 0);
}

__device__ __forceinline__ std::uint32_t q4_load_scale_bits_x4_lane(
    const std::uint8_t* payload, std::int64_t off0, std::int32_t rit, int lane) {
    std::uint32_t lane_scale_bits = 0u;
    if (lane < 4) {
        const std::int64_t scale_off =
            off0 + static_cast<std::int64_t>(lane) * Q4Codec::kTileBytes +
            static_cast<std::int64_t>(rit) * 2;
        lane_scale_bits =
            static_cast<std::uint32_t>(*reinterpret_cast<const std::uint16_t*>(payload + scale_off));
    }
    return lane_scale_bits;
}

__device__ __forceinline__ float q4_accumulate_lane_pair(
    const std::uint8_t* payload, const __nv_bfloat16* x, std::int32_t tile, std::int32_t rit,
    std::int32_t group, std::int32_t group_cnt, std::int32_t base_k, std::int32_t limit, int lane,
    float acc) {
    const int offset = lane * 2;

    const std::int64_t off =
        (static_cast<std::int64_t>(tile) * group_cnt + group) * Q4Codec::kTileBytes;
    const std::uint32_t scale_bits = q4_load_scale_bits(payload, off, rit, lane);
    const float scale = __half2float(__ushort_as_half(static_cast<std::uint16_t>(scale_bits)));

    const std::uint8_t* packed =
        payload + off + 64 * 2 + static_cast<std::int64_t>(rit) * Q4Codec::kBytesPerRowPerGroup;
    const std::uint32_t bits = packed[lane];
    const int s0             = static_cast<int>((bits & 0x0fu) << 28) >> 28;
    const int s1             = static_cast<int>(((bits >> 4) & 0x0fu) << 28) >> 28;

    if (offset + 1 < limit) {
        const float2 xv =
            __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(x + base_k + offset));
        acc = fmaf(static_cast<float>(s0) * scale, xv.x, acc);
        acc = fmaf(static_cast<float>(s1) * scale, xv.y, acc);
    } else if (offset < limit) {
        acc = fmaf(static_cast<float>(s0) * scale, __bfloat162float(x[base_k + offset]), acc);
    }
    return acc;
}

__global__ void __launch_bounds__(128, 12)
    linear_tuned_q4_gemv_kernel(const __nv_bfloat16* x, const std::uint8_t* payload,
                                __nv_bfloat16* out, std::int32_t n, std::int32_t k,
                                std::int32_t padded_k) {
    constexpr int kWarpSize          = 32;
    constexpr int kWarpsPerRow       = 2;
    constexpr int kGroupsPerFastStep = 4;
    static_assert(Q4Codec::kGroupK == 64);

    const int lane            = threadIdx.x & (kWarpSize - 1);
    const int warp_in_block   = threadIdx.x / kWarpSize;
    const int warps_per_block = blockDim.x / kWarpSize;
    const int row_warp        = warp_in_block % kWarpsPerRow;
    const int row_in_block    = warp_in_block / kWarpsPerRow;
    const int rows_per_block  = warps_per_block / kWarpsPerRow;
    const std::int32_t row = static_cast<std::int32_t>(blockIdx.x * rows_per_block + row_in_block);
    const std::int32_t group_cnt        = padded_k / Q4Codec::kGroupK;
    const bool valid_row                = row < n;
    const std::int32_t active_group_cnt = valid_row ? group_cnt : 0;

    __shared__ float warp_partials[4];

    const std::int32_t tile = row / 64;
    const std::int32_t rit  = row - tile * 64;
    const int offset        = lane * 2;
    float acc               = 0.0f;

    std::int32_t chunk_base           = 0;
    const std::int32_t full_group_cnt = valid_row ? k / Q4Codec::kGroupK : 0;
    for (; chunk_base + kWarpsPerRow * kGroupsPerFastStep - 1 < full_group_cnt;
         chunk_base += kWarpsPerRow * kGroupsPerFastStep) {
        const std::int32_t group   = chunk_base + row_warp * kGroupsPerFastStep;
        const std::int32_t base_k0 = group * Q4Codec::kGroupK;
        const std::int32_t base_k1 = base_k0 + Q4Codec::kGroupK;
        const std::int32_t base_k2 = base_k1 + Q4Codec::kGroupK;
        const std::int32_t base_k3 = base_k2 + Q4Codec::kGroupK;
        const std::int64_t off0 =
            (static_cast<std::int64_t>(tile) * group_cnt + group) * Q4Codec::kTileBytes;
        const std::int64_t off1 = off0 + Q4Codec::kTileBytes;
        const std::int64_t off2 = off1 + Q4Codec::kTileBytes;
        const std::int64_t off3 = off2 + Q4Codec::kTileBytes;

        const std::uint32_t lane_scale_bits = q4_load_scale_bits_x4_lane(payload, off0, rit, lane);

        const std::uint8_t* packed0 =
            payload + off0 + 64 * 2 + static_cast<std::int64_t>(rit) * Q4Codec::kBytesPerRowPerGroup;
        const std::uint8_t* packed1 =
            payload + off1 + 64 * 2 + static_cast<std::int64_t>(rit) * Q4Codec::kBytesPerRowPerGroup;
        const std::uint8_t* packed2 =
            payload + off2 + 64 * 2 + static_cast<std::int64_t>(rit) * Q4Codec::kBytesPerRowPerGroup;
        const std::uint8_t* packed3 =
            payload + off3 + 64 * 2 + static_cast<std::int64_t>(rit) * Q4Codec::kBytesPerRowPerGroup;

        const std::uint32_t bits0 = packed0[lane];
        const std::uint32_t bits1 = packed1[lane];
        const std::uint32_t bits2 = packed2[lane];
        const std::uint32_t bits3 = packed3[lane];
        const float2 xv0 =
            __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(x + base_k0 + offset));
        const float2 xv1 =
            __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(x + base_k1 + offset));
        const float2 xv2 =
            __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(x + base_k2 + offset));
        const float2 xv3 =
            __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(x + base_k3 + offset));

        const int s00 = static_cast<int>((bits0 & 0x0fu) << 28) >> 28;
        const int s01 = static_cast<int>(((bits0 >> 4) & 0x0fu) << 28) >> 28;
        const int s10 = static_cast<int>((bits1 & 0x0fu) << 28) >> 28;
        const int s11 = static_cast<int>(((bits1 >> 4) & 0x0fu) << 28) >> 28;
        const int s20 = static_cast<int>((bits2 & 0x0fu) << 28) >> 28;
        const int s21 = static_cast<int>(((bits2 >> 4) & 0x0fu) << 28) >> 28;
        const int s30 = static_cast<int>((bits3 & 0x0fu) << 28) >> 28;
        const int s31 = static_cast<int>(((bits3 >> 4) & 0x0fu) << 28) >> 28;
        const std::uint32_t scale_bits0 = __shfl_sync(0xffffffffu, lane_scale_bits, 0);
        const std::uint32_t scale_bits1 = __shfl_sync(0xffffffffu, lane_scale_bits, 1);
        const std::uint32_t scale_bits2 = __shfl_sync(0xffffffffu, lane_scale_bits, 2);
        const std::uint32_t scale_bits3 = __shfl_sync(0xffffffffu, lane_scale_bits, 3);
        const float scale0 = __half2float(__ushort_as_half(static_cast<std::uint16_t>(scale_bits0)));
        const float scale1 = __half2float(__ushort_as_half(static_cast<std::uint16_t>(scale_bits1)));
        const float scale2 = __half2float(__ushort_as_half(static_cast<std::uint16_t>(scale_bits2)));
        const float scale3 = __half2float(__ushort_as_half(static_cast<std::uint16_t>(scale_bits3)));

        acc = fmaf(static_cast<float>(s00) * scale0, xv0.x, acc);
        acc = fmaf(static_cast<float>(s01) * scale0, xv0.y, acc);
        acc = fmaf(static_cast<float>(s10) * scale1, xv1.x, acc);
        acc = fmaf(static_cast<float>(s11) * scale1, xv1.y, acc);
        acc = fmaf(static_cast<float>(s20) * scale2, xv2.x, acc);
        acc = fmaf(static_cast<float>(s21) * scale2, xv2.y, acc);
        acc = fmaf(static_cast<float>(s30) * scale3, xv3.x, acc);
        acc = fmaf(static_cast<float>(s31) * scale3, xv3.y, acc);
    }

    for (std::int32_t group = chunk_base + row_warp; group < active_group_cnt;
         group += kWarpsPerRow) {
        const std::int32_t base_k = group * Q4Codec::kGroupK;
        if (base_k >= k) { continue; }
        const std::int32_t remaining = k - base_k;
        const std::int32_t limit = remaining < Q4Codec::kGroupK ? remaining : Q4Codec::kGroupK;
        acc = q4_accumulate_lane_pair(payload, x, tile, rit, group, group_cnt, base_k, limit, lane,
                                      acc);
    }

#pragma unroll
    for (int delta = kWarpSize / 2; delta > 0; delta >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, delta);
    }

    if (lane == 0) { warp_partials[warp_in_block] = acc; }
    __syncthreads();

    if (valid_row && row_warp == 0 && lane == 0) {
        out[row] = __float2bfloat16(warp_partials[warp_in_block] + warp_partials[warp_in_block + 1]);
    }
}

} // namespace qus::kernels::detail
