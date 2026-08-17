#pragma once

// ninfer::ops - split-KV GQA small-T attention, int8 KV-cache partial kernel.
// Historical design: docs/archive/optimization-era/2026-07-08-gqa-decode-int8-kernel-redesign.md.
//
//   * QK runs on native m16n8k32.s8 tensor cores. Q is quantized on-chip to int8
//     per (row, 64-group); K stays int8 in the cache and is read straight into
//     smem (no dequant). The int32 MMA output is rescaled per 64-group by
//     qs[row,g]*ks[key,g]. This halves the QK MMA count vs bf16 and removes the
//     entire K dequant.
//   * PV stays bf16 (V is quantized per key, so its scale cannot be factored out
//     of a key-contracted int8 accumulation): V int8 is staged, dequanted once to
//     a bf16 tile, then the existing bf16 PV MMA runs. V is still read from DRAM
//     as int8, so the bandwidth win is kept.
//   * All keys (history AND the current/diagonal tokens) are read from the
//     quantized cache; the fused append writes the new tokens first and a
//     __syncthreads orders the in-block readback. No from_new special-casing.
//
// Standalone from the bf16 kernel; shared scaffolding (layout constants, ldmatrix
// helpers, the s8/bf16 MMA helpers, the reducer) lives in gqa_attention_decode.cuh.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include "ops/kernel/e8_lattice.cuh"
#include "ops/kernel/e8_root_codec.cuh"
#include "ops/kernel/gqa_attention_decode.cuh"
#include "ops/kernel/gqa_attention_kv_quant.cuh"

#include <cstdint>

namespace ninfer::ops {

// Store one int8 code into a d-contiguous-as-b16 swizzled tile so the same
// gqa_small_t_tc_swz / ldmatrix path that serves bf16 tiles serves the int8 tile.
// A b16 lane holds two packed int8 (d even = low byte, d odd = high byte); this
// matches the byte layout a 16 B cp.async of d-contiguous cache bytes produces
// (see the design doc / kernel comments), so Q (byte stores) and K (cp.async)
// agree.
__device__ __forceinline__ void gqa_small_t_i8_store_swz(std::int8_t* tile, int row, int d,
                                                         int d_b16_stride, std::int8_t code) {
    const int c   = d >> 1;
    const int lo  = d & 1;
    const int off = (row * d_b16_stride + gqa_small_t_tc_swz(row, c)) * 2 + lo;
    tile[off]     = code;
}

// Decode-specialized producer/consumer kernel for T=1..6. One producer warp per
// m16 row tile computes QK + online softmax, while all CTA warps partition the
// tile's 256-wide PV output. This keeps each thread's PV accumulator at 16, 32,
// or 64 floats instead of 128 and uses otherwise-idle warps for useful output
// work.
//
// Q has a dedicated shared tile so producers can reload one 64-dimension group
// at a time. K/V codes and scales are staged asynchronously; non-producer warps
// dequantize V while producers execute QK. After both consume the code tile, the
// next K/V tile is prefetched into the same arena while the current PV runs.
template <typename Geometry, int TokenTile, int WarpsPerCta, int MinBlocksPerSm, int KeyBlock,
          bool DynamicArena, bool PackedV, bool RotateK, bool RotateV, bool PackedK,
          bool E8Lattice = false, bool E8Root = false, bool MultiBatch = false, bool Masked = false, typename CacheInput = void*>
__launch_bounds__(WarpsPerCta * 32, MinBlocksPerSm) __global__
    void gqa_attention_decode_i8_tiled_kernel(
        const __nv_bfloat16* q, CacheInput input, const std::int32_t* pos, std::int8_t* cache_k_i8,
        std::uint8_t* cache_v_codes, __half* cache_k_scale, __half* cache_v_scale,
        const std::int32_t* block_tables, const std::int32_t* valid_columns,
        const std::int32_t* table_rows, std::int32_t table_stride, std::int32_t full_width,
        std::int32_t column_begin, std::int32_t logical_capacity, float scale,
        __nv_bfloat16* partial_acc, float* partial_m, float* partial_l) {
    constexpr int Wc                   = WarpsPerCta;
    constexpr int RowCount             = TokenTile * Geometry::GroupSize;
    constexpr int RowTiles             = (RowCount + 15) / 16;
    constexpr int Br                   = RowTiles * 16;
    constexpr int Bc                   = KeyBlock;
    constexpr int D                    = kGqaHeadDim;
    constexpr int DB16                 = D / 2;
    constexpr int Threads              = Wc * 32;
    constexpr int Groups               = kGqaKvQuantGroups;
    constexpr int GroupKc              = kGqaKvQuantGroup / 32;
    constexpr int QKKs                 = D / 32;
    constexpr int QKNt                 = Bc / 8;
    constexpr int ConsumerWarpsPerTile = Wc / RowTiles;
    constexpr int PVNtPerWarp          = D / (ConsumerWarpsPerTile * 8);
    constexpr int PVKs                 = Bc / 16;
    constexpr int ProducerThreads = RowTiles * 32;
    constexpr int VLoaderThreads  = Threads - ProducerThreads;
    constexpr float Log2E         = 1.4426950408889634074f;
    constexpr unsigned FullMask   = 0xffffffffu;

    static_assert(TokenTile >= 1 && TokenTile <= 6);
    static_assert(Bc == 32 || Bc == 64);
    static_assert(RowTiles >= 1 && RowTiles <= 3);
    static_assert(Wc % RowTiles == 0);
    static_assert(PVNtPerWarp == 2 || PVNtPerWarp == 4 || PVNtPerWarp == 8 || PVNtPerWarp == 16);
    static_assert(QKKs == Groups * GroupKc);

    // Keep Q in a compact dedicated tile so the producer can reload one
    // 64-dimension group at a time instead of carrying all eight fragments in
    // registers across the whole kernel. The main arena holds K i8, V i8, and
    // V bf16 during the key loop.
    __shared__ __align__(16) std::int8_t q_s[Br * D];
    __shared__ __align__(16) std::int8_t static_r_s[DynamicArena ? 16 : 4 * Bc * D];
    extern __shared__ __align__(16) std::int8_t dynamic_r_s[];
    std::int8_t* r_s      = DynamicArena ? dynamic_r_s : static_r_s;
    std::int8_t* q_i8     = q_s;
    float* q_scale_tmp    = reinterpret_cast<float*>(r_s);
    std::int8_t* k_i8     = r_s;
    __nv_bfloat16* q_b16  = reinterpret_cast<__nv_bfloat16*>(q_i8);
    __nv_bfloat16* k_b16  = reinterpret_cast<__nv_bfloat16*>(k_i8);
    std::int8_t* v_i8     = r_s + Bc * D;
    __nv_bfloat16* v_bf16 = reinterpret_cast<__nv_bfloat16*>(r_s + 2 * Bc * D);
    __shared__ __align__(16) __nv_bfloat16 p_s[Br * Bc];
    __shared__ float alpha_s[Br];
    __shared__ __align__(16) __half k_scale_s[Bc * Groups];
    __shared__ __align__(16) __half v_scale_s[Bc * Groups];

    const int kv_head     = static_cast<int>(blockIdx.x);
    const int split       = static_cast<int>(blockIdx.y);
    const int batch       = MultiBatch ? static_cast<int>(blockIdx.z) : 0;
    const int split_count = static_cast<int>(gridDim.y);
    const int tid         = static_cast<int>(threadIdx.x);
    const int warp        = tid >> 5;
    const int lane        = tid & 31;

    int valid_tokens = TokenTile;
    if constexpr (Masked) {
        const int remaining = valid_columns[batch] - column_begin;
        valid_tokens        = remaining <= 0 ? 0 : (remaining < TokenTile ? remaining : TokenTile);
    }
    std::int64_t column_base = column_begin;
    if constexpr (MultiBatch) { column_base += static_cast<std::int64_t>(batch) * full_width; }
    q += static_cast<std::int64_t>(kGqaHeadDim) * Geometry::QHeads * column_base;
    pos += column_base;
    if constexpr (CacheInput::writes_cache) {
        input.k += static_cast<std::int64_t>(kGqaHeadDim) * Geometry::KVHeads * column_base;
        input.v += static_cast<std::int64_t>(kGqaHeadDim) * Geometry::KVHeads * column_base;
    }
    const int table_row = table_rows == nullptr ? 0 : table_rows[batch];
    const std::int32_t* block_table =
        block_tables + static_cast<std::int64_t>(table_row) * table_stride;
    if constexpr (MultiBatch) {
        partial_acc += static_cast<std::int64_t>(batch) * kGqaHeadDim * Geometry::QHeads *
                       TokenTile * split_count;
        partial_m += static_cast<std::int64_t>(batch) * Geometry::QHeads * TokenTile * split_count;
        partial_l += static_cast<std::int64_t>(batch) * Geometry::QHeads * TokenTile * split_count;
    }

    auto write_neutral = [&]() {
        for (int row = tid; row < RowCount; row += Threads) {
            int q_head = 0;
            int token  = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row, TokenTile, kv_head, q_head, token);
            if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] =
                    -CUDART_INF_F;
                partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] = 0.0f;
            }
        }
        for (int idx = tid; idx < RowCount * D; idx += Threads) {
            const int row = idx / D;
            const int d   = idx - row * D;
            int q_head    = 0;
            int token     = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row, TokenTile, kv_head, q_head, token);
            if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_acc[gqa_partial_acc_index<Geometry>(q_head, d, token, split, TokenTile)] =
                    __float2bfloat16(0.0f);
            }
        }
    };

    if (kv_head < 0 || kv_head >= Geometry::KVHeads || split_count <= 0) { return; }
    if (valid_tokens == 0) {
        write_neutral();
        return;
    }

    const std::int32_t first_pos = pos[0];
    const std::int32_t last_pos  = pos[TokenTile - 1];
    if (first_pos < 0 || last_pos < 0 || last_pos >= logical_capacity) {
        write_neutral();
        return;
    }

    const int window = last_pos + 1;
    const int active_split_count =
        gqa_small_t_active_splits<Geometry, true>(window, split_count, TokenTile);
    if (split >= active_split_count) { return; }

    const int logical_tiles = div_up(window, Bc);
    const bool tile_split   = logical_tiles >= active_split_count;
    const int units_per_split =
        tile_split ? div_up(logical_tiles, active_split_count) : div_up(window, active_split_count);
    const int split_start = split * units_per_split * (tile_split ? Bc : 1);
    const int split_limit = split_start + units_per_split * (tile_split ? Bc : 1);
    const int split_end   = (split_limit < window) ? split_limit : window;
    if (split_start >= split_end) {
        write_neutral();
        return;
    }
    const int first_tile = (split_start / Bc) * Bc;
    const int key_blocks = div_up(split_end - first_tile, Bc);

    if constexpr (CacheInput::writes_cache) {
        // The owning split quantizes each current row before its cache tile is consumed.
        for (int pair = warp; pair < valid_tokens * Groups; pair += Wc) {
            const int token    = pair / Groups;
            const int grp      = pair - token * Groups;
            const int position = pos[token];
            if (position < split_start || position >= split_end) { continue; }
            int physical_page       = lane == 0 ? paged_kv_physical_page(block_table, position) : 0;
            const int page_offset   = position & kPagedKVPageMask;
            const int d0            = grp * kGqaKvQuantGroup + lane;
            const int d1            = d0 + 32;
            const std::int64_t src0 = gqa_kv_new_index<Geometry>(kv_head, d0, token);
            const std::int64_t src1 = gqa_kv_new_index<Geometry>(kv_head, d1, token);
            float kv0               = __bfloat162float(input.k[src0]);
            float kv1               = __bfloat162float(input.k[src1]);
            float vv0               = __bfloat162float(input.v[src0]);
            float vv1               = __bfloat162float(input.v[src1]);
            if constexpr (RotateK) { gqa_kv_hadamard64(kv0, kv1, FullMask); }
            if constexpr (RotateV) { gqa_kv_hadamard64(vv0, vv1, FullMask); }
            float kamax             = fmaxf(fabsf(kv0), fabsf(kv1));
            float vamax             = fmaxf(fabsf(vv0), fabsf(vv1));
            kamax                   = warp_max(kamax, FullMask);
            vamax                   = warp_max(vamax, FullMask);
            const __half ksh = __float2half_rn(kamax > 0.0f ? kamax / ((PackedK || E8Root) ? 7.0f : 127.0f) : 0.0f);
            const __half vsh = __float2half_rn(vamax > 0.0f ? vamax / (PackedV ? 7.0f : 127.0f)
                                                                : 0.0f);
            const float ks          = __half2float(ksh);
            const float vs          = __half2float(vsh);
            const float k_inv       = ks > 0.0f ? 1.0f / ks : 0.0f;
            const float v_inv       = vs > 0.0f ? 1.0f / vs : 0.0f;
            physical_page           = __shfl_sync(FullMask, physical_page, 0);
            if constexpr (E8Root) {
                uint8_t c1_0, c2_0, c1_1, c2_1;
                e8_encode_cylinder_8d_warp(kv0, ks, c1_0, c2_0, lane);
                e8_encode_cylinder_8d_warp(kv1, ks, c1_1, c2_1, lane);
                if ((lane & 7) == 0) {
                    int s0 = (lane / 8);
                    int s1 = 4 + (lane / 8);
                    const std::int64_t k_base = paged_kv_page_head_offset<64, Geometry::KVHeads>(physical_page, kv_head) +
                                                static_cast<std::int64_t>(page_offset) * 64 + grp * 16;
                    reinterpret_cast<std::uint8_t*>(cache_k_i8)[k_base + s0 * 2 + 0] = c1_0;
                    reinterpret_cast<std::uint8_t*>(cache_k_i8)[k_base + s0 * 2 + 1] = c2_0;
                    reinterpret_cast<std::uint8_t*>(cache_k_i8)[k_base + s1 * 2 + 0] = c1_1;
                    reinterpret_cast<std::uint8_t*>(cache_k_i8)[k_base + s1 * 2 + 1] = c2_1;
                }
                const float vv0_hi = __shfl_down_sync(FullMask, vv0, 1);
                const float vv1_hi = __shfl_down_sync(FullMask, vv1, 1);
                if ((lane & 1) == 0) {
                    cache_v_codes[gqa_kv_i4_code_index<Geometry>(physical_page, kv_head, d0 / 2, page_offset)] =
                        gqa_kv_pack_i4(gqa_kv_quant_i4_code(vv0, v_inv), gqa_kv_quant_i4_code(vv0_hi, v_inv));
                    cache_v_codes[gqa_kv_i4_code_index<Geometry>(physical_page, kv_head, d1 / 2, page_offset)] =
                        gqa_kv_pack_i4(gqa_kv_quant_i4_code(vv1, v_inv), gqa_kv_quant_i4_code(vv1_hi, v_inv));
                }
            } else if constexpr (PackedK) {
                std::int8_t c0 = 0, c1 = 0;
                if constexpr (E8Lattice) {
                    float kv0_scaled = kv0 * k_inv;
                    float kv1_scaled = kv1 * k_inv;
                    e8_project_8d_warp(kv0_scaled, kv1_scaled, lane);
                    // NOTE (correctness hardening of the ported codec): e8_project_8d_warp
                    // returns the nearest E8 lattice point, which may lie in the D8+0.5
                    // coset (all coordinates half-integral). The i4/int8 codes can only hold
                    // integers and no coset bit is stored, so the half is intentionally
                    // collapsed here by the rintf()+cast below and the D8+0.5 coset is NOT
                    // reconstructed on decode (reconstructed K == code * ks). This is a
                    // deliberate half-coset approximation of rk4v4-e8, not an exact E8
                    // lattice projection, for any vector whose nearest lattice point is
                    // half-integral. Preserving the exact E8 point would require a per-8-dim
                    // coset bit in the packed layout (a format change, out of scope here).
                    // The rk4v4 (non-E8) path below is unaffected. Original design credit:
                    // UDPSendToFailed/ninfer-4090 (and Don-Chad/ninfer-3090 lineage).
                    int q0 = static_cast<int>(rintf(kv0_scaled));
                    int q1 = static_cast<int>(rintf(kv1_scaled));
                    c0 = static_cast<std::int8_t>(max(-8, min(7, q0)));
                    c1 = static_cast<std::int8_t>(max(-8, min(7, q1)));
                } else {
                    c0 = gqa_kv_quant_i4_code(kv0, k_inv);
                    c1 = gqa_kv_quant_i4_code(kv1, k_inv);
                }
                const std::int8_t c0_hi = static_cast<std::int8_t>(__shfl_down_sync(FullMask, static_cast<int>(c0), 1));
                const std::int8_t c1_hi = static_cast<std::int8_t>(__shfl_down_sync(FullMask, static_cast<int>(c1), 1));
                const float vv0_hi = __shfl_down_sync(FullMask, vv0, 1);
                const float vv1_hi = __shfl_down_sync(FullMask, vv1, 1);
                if ((lane & 1) == 0) {
                    reinterpret_cast<std::uint8_t*>(cache_k_i8)[gqa_kv_i4_code_index<Geometry>(physical_page, kv_head, d0 / 2, page_offset)] =
                        gqa_kv_pack_i4(c0, c0_hi);
                    reinterpret_cast<std::uint8_t*>(cache_k_i8)[gqa_kv_i4_code_index<Geometry>(physical_page, kv_head, d1 / 2, page_offset)] =
                        gqa_kv_pack_i4(c1, c1_hi);
                    cache_v_codes[gqa_kv_i4_code_index<Geometry>(physical_page, kv_head, d0 / 2, page_offset)] =
                        gqa_kv_pack_i4(gqa_kv_quant_i4_code(vv0, v_inv), gqa_kv_quant_i4_code(vv0_hi, v_inv));
                    cache_v_codes[gqa_kv_i4_code_index<Geometry>(physical_page, kv_head, d1 / 2, page_offset)] =
                        gqa_kv_pack_i4(gqa_kv_quant_i4_code(vv1, v_inv), gqa_kv_quant_i4_code(vv1_hi, v_inv));
                }
            } else {
                cache_k_i8[gqa_kv_quant_code_index<Geometry>(physical_page, kv_head, d0, page_offset)] =
                    gqa_kv_quant_code(kv0, k_inv);
                cache_k_i8[gqa_kv_quant_code_index<Geometry>(physical_page, kv_head, d1, page_offset)] =
                    gqa_kv_quant_code(kv1, k_inv);
                if constexpr (PackedV) {
                    const float vv0_hi = __shfl_down_sync(FullMask, vv0, 1);
                    const float vv1_hi = __shfl_down_sync(FullMask, vv1, 1);
                    if ((lane & 1) == 0) {
                        cache_v_codes[gqa_kv_i4_code_index<Geometry>(physical_page, kv_head, d0 / 2,
                                                                     page_offset)] =
                            gqa_kv_pack_i4(gqa_kv_quant_i4_code(vv0, v_inv),
                                           gqa_kv_quant_i4_code(vv0_hi, v_inv));
                        cache_v_codes[gqa_kv_i4_code_index<Geometry>(physical_page, kv_head, d1 / 2,
                                                                     page_offset)] =
                            gqa_kv_pack_i4(gqa_kv_quant_i4_code(vv1, v_inv),
                                           gqa_kv_quant_i4_code(vv1_hi, v_inv));
                    }
                } else {
                    auto* cache_v_i8 = reinterpret_cast<std::int8_t*>(cache_v_codes);
                    cache_v_i8[gqa_kv_quant_code_index<Geometry>(physical_page, kv_head, d0,
                                                                  page_offset)] =
                        gqa_kv_quant_code(vv0, v_inv);
                    cache_v_i8[gqa_kv_quant_code_index<Geometry>(physical_page, kv_head, d1,
                                                                  page_offset)] =
                        gqa_kv_quant_code(vv1, v_inv);
                }
            }
            if (lane == 0) {
                const std::int64_t so =
                    gqa_kv_quant_scale_index<Geometry>(physical_page, kv_head, grp, page_offset);
                cache_k_scale[so] = ksh;
                cache_v_scale[so] = vsh;
            }
        }
        __syncthreads();
    }

    for (int i = tid; i < Br * D; i += Threads) { q_i8[i] = 0; }
    for (int i = tid; i < RowCount * Groups; i += Threads) { q_scale_tmp[i] = 0.0f; }
    __syncthreads();

    for (int unit = warp; unit < RowCount * Groups; unit += Wc) {
        const int row = unit / Groups;
        const int grp = unit - row * Groups;
        const int d0  = grp * kGqaKvQuantGroup + lane;
        const int d1  = d0 + 32;
        int q_head    = 0;
        int token     = 0;
        gqa_small_t_tc_row_to_qt<Geometry>(row, TokenTile, kv_head, q_head, token);
        float x0        = __bfloat162float(q[gqa_q_index<Geometry>(q_head, d0, token)]);
        float x1        = __bfloat162float(q[gqa_q_index<Geometry>(q_head, d1, token)]);
        if constexpr (RotateK) {
            gqa_kv_hadamard64(x0, x1, FullMask);
        }
        float amax      = fmaxf(fabsf(x0), fabsf(x1));
        amax            = warp_max(amax, FullMask);
        const float qs  = amax > 0.0f ? amax / 127.0f : 0.0f;
        const float inv = qs > 0.0f ? 1.0f / qs : 0.0f;
        gqa_small_t_i8_store_swz(q_i8, row, d0, DB16, gqa_kv_quant_code(x0, inv));
        gqa_small_t_i8_store_swz(q_i8, row, d1, DB16, gqa_kv_quant_code(x1, inv));
        if (lane == 0) { q_scale_tmp[row * Groups + grp] = qs; }
    }
    __syncthreads();

    const int gid = lane >> 2;
    const int lid = lane & 3;

    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_rin    = lane & 7;
    const int b_koff   = ((lane >> 3) & 1) << 3;

    float q_scale_r0[Groups];
    float q_scale_r1[Groups];
    if (warp < RowTiles) {
        const int producer_row0 = warp * 16 + gid;
#pragma unroll
        for (int g = 0; g < Groups; ++g) {
            float qs0     = (lid == 0 && producer_row0 < RowCount)
                                ? q_scale_tmp[producer_row0 * Groups + g]
                                : 0.0f;
            float qs1     = (lid == 0 && producer_row0 + 8 < RowCount)
                                ? q_scale_tmp[(producer_row0 + 8) * Groups + g]
                                : 0.0f;
            q_scale_r0[g] = __shfl_sync(FullMask, qs0, gid * 4);
            q_scale_r1[g] = __shfl_sync(FullMask, qs1, gid * 4);
        }
    }
    __syncthreads();

    float acc[PVNtPerWarp][4];
#pragma unroll
    for (int n = 0; n < PVNtPerWarp; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { acc[n][i] = 0.0f; }
    }

    float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F;
    float l0 = 0.0f, l1 = 0.0f;

    auto issue_kv_tile = [&](int tile_k0, int physical_page) {
        for (int key_l = tid; key_l < Bc; key_l += Threads) {
            const int key = tile_k0 + key_l;
            if (key >= split_start && key < split_end) {
                const std::int64_t off = gqa_kv_quant_scale_index<Geometry>(
                    physical_page, kv_head, 0, key & kPagedKVPageMask);
                ninfer::ops::cp_async<8>(&k_scale_s[key_l * Groups], &cache_k_scale[off]);
                ninfer::ops::cp_async<8>(&v_scale_s[key_l * Groups], &cache_v_scale[off]);
            } else {
                store_vec(&k_scale_s[key_l * Groups], make_int2(0, 0));
                store_vec(&v_scale_s[key_l * Groups], make_int2(0, 0));
            }
        }
#pragma unroll 1
        for (int chunk = tid; chunk < Bc * (D / 16); chunk += Threads) {
            const int key_l = chunk / (D / 16);
            const int dc    = chunk - key_l * (D / 16);
            const int d     = dc * 16;
            const int key   = tile_k0 + key_l;
            if (key >= split_start && key < split_end) {
                if constexpr (E8Root) {
                    const std::int64_t koff = paged_kv_page_head_offset<64, Geometry::KVHeads>(
                        physical_page, kv_head) + static_cast<std::int64_t>(key & kPagedKVPageMask) * 64 + (d / 4);
                    const uint32_t src4 = *reinterpret_cast<const uint32_t*>(&reinterpret_cast<const std::uint8_t*>(cache_k_i8)[koff]);
                    const uint8_t c1_0 = static_cast<uint8_t>(src4 & 0xFF);
                    const uint8_t c2_0 = static_cast<uint8_t>((src4 >> 8) & 0xFF);
                    const uint8_t c1_1 = static_cast<uint8_t>((src4 >> 16) & 0xFF);
                    const uint8_t c2_1 = static_cast<uint8_t>((src4 >> 24) & 0xFF);
                    int8_t dec8_0[8], dec8_1[8];
                    e8_root_decode_8d_int8(c1_0, c2_0, dec8_0);
                    e8_root_decode_8d_int8(c1_1, c2_1, dec8_1);
                    std::int8_t* dst = &k_i8[key_l * D + gqa_small_t_tc_swz(key_l, dc * 8) * 2];
                    *reinterpret_cast<uint64_t*>(&dst[0]) = *reinterpret_cast<const uint64_t*>(dec8_0);
                    *reinterpret_cast<uint64_t*>(&dst[8]) = *reinterpret_cast<const uint64_t*>(dec8_1);
                    const std::int64_t voff = gqa_kv_i4_code_index<Geometry>(
                        physical_page, kv_head, d / 2, key & kPagedKVPageMask);
                    gqa_kv_unpack_i4x16(&cache_v_codes[voff], &v_i8[key_l * D + d]);
                } else if constexpr (PackedK) {
                    const std::int64_t koff = gqa_kv_i4_code_index<Geometry>(
                        physical_page, kv_head, d / 2, key & kPagedKVPageMask);
                    std::int8_t* dst = &k_i8[key_l * D + gqa_small_t_tc_swz(key_l, dc * 8) * 2];
                    gqa_kv_unpack_i4x16(&reinterpret_cast<const std::uint8_t*>(cache_k_i8)[koff], dst);
                    const std::int64_t voff = gqa_kv_i4_code_index<Geometry>(
                        physical_page, kv_head, d / 2, key & kPagedKVPageMask);
                    gqa_kv_unpack_i4x16(&cache_v_codes[voff], &v_i8[key_l * D + d]);
                } else {
                    const std::int64_t off = gqa_kv_quant_code_index<Geometry>(
                        physical_page, kv_head, d, key & kPagedKVPageMask);
                    std::int8_t* dst = &k_i8[key_l * D + gqa_small_t_tc_swz(key_l, dc * 8) * 2];
                    ninfer::ops::cp_async<16>(dst, &cache_k_i8[off]);
                    if constexpr (PackedV) {
                        const std::int64_t voff = gqa_kv_i4_code_index<Geometry>(
                            physical_page, kv_head, d / 2, key & kPagedKVPageMask);
                        gqa_kv_unpack_i4x16(&cache_v_codes[voff], &v_i8[key_l * D + d]);
                    } else {
                        ninfer::ops::cp_async<16>(&v_i8[key_l * D + d],
                                                  &reinterpret_cast<std::int8_t*>(cache_v_codes)[off]);
                    }
                }
            } else {
                std::int8_t* dst = &k_i8[key_l * D + gqa_small_t_tc_swz(key_l, dc * 8) * 2];
                store_vec(dst, make_int4(0, 0, 0, 0));
                store_vec(&v_i8[key_l * D + d], make_int4(0, 0, 0, 0));
            }
        }
        ninfer::ops::cp_commit();
    };

    int physical_page = block_table[first_tile >> kPagedKVPageShift];
    issue_kv_tile(first_tile, physical_page);
    ninfer::ops::cp_wait<0>();
    __syncthreads();

    for (int kb = 0; kb < key_blocks; ++kb) {
        const int k0 = first_tile + kb * Bc;

        // One warp per row tile produces P and alpha while the remaining warps
        // stream/dequant V.
        if (warp < RowTiles) {
            const int producer_row_base = warp * 16;
            __nv_bfloat16* p_sw         = &p_s[producer_row_base * Bc];
            float score[QKNt][4];
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                score[nt][0] = 0.0f;
                score[nt][1] = 0.0f;
                score[nt][2] = 0.0f;
                score[nt][3] = 0.0f;
            }

#pragma unroll
            for (int g = 0; g < Groups; ++g) {
                unsigned af[GroupKc][4];
#pragma unroll
                for (int kk = 0; kk < GroupKc; ++kk) {
                    const int k    = g * GroupKc + kk;
                    const int acol = k * 16 + a_coloff;
                    ldmatrix_x4(
                        af[kk][0], af[kk][1], af[kk][2], af[kk][3],
                        smem_addr(&q_b16[(producer_row_base + a_rowoff) * DB16 +
                                         gqa_small_t_tc_swz(producer_row_base + a_rowoff, acol)]));
                }

#pragma unroll
                for (int nt = 0; nt < QKNt; ++nt) {
                    int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
#pragma unroll
                    for (int kk = 0; kk < GroupKc; ++kk) {
                        const int k    = g * GroupKc + kk;
                        const int brow = nt * 8 + b_rin;
                        const int bcol = k * 16 + b_koff;
                        unsigned bf[2];
                        ldmatrix_x2(
                            bf[0], bf[1],
                            smem_addr(&k_b16[brow * DB16 + gqa_small_t_tc_swz(brow, bcol)]));
                        mma_s8(c0, c1, c2, c3, af[kk][0], af[kk][1], af[kk][2], af[kk][3], bf[0],
                               bf[1]);
                    }
                    const int keya = nt * 8 + 2 * lid;
                    const int keyb = keya + 1;
                    float ka       = 0.0f;
                    float kb2      = 0.0f;
                    if (gid == 0) {
                        ka  = __half2float(k_scale_s[keya * Groups + g]);
                        kb2 = __half2float(k_scale_s[keyb * Groups + g]);
                    }
                    ka  = __shfl_sync(FullMask, ka, lid);
                    kb2 = __shfl_sync(FullMask, kb2, lid);
                    score[nt][0] += q_scale_r0[g] * ka * static_cast<float>(c0);
                    score[nt][1] += q_scale_r0[g] * kb2 * static_cast<float>(c1);
                    score[nt][2] += q_scale_r1[g] * ka * static_cast<float>(c2);
                    score[nt][3] += q_scale_r1[g] * kb2 * static_cast<float>(c3);
                }
            }

            const int row0 = producer_row_base + gid;
            const int row1 = row0 + 8;
            int q_head0 = 0, token0 = 0, q_head1 = 0, token1 = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row0, TokenTile, kv_head, q_head0, token0);
            gqa_small_t_tc_row_to_qt<Geometry>(row1, TokenTile, kv_head, q_head1, token1);
            const int qabs0 = (row0 < RowCount) ? pos[token0] : -1;
            const int qabs1 = (row1 < RowCount) ? pos[token1] : -1;
            float bm0 = -CUDART_INF_F, bm1 = -CUDART_INF_F;
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int col0 = nt * 8 + 2 * lid;
                const int col1 = col0 + 1;
                const int key0 = k0 + col0;
                const int key1 = k0 + col1;
                score[nt][0] =
                    (row0 < RowCount && key0 >= split_start && key0 < split_end && key0 <= qabs0)
                        ? score[nt][0] * scale
                        : -CUDART_INF_F;
                score[nt][1] =
                    (row0 < RowCount && key1 >= split_start && key1 < split_end && key1 <= qabs0)
                        ? score[nt][1] * scale
                        : -CUDART_INF_F;
                score[nt][2] =
                    (row1 < RowCount && key0 >= split_start && key0 < split_end && key0 <= qabs1)
                        ? score[nt][2] * scale
                        : -CUDART_INF_F;
                score[nt][3] =
                    (row1 < RowCount && key1 >= split_start && key1 < split_end && key1 <= qabs1)
                        ? score[nt][3] * scale
                        : -CUDART_INF_F;
                bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
                bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
            }
            bm0 = warp_max<4>(bm0, FullMask);
            bm1 = warp_max<4>(bm1, FullMask);

            const float nm0    = fmaxf(m0, bm0);
            const float nm1    = fmaxf(m1, bm1);
            const float alpha0 = (m0 == -CUDART_INF_F) ? 0.0f : exp2_approx((m0 - nm0) * Log2E);
            const float alpha1 = (m1 == -CUDART_INF_F) ? 0.0f : exp2_approx((m1 - nm1) * Log2E);

            float bl0 = 0.0f, bl1 = 0.0f;
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int col0  = nt * 8 + 2 * lid;
                const int col1  = col0 + 1;
                const float p00 = (nm0 > -CUDART_INF_F && score[nt][0] > -CUDART_INF_F)
                                      ? exp2_approx((score[nt][0] - nm0) * Log2E)
                                      : 0.0f;
                const float p01 = (nm0 > -CUDART_INF_F && score[nt][1] > -CUDART_INF_F)
                                      ? exp2_approx((score[nt][1] - nm0) * Log2E)
                                      : 0.0f;
                const float p10 = (nm1 > -CUDART_INF_F && score[nt][2] > -CUDART_INF_F)
                                      ? exp2_approx((score[nt][2] - nm1) * Log2E)
                                      : 0.0f;
                const float p11 = (nm1 > -CUDART_INF_F && score[nt][3] > -CUDART_INF_F)
                                      ? exp2_approx((score[nt][3] - nm1) * Log2E)
                                      : 0.0f;
                bl0 += p00 + p01;
                bl1 += p10 + p11;
                p_sw[gid * Bc + gqa_small_t_tc_swz32(gid, col0)]           = __float2bfloat16(p00);
                p_sw[gid * Bc + gqa_small_t_tc_swz32(gid, col1)]           = __float2bfloat16(p01);
                p_sw[(gid + 8) * Bc + gqa_small_t_tc_swz32(gid + 8, col0)] = __float2bfloat16(p10);
                p_sw[(gid + 8) * Bc + gqa_small_t_tc_swz32(gid + 8, col1)] = __float2bfloat16(p11);
            }
            bl0 = warp_sum<4>(bl0, FullMask);
            bl1 = warp_sum<4>(bl1, FullMask);

            l0 = l0 * alpha0 + bl0;
            l1 = l1 * alpha1 + bl1;
            m0 = nm0;
            m1 = nm1;
            if (lid == 0) {
                alpha_s[row0] = alpha0;
                alpha_s[row1] = alpha1;
            }
        } else {
            const int loader_tid = tid - ProducerThreads;
#pragma unroll 1
            for (int chunk = loader_tid; chunk < Bc * (D / 8); chunk += VLoaderThreads) {
                const int key_l    = chunk / (D / 8);
                const int dc       = chunk - key_l * (D / 8);
                const int d        = dc * 8;
                const int key      = k0 + key_l;
                __nv_bfloat16* dst = &v_bf16[key_l * D + gqa_small_t_tc_swz(key_l, d)];
                if (key >= split_start && key < split_end) {
                    const int grp = d >> 6;
                    float vs      = 0.0f;
                    if ((lane & 7) == 0) { vs = __half2float(v_scale_s[key_l * Groups + grp]); }
                    vs = __shfl_sync(FullMask, vs, grp * 8);
                    store_vec(dst, gqa_kv_dequant_i8x8_from(&v_i8[key_l * D + d], vs));
                } else {
                    store_vec(dst, make_int4(0, 0, 0, 0));
                }
            }
        }
        __syncthreads();

        const bool has_next = kb + 1 < key_blocks;
        if (has_next) {
            const int next_k0 = k0 + Bc;
            if ((next_k0 & kPagedKVPageMask) == 0) {
                physical_page = block_table[next_k0 >> kPagedKVPageShift];
            }
            issue_kv_tile(next_k0, physical_page);
        }

        const int consumer_tile     = warp % RowTiles;
        const int consumer_slice    = warp / RowTiles;
        const int consumer_row_base = consumer_tile * 16;
        __nv_bfloat16* p_consumer   = &p_s[consumer_row_base * Bc];
        const float alpha0          = alpha_s[consumer_row_base + gid];
        const float alpha1          = alpha_s[consumer_row_base + gid + 8];
#pragma unroll
        for (int n = 0; n < PVNtPerWarp; ++n) {
            acc[n][0] *= alpha0;
            acc[n][1] *= alpha0;
            acc[n][2] *= alpha1;
            acc[n][3] *= alpha1;
        }

#pragma unroll
        for (int n = 0; n < PVNtPerWarp; ++n) {
            const int global_n = consumer_slice * PVNtPerWarp + n;
#pragma unroll
            for (int k = 0; k < PVKs; ++k) {
                unsigned pf[4];
                const int pcol = k * 16 + a_coloff;
                ldmatrix_x4(
                    pf[0], pf[1], pf[2], pf[3],
                    smem_addr(&p_consumer[a_rowoff * Bc + gqa_small_t_tc_swz32(a_rowoff, pcol)]));
                unsigned vf[2];
                const int vrow = k * 16 + b_koff + b_rin;
                const int vcol = global_n * 8;
                ldmatrix_x2_t(vf[0], vf[1],
                              smem_addr(&v_bf16[vrow * D + gqa_small_t_tc_swz(vrow, vcol)]));
                mma_bf16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], pf[0], pf[1], pf[2], pf[3],
                         vf[0], vf[1]);
            }
        }
        if (has_next) { ninfer::ops::cp_wait<0>(); }
        __syncthreads();
    }

    if (warp < RowTiles && lid == 0) {
        const int row0 = warp * 16 + gid;
        const int row1 = row0 + 8;
        if (row0 < RowCount) {
            int q_head = 0;
            int token  = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row0, TokenTile, kv_head, q_head, token);
            partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] = m0;
            partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] = l0;
        }
        if (row1 < RowCount) {
            int q_head = 0;
            int token  = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row1, TokenTile, kv_head, q_head, token);
            partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] = m1;
            partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] = l1;
        }
    }

#pragma unroll
    for (int n = 0; n < PVNtPerWarp; ++n) {
        const int consumer_tile     = warp % RowTiles;
        const int consumer_slice    = warp / RowTiles;
        const int consumer_row_base = consumer_tile * 16;
        const int d0                = (consumer_slice * PVNtPerWarp + n) * 8 + 2 * lid;
        const int row0              = consumer_row_base + gid;
        const int row1              = row0 + 8;
        if (row0 < RowCount) {
            int q_head = 0;
            int token  = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row0, TokenTile, kv_head, q_head, token);
            const std::int64_t dst =
                gqa_partial_acc_index<Geometry>(q_head, d0, token, split, TokenTile);
            *reinterpret_cast<unsigned*>(&partial_acc[dst]) = pack_bf16x2(acc[n][0], acc[n][1]);
        }
        if (row1 < RowCount) {
            int q_head = 0;
            int token  = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row1, TokenTile, kv_head, q_head, token);
            const std::int64_t dst =
                gqa_partial_acc_index<Geometry>(q_head, d0, token, split, TokenTile);
            *reinterpret_cast<unsigned*>(&partial_acc[dst]) = pack_bf16x2(acc[n][2], acc[n][3]);
        }
    }
}

} // namespace ninfer::ops
