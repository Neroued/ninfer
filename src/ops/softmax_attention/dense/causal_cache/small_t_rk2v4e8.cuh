#pragma once

// ninfer::ops - split-KV causal small-T attention, rk2v4-e8 KV-cache partial kernel.
//
//   * QK runs on native m16n8k32.s8 tensor cores exactly as the G64 INT8 kernel does. Q and
//     newly appended K receive the same fixed register-only D256 rotation; K is then E8-root
//     ("cylinder") encoded into a quarter-width plane. A 2-byte code pair expands back to eight
//     int8 codes sharing the per-64-group scale, so the int32 MMA output is still rescaled by
//     qs[row,g]*ks[key,g].
//   * V is Hadamard-rotated per 64-dim group before its packed int4 encode, so the persistent
//     value plane is half width. It is unpacked to full-width int8, dequantized once to a bf16
//     tile, and the existing bf16 PV MMA runs. PV therefore accumulates in rotated-V
//     coordinates; the reduce kernel below inverse-rotates before writing `out`.
//   * Partial numerators stay FP32 so the inverse rotation runs before any bf16 narrowing.
//   * All keys (history AND the current/diagonal tokens) are read from the compressed cache;
//     the fused append writes the new tokens first and a __syncthreads orders the readback.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include "ops/kv_cache/int8_g64_codec.cuh"
#include "ops/kv_cache/rk2v4e8_codec.cuh"
#include "ops/softmax_attention/dense/causal_cache/small_t.cuh"

#include <cstdint>

namespace ninfer::ops {

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
          bool DynamicArena, bool MultiBatch, bool Masked, typename CacheInput>
__launch_bounds__(WarpsPerCta * 32, MinBlocksPerSm) __global__
    void causal_attention_small_t_rk2v4e8_tiled_kernel(
        const __nv_bfloat16* q, CacheInput input, const std::int32_t* pos,
        std::uint8_t* cache_k, std::uint8_t* cache_v, __half* cache_k_scale,
        __half* cache_v_scale,
        const std::int32_t* block_tables, const std::int32_t* valid_columns,
        const std::int32_t* table_rows, std::int32_t table_stride, std::int32_t full_width,
        std::int32_t column_begin, std::int32_t logical_capacity, float scale,
        float* partial_acc, float* partial_m, float* partial_l) {
    constexpr int Wc                   = WarpsPerCta;
    constexpr int RowCount             = TokenTile * Geometry::GroupSize;
    constexpr int RowTiles             = (RowCount + 15) / 16;
    constexpr int Br                   = RowTiles * 16;
    constexpr int Bc                   = KeyBlock;
    constexpr int D                    = kCausalHeadDim;
    constexpr int DB16                 = D / 2;
    constexpr int Threads              = Wc * 32;
    constexpr int Groups               = kKVCacheInt8Groups;
    constexpr int GroupKc              = kKVCacheInt8Group / 32;
    constexpr int QKKs                 = D / 32;
    constexpr int QKNt                 = Bc / 8;
    constexpr int ConsumerWarpsPerTile = Wc / RowTiles;
    constexpr int PVNtPerWarp          = D / (ConsumerWarpsPerTile * 8);
    constexpr int PVKs                 = Bc / 16;
    // The 262144-key maximum envelope spans at most 49 pages in this split geometry.
    constexpr int PageIds         = 64;
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
    __shared__ std::int32_t physical_pages_s[PageIds];

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
    q += static_cast<std::int64_t>(kCausalHeadDim) * Geometry::QHeads * column_base;
    pos += column_base;
    if constexpr (CacheInput::writes_cache) {
        input.k += static_cast<std::int64_t>(kCausalHeadDim) * Geometry::KVHeads * column_base;
        input.v += static_cast<std::int64_t>(kCausalHeadDim) * Geometry::KVHeads * column_base;
    }
    const int table_row = table_rows == nullptr ? 0 : table_rows[batch];
    const std::int32_t* block_table =
        block_tables + static_cast<std::int64_t>(table_row) * table_stride;
    if constexpr (MultiBatch) {
        partial_acc += static_cast<std::int64_t>(batch) * kCausalHeadDim * Geometry::QHeads *
                       TokenTile * split_count;
        partial_m += static_cast<std::int64_t>(batch) * Geometry::QHeads * TokenTile * split_count;
        partial_l += static_cast<std::int64_t>(batch) * Geometry::QHeads * TokenTile * split_count;
    }

    auto write_neutral = [&]() {
        for (int row = tid; row < RowCount; row += Threads) {
            int q_head = 0;
            int token  = 0;
            causal_small_t_tc_row_to_qt<Geometry>(row, TokenTile, kv_head, q_head, token);
            if (causal_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_m[causal_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] =
                    -CUDART_INF_F;
                partial_l[causal_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] =
                    0.0f;
            }
        }
        for (int idx = tid; idx < RowCount * D; idx += Threads) {
            const int row = idx / D;
            const int d   = idx - row * D;
            int q_head    = 0;
            int token     = 0;
            causal_small_t_tc_row_to_qt<Geometry>(row, TokenTile, kv_head, q_head, token);
            if (causal_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_acc[causal_partial_acc_index<Geometry>(q_head, d, token, split,
                                                               TokenTile)] = 0.0f;
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
        causal_small_t_active_splits<Geometry, false>(window, split_count, TokenTile);
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
    const int first_page = first_tile >> kPagedKVPageShift;
    const int page_count = ((split_end - 1) >> kPagedKVPageShift) - first_page + 1;
    for (int page = tid; page < page_count; page += Threads) {
        physical_pages_s[page] = block_table[first_page + page];
    }
    __syncthreads();

    if constexpr (CacheInput::writes_cache) {
        // One warp owns the complete D256 K row and then the complete V row, which is the layout
        // the warp-collective E8 encoder needs (each 8-lane subgroup holds eight consecutive
        // rotated dimensions). Identical body to the standalone rk2v4-e8 append, so the fused
        // and standalone paths emit identical codes and both write every group scale once.
        for (int token = warp; token < valid_tokens; token += Wc) {
            const int position = pos[token];
            if (position < split_start || position >= split_end) { continue; }
            const int physical_page =
                physical_pages_s[(position >> kPagedKVPageShift) - first_page];
            rk2v4e8_append_row<Geometry>(input.k, input.v, cache_k, cache_v, cache_k_scale,
                                         cache_v_scale, token, kv_head, physical_page,
                                         position & kPagedKVPageMask, lane);
        }
        __syncthreads();
    }

    for (int i = tid; i < Br * D; i += Threads) { q_i8[i] = 0; }
    for (int i = tid; i < RowCount * Groups; i += Threads) { q_scale_tmp[i] = 0.0f; }
    __syncthreads();

    for (int row = warp; row < RowCount; row += Wc) {
        int q_head = 0;
        int token  = 0;
        causal_small_t_tc_row_to_qt<Geometry>(row, TokenTile, kv_head, q_head, token);
        float q_values[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int d = lane + 32 * r;
            q_values[r] = __bfloat162float(q[causal_q_index<Geometry>(q_head, d, token)]);
        }
        normalized_hadamard_d256_inplace(q_values, lane);

#pragma unroll
        for (int grp = 0; grp < Groups; ++grp) {
            const int d0    = grp * kKVCacheInt8Group + lane;
            const int d1    = d0 + 32;
            const float x0  = q_values[2 * grp];
            const float x1  = q_values[2 * grp + 1];
            float amax      = fmaxf(fabsf(x0), fabsf(x1));
            amax            = warp_max(amax, FullMask);
            const float qs  = amax > 0.0f ? amax / 127.0f : 0.0f;
            const float inv = qs > 0.0f ? 1.0f / qs : 0.0f;
            causal_small_t_store_byte_swizzled(q_i8, row, d0, DB16,
                                               kv_cache_int8_quant_code(x0, inv));
            causal_small_t_store_byte_swizzled(q_i8, row, d1, DB16,
                                               kv_cache_int8_quant_code(x1, inv));
            if (lane == 0) { q_scale_tmp[row * Groups + grp] = qs; }
        }
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
                const std::int64_t off = rk2v4e8_scale_index<Geometry>(
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
            const int key_l    = chunk / (D / 16);
            const int dc       = chunk - key_l * (D / 16);
            const int d        = dc * 16;
            const int key      = tile_k0 + key_l;
            std::int8_t* k_dst = &k_i8[key_l * D + causal_small_t_tc_swz(key_l, dc * 8) * 2];
            std::int8_t* v_dst = &v_i8[key_l * D + d];
            if (key >= split_start && key < split_end) {
                // Both planes are narrower than the int8 cache, so a 16-byte cp_async would
                // overrun them; expand synchronously into the full-width staging tiles instead.
                const int page_offset = key & kPagedKVPageMask;
                __align__(16) std::int8_t decoded[16];
                rk2v4e8_decode_k_16d(&cache_k[rk2v4e8_k_code_index<Geometry>(physical_page, kv_head,
                                                                             d, page_offset)],
                                     decoded);
                store_vec(k_dst, *reinterpret_cast<const int4*>(decoded));
                rk2v4e8_unpack_v_16d(&cache_v[rk2v4e8_v_code_index<Geometry>(physical_page, kv_head,
                                                                             d, page_offset)],
                                     decoded);
                store_vec(v_dst, *reinterpret_cast<const int4*>(decoded));
            } else {
                store_vec(k_dst, make_int4(0, 0, 0, 0));
                store_vec(v_dst, make_int4(0, 0, 0, 0));
            }
        }
        ninfer::ops::cp_commit();
    };

    int physical_page = physical_pages_s[0];
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
                        smem_addr(
                            &q_b16[(producer_row_base + a_rowoff) * DB16 +
                                   causal_small_t_tc_swz(producer_row_base + a_rowoff, acol)]));
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
                            smem_addr(&k_b16[brow * DB16 + causal_small_t_tc_swz(brow, bcol)]));
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
            causal_small_t_tc_row_to_qt<Geometry>(row0, TokenTile, kv_head, q_head0, token0);
            causal_small_t_tc_row_to_qt<Geometry>(row1, TokenTile, kv_head, q_head1, token1);
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
                p_sw[gid * Bc + causal_small_t_tc_swz32(gid, col0)] = __float2bfloat16(p00);
                p_sw[gid * Bc + causal_small_t_tc_swz32(gid, col1)] = __float2bfloat16(p01);
                p_sw[(gid + 8) * Bc + causal_small_t_tc_swz32(gid + 8, col0)] =
                    __float2bfloat16(p10);
                p_sw[(gid + 8) * Bc + causal_small_t_tc_swz32(gid + 8, col1)] =
                    __float2bfloat16(p11);
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
                __nv_bfloat16* dst = &v_bf16[key_l * D + causal_small_t_tc_swz(key_l, d)];
                if (key >= split_start && key < split_end) {
                    const int grp = d >> 6;
                    float vs      = 0.0f;
                    if ((lane & 7) == 0) { vs = __half2float(v_scale_s[key_l * Groups + grp]); }
                    vs = __shfl_sync(FullMask, vs, grp * 8);
                    store_vec(dst, kv_cache_int8_dequant_i8x8_from(&v_i8[key_l * D + d], vs));
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
                physical_page = physical_pages_s[(next_k0 >> kPagedKVPageShift) - first_page];
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
                    smem_addr(
                        &p_consumer[a_rowoff * Bc + causal_small_t_tc_swz32(a_rowoff, pcol)]));
                unsigned vf[2];
                const int vrow = k * 16 + b_koff + b_rin;
                const int vcol = global_n * 8;
                ldmatrix_x2_t(vf[0], vf[1],
                              smem_addr(&v_bf16[vrow * D + causal_small_t_tc_swz(vrow, vcol)]));
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
            causal_small_t_tc_row_to_qt<Geometry>(row0, TokenTile, kv_head, q_head, token);
            partial_m[causal_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] = m0;
            partial_l[causal_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] = l0;
        }
        if (row1 < RowCount) {
            int q_head = 0;
            int token  = 0;
            causal_small_t_tc_row_to_qt<Geometry>(row1, TokenTile, kv_head, q_head, token);
            partial_m[causal_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] = m1;
            partial_l[causal_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] = l1;
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
            causal_small_t_tc_row_to_qt<Geometry>(row0, TokenTile, kv_head, q_head, token);
            const std::int64_t dst =
                causal_partial_acc_index<Geometry>(q_head, d0, token, split, TokenTile);
            *reinterpret_cast<float2*>(&partial_acc[dst]) = make_float2(acc[n][0], acc[n][1]);
        }
        if (row1 < RowCount) {
            int q_head = 0;
            int token  = 0;
            causal_small_t_tc_row_to_qt<Geometry>(row1, TokenTile, kv_head, q_head, token);
            const std::int64_t dst =
                causal_partial_acc_index<Geometry>(q_head, d0, token, split, TokenTile);
            *reinterpret_cast<float2*>(&partial_acc[dst]) = make_float2(acc[n][2], acc[n][3]);
        }
    }
}

// Cross-split reduce for the rk2v4-e8 partial numerators. One 256-thread block owns one
// (q_head, column); thread `tid` owns dimension `tid`. The trailing warp then re-applies the
// per-64-group Hadamard, which is its own inverse, to bring the PV result out of the rotated-V
// basis before it narrows to bf16. Without this the output is garbage.
template <typename Geometry, bool MultiBatch, bool Masked, bool Offset>
__launch_bounds__(256) __global__ void causal_attention_small_t_rk2v4e8_reduce_output_kernel(
    const float* partial_acc, const float* partial_m, const float* partial_l,
    const std::int32_t* positions, const std::int32_t* valid_columns, std::int32_t tokens,
    std::int32_t full_width, std::int32_t column_begin, std::int32_t batch_size,
    std::int32_t split_count, __nv_bfloat16* out) {
    static_assert(kCausalHeadDim == 256);

    const int q_head      = static_cast<int>(blockIdx.x);
    const int flat_column = static_cast<int>(blockIdx.y);
    int batch             = 0;
    int token             = flat_column;
    if constexpr (MultiBatch) {
        batch = flat_column / tokens;
        token = flat_column - batch * tokens;
    }
    const int tid = static_cast<int>(threadIdx.x);
    if (q_head >= Geometry::QHeads || token >= tokens) { return; }
    if constexpr (MultiBatch) {
        if (batch >= batch_size) { return; }
    }
    if constexpr (Offset) { positions += column_begin; }
    if constexpr (MultiBatch) { positions += static_cast<std::int64_t>(batch) * full_width; }
    const int window  = positions[tokens - 1] + 1;
    int output_column = token;
    if constexpr (Offset) { output_column += column_begin; }
    if constexpr (MultiBatch) { output_column += batch * full_width; }

    if constexpr (MultiBatch) {
        partial_acc += static_cast<std::int64_t>(batch) * kCausalHeadDim * Geometry::QHeads *
                       tokens * split_count;
        partial_m += static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * split_count;
        partial_l += static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * split_count;
    }
    const int active_splits =
        causal_small_t_active_splits<Geometry, false>(window, split_count, tokens);

    __shared__ float reduce_or_weight[256];
    __shared__ float normalized[256];
    float local_m = -CUDART_INF_F;
    for (int split = tid; split < active_splits; split += 256) {
        local_m = fmaxf(
            local_m, partial_m[causal_partial_stat_index<Geometry>(q_head, token, split, tokens)]);
    }
    reduce_or_weight[tid] = local_m;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduce_or_weight[tid] = fmaxf(reduce_or_weight[tid], reduce_or_weight[tid + stride]);
        }
        __syncthreads();
    }
    const float head_m = reduce_or_weight[0];
    __syncthreads();

    float local_l = 0.0f;
    for (int split = tid; split < active_splits; split += 256) {
        const float tile_l =
            partial_l[causal_partial_stat_index<Geometry>(q_head, token, split, tokens)];
        if (tile_l > 0.0f && head_m > -CUDART_INF_F) {
            local_l +=
                tile_l *
                expf(partial_m[causal_partial_stat_index<Geometry>(q_head, token, split, tokens)] -
                     head_m);
        }
    }
    reduce_or_weight[tid] = local_l;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) { reduce_or_weight[tid] += reduce_or_weight[tid + stride]; }
        __syncthreads();
    }
    const float head_l = reduce_or_weight[0];
    __syncthreads();
    if (tid < active_splits) {
        const float tile_l =
            partial_l[causal_partial_stat_index<Geometry>(q_head, token, tid, tokens)];
        reduce_or_weight[tid] =
            tile_l > 0.0f && head_l > 0.0f
                ? expf(partial_m[causal_partial_stat_index<Geometry>(q_head, token, tid, tokens)] -
                       head_m)
                : 0.0f;
    }
    __syncthreads();
    bool valid = true;
    if constexpr (Masked) {
        int absolute_column = token;
        if constexpr (Offset) { absolute_column += column_begin; }
        valid = absolute_column < valid_columns[batch];
    }
    float numerator = 0.0f;
    for (int split = 0; split < active_splits; ++split) {
        numerator +=
            partial_acc[causal_partial_acc_index<Geometry>(q_head, tid, token, split, tokens)] *
            reduce_or_weight[split];
    }
    normalized[tid] = valid && head_l > 0.0f ? numerator / head_l : 0.0f;
    __syncthreads();

    if (tid >= 32) { return; }
    float values[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) { values[r] = normalized[tid + 32 * r]; }
    // values[2g] and values[2g+1] are dimensions (tid, tid + 32) of group g, exactly the
    // register layout the append-side rotation used.
#pragma unroll
    for (int g = 0; g < kRk2v4E8Groups; ++g) {
        rk2v4e8_hadamard64(values[2 * g], values[2 * g + 1], tid);
    }
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int d = tid + 32 * r;
        out[causal_q_index<Geometry>(q_head, d, output_column)] = __float2bfloat16(values[r]);
    }
}

} // namespace ninfer::ops
