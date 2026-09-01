#pragma once

// ninfer::ops - split-KV causal small-T attention, BF16 KV-cache partial kernel.
// Standalone from the int8 kernel (causal_attention_small_t_i8.cuh): shared scaffolding
// lives in causal_attention_small_t.cuh, but the body/append/load are not shared so the
// bf16 path can be tuned independently. Processes one KV head, one query-head
// subgroup, and one token tile; a reducer combines the split-local partials.

#include <cuda_bf16.h>
#include <math_constants.h>

#include "ops/softmax_attention/dense/causal_cache/small_t.cuh"

#include <cstdint>

namespace ninfer::ops {

template <typename Geometry, int TokenTile, int WarpsPerCta, bool MultiBatch, bool Masked,
          typename CacheInput>
__launch_bounds__(128, 2) __global__ void causal_attention_small_t_tc_partial_bf16_kernel(
    const __nv_bfloat16* q, CacheInput input, const std::int32_t* pos, __nv_bfloat16* cache_k,
    __nv_bfloat16* cache_v, const std::int32_t* block_tables, const std::int32_t* valid_columns,
    const std::int32_t* table_rows, std::int32_t table_stride, std::int32_t tokens,
    std::int32_t full_width, std::int32_t column_begin, std::int32_t logical_capacity, float scale,
    __nv_bfloat16* partial_acc, float* partial_m, float* partial_l) {
    static_assert(TokenTile >= 1 && TokenTile <= 6);
    static_assert(WarpsPerCta >= 1 && WarpsPerCta <= 4);

// The tensor-core path below needs ldmatrix (sm_75+) and mma.m16n8k16 (sm_80+), neither
// available on Volta. Below sm_80 this kernel takes a separate, fully independent SIMT
// body (see the #else branch) rather than forking individual call sites: the two paths
// share no tiling geometry (Volta has no equivalent of the m16n8k16 warp tile), so a
// shared prologue would couple control flow between two algorithms that have nothing in
// common past "read paged KV, do online softmax." See docs/v100.md.
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800

    constexpr int Wc      = WarpsPerCta;
    constexpr int Br      = Wc * 16;
    constexpr int Bc      = 32;
    constexpr int D       = kCausalHeadDim;
    constexpr int Threads = Wc * 32;
    constexpr int QKNt    = Bc / 8;
    constexpr int QKKs    = D / 16;
    constexpr int PVNt    = D / 8;
    constexpr int PVKs    = Bc / 16;
    // The 262144-key maximum envelope spans at most 49 pages in this split geometry.
    constexpr int PageIds       = 64;
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;
    constexpr int QkvRows       = 2 * Bc;

    static_assert(QkvRows >= Br);

    __shared__ __align__(16) __nv_bfloat16 qkv_s[QkvRows * D];
    __shared__ __align__(16) __nv_bfloat16 p_s[Wc * 16 * Bc];
    __shared__ std::int32_t physical_pages_s[PageIds];
    __nv_bfloat16* k_s = qkv_s;
    __nv_bfloat16* v_s = qkv_s + Bc * D;

    const int kv_head     = static_cast<int>(blockIdx.x);
    const int split       = static_cast<int>(blockIdx.y);
    const int batch       = MultiBatch ? static_cast<int>(blockIdx.z) : 0;
    const int split_count = static_cast<int>(gridDim.y);
    const int tid         = static_cast<int>(threadIdx.x);
    const int warp        = tid >> 5;
    const int lane        = tid & 31;
    int valid_tokens      = tokens;
    if constexpr (Masked) {
        const int remaining = valid_columns[batch] - column_begin;
        valid_tokens        = remaining <= 0 ? 0 : (remaining < tokens ? remaining : tokens);
    }
    const int row_count = tokens * Geometry::GroupSize;

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
                       tokens * split_count;
        partial_m += static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * split_count;
        partial_l += static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * split_count;
    }

    auto write_neutral = [&]() {
        for (int row = tid; row < row_count; row += Threads) {
            int q_head = 0;
            int token  = 0;
            causal_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
            if (causal_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_m[causal_partial_stat_index<Geometry>(q_head, token, split, tokens)] =
                    -CUDART_INF_F;
                partial_l[causal_partial_stat_index<Geometry>(q_head, token, split, tokens)] = 0.0f;
            }
        }
        for (int idx = tid; idx < row_count * D; idx += Threads) {
            const int row = idx / D;
            const int d   = idx - row * D;
            int q_head    = 0;
            int token     = 0;
            causal_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
            if (causal_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_acc[causal_partial_acc_index<Geometry>(q_head, d, token, split, tokens)] =
                    __float2bfloat16(0.0f);
            }
        }
    };

    if (kv_head < 0 || kv_head >= Geometry::KVHeads || tokens < 1 || tokens > TokenTile ||
        row_count > Br || split_count <= 0) {
        return;
    }
    if (valid_tokens == 0) {
        write_neutral();
        return;
    }

    const std::int32_t first_pos = pos[0];
    const std::int32_t last_pos  = pos[tokens - 1];
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

    if constexpr (CacheInput::writes_cache) {
        // The owning split writes each new row. Current attention reads those rows directly from
        // input below, so no split depends on another split's cache write.
        for (int chunk = tid; chunk < valid_tokens * (D / 8); chunk += Threads) {
            const int token = chunk / (D / 8);
            const int d     = (chunk - token * (D / 8)) * 8;
            const int p_tok = pos[token];
            if (p_tok >= split_start && p_tok < split_end && p_tok >= 0 &&
                p_tok < logical_capacity) {
                const std::int64_t new_off = kv_cache_int8_new_index<Geometry>(kv_head, d, token);
                const int lane             = tid & 31;
                int physical_page = lane == 0 ? paged_kv_physical_page(block_table, p_tok) : 0;
                physical_page     = __shfl_sync(FullMask, physical_page, 0);
                const std::int64_t cache_off = causal_cache_index<Geometry>(
                    physical_page, kv_head, d, p_tok & kPagedKVPageMask);
                store_vec(&cache_k[cache_off], load_vec<int4>(&input.k[new_off]));
                store_vec(&cache_v[cache_off], load_vec<int4>(&input.v[new_off]));
            }
        }
        __syncthreads();
    }

    for (int idx = tid; idx < Br * D; idx += Threads) {
        const int row = idx / D;
        const int d   = idx - row * D;
        int q_head    = 0;
        int token     = 0;
        causal_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
        __nv_bfloat16 value = __float2bfloat16(0.0f);
        if (row < row_count && causal_valid_q_head<Geometry>(kv_head, q_head)) {
            value = q[causal_q_index<Geometry>(q_head, d, token)];
        }
        qkv_s[row * D + causal_small_t_tc_swz(row, d)] = value;
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

    const int warp_row0 = warp * 16;
    __nv_bfloat16* p_sw = &p_s[warp * 16 * Bc];

    unsigned af_q[QKKs][4];
#pragma unroll
    for (int k = 0; k < QKKs; ++k) {
        const int arow = warp_row0 + a_rowoff;
        const int acol = k * 16 + a_coloff;
        ldmatrix_x4(af_q[k][0], af_q[k][1], af_q[k][2], af_q[k][3],
                    smem_addr(&qkv_s[arow * D + causal_small_t_tc_swz(arow, acol)]));
    }
    __syncthreads();
    int physical_page = physical_pages_s[0];
    float acc[PVNt][4];
#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { acc[n][i] = 0.0f; }
    }
    float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F, l0 = 0.0f, l1 = 0.0f;

    for (int kb = 0; kb < key_blocks; ++kb) {
        const int k0 = first_tile + kb * Bc;
        if (kb != 0 && (k0 & kPagedKVPageMask) == 0) {
            physical_page = physical_pages_s[(k0 >> kPagedKVPageShift) - first_page];
        }
        // Stage the bf16 K/V key tile with one cp.async wave (16B/thread, high MLP).
        // Current-step tokens come from k_new/v_new; tail slots are zeroed.
#pragma unroll 1
        for (int chunk = tid; chunk < Bc * (D / 8); chunk += Threads) {
            const int key_l      = chunk / (D / 8);
            const int d          = (chunk - key_l * (D / 8)) * 8;
            const int key        = k0 + key_l;
            __nv_bfloat16* k_dst = &k_s[key_l * D + causal_small_t_tc_swz(key_l, d)];
            __nv_bfloat16* v_dst = &v_s[key_l * D + causal_small_t_tc_swz(key_l, d)];
            if (key >= split_start && key < split_end) {
                if constexpr (CacheInput::writes_cache) {
                    const int new_token = key - first_pos;
                    const bool from_new =
                        new_token >= 0 && new_token < valid_tokens && key >= first_pos;
                    if (from_new) {
                        const std::int64_t off =
                            kv_cache_int8_new_index<Geometry>(kv_head, d, new_token);
                        ninfer::ops::cp_async<16>(k_dst, &input.k[off]);
                        ninfer::ops::cp_async<16>(v_dst, &input.v[off]);
                    } else {
                        const std::int64_t off = causal_cache_index<Geometry>(
                            physical_page, kv_head, d, key & kPagedKVPageMask);
                        ninfer::ops::cp_async<16>(k_dst, &cache_k[off]);
                        ninfer::ops::cp_async<16>(v_dst, &cache_v[off]);
                    }
                } else {
                    const std::int64_t off = causal_cache_index<Geometry>(physical_page, kv_head, d,
                                                                          key & kPagedKVPageMask);
                    ninfer::ops::cp_async<16>(k_dst, &cache_k[off]);
                    ninfer::ops::cp_async<16>(v_dst, &cache_v[off]);
                }
            } else {
                store_vec(k_dst, make_int4(0, 0, 0, 0));
                store_vec(v_dst, make_int4(0, 0, 0, 0));
            }
        }
        ninfer::ops::cp_commit();
        ninfer::ops::cp_wait<0>();
        __syncthreads();

        float score[QKNt][4];
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0f;
#pragma unroll
            for (int k = 0; k < QKKs; ++k) {
                unsigned bf[2];
                const int brow = nt * 8 + b_rin;
                const int bcol = k * 16 + b_koff;
                ldmatrix_x2(bf[0], bf[1],
                            smem_addr(&k_s[brow * D + causal_small_t_tc_swz(brow, bcol)]));
                mma_bf16(score[nt][0], score[nt][1], score[nt][2], score[nt][3], af_q[k][0],
                         af_q[k][1], af_q[k][2], af_q[k][3], bf[0], bf[1]);
            }
        }

        const int row0 = warp_row0 + gid;
        const int row1 = row0 + 8;
        int q_head0 = 0, token0 = 0, q_head1 = 0, token1 = 0;
        causal_small_t_tc_row_to_qt<Geometry>(row0, tokens, kv_head, q_head0, token0);
        causal_small_t_tc_row_to_qt<Geometry>(row1, tokens, kv_head, q_head1, token1);
        const int qabs0 = (row0 < row_count) ? pos[token0] : -1;
        const int qabs1 = (row1 < row_count) ? pos[token1] : -1;

        float bm0 = -CUDART_INF_F, bm1 = -CUDART_INF_F;
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            const int col0 = nt * 8 + 2 * lid;
            const int col1 = col0 + 1;
            const int key0 = k0 + col0;
            const int key1 = col1 + k0;
            score[nt][0] =
                (row0 < row_count && key0 >= split_start && key0 < split_end && key0 <= qabs0)
                    ? score[nt][0] * scale
                    : -CUDART_INF_F;
            score[nt][1] =
                (row0 < row_count && key1 >= split_start && key1 < split_end && key1 <= qabs0)
                    ? score[nt][1] * scale
                    : -CUDART_INF_F;
            score[nt][2] =
                (row1 < row_count && key0 >= split_start && key0 < split_end && key0 <= qabs1)
                    ? score[nt][2] * scale
                    : -CUDART_INF_F;
            score[nt][3] =
                (row1 < row_count && key1 >= split_start && key1 < split_end && key1 <= qabs1)
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
            p_sw[gid * Bc + causal_small_t_tc_swz32(gid, col0)]           = __float2bfloat16(p00);
            p_sw[gid * Bc + causal_small_t_tc_swz32(gid, col1)]           = __float2bfloat16(p01);
            p_sw[(gid + 8) * Bc + causal_small_t_tc_swz32(gid + 8, col0)] = __float2bfloat16(p10);
            p_sw[(gid + 8) * Bc + causal_small_t_tc_swz32(gid + 8, col1)] = __float2bfloat16(p11);
        }
        bl0 = warp_sum<4>(bl0, FullMask);
        bl1 = warp_sum<4>(bl1, FullMask);

        l0 = l0 * alpha0 + bl0;
        l1 = l1 * alpha1 + bl1;
        m0 = nm0;
        m1 = nm1;
#pragma unroll
        for (int n = 0; n < PVNt; ++n) {
            acc[n][0] *= alpha0;
            acc[n][1] *= alpha0;
            acc[n][2] *= alpha1;
            acc[n][3] *= alpha1;
        }
        __syncwarp();

#pragma unroll
        for (int n = 0; n < PVNt; ++n) {
#pragma unroll
            for (int k = 0; k < PVKs; ++k) {
                unsigned pf[4];
                const int pcol = k * 16 + a_coloff;
                ldmatrix_x4(
                    pf[0], pf[1], pf[2], pf[3],
                    smem_addr(&p_sw[a_rowoff * Bc + causal_small_t_tc_swz32(a_rowoff, pcol)]));
                unsigned vf[2];
                const int vrow = k * 16 + b_koff + b_rin;
                const int vcol = n * 8;
                ldmatrix_x2_t(vf[0], vf[1],
                              smem_addr(&v_s[vrow * D + causal_small_t_tc_swz(vrow, vcol)]));
                mma_bf16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], pf[0], pf[1], pf[2], pf[3],
                         vf[0], vf[1]);
            }
        }
        __syncthreads();
    }

    if (lid == 0) {
        const int row0 = warp_row0 + gid;
        const int row1 = row0 + 8;
        if (row0 < row_count) {
            int q_head = 0;
            int token  = 0;
            causal_small_t_tc_row_to_qt<Geometry>(row0, tokens, kv_head, q_head, token);
            partial_m[causal_partial_stat_index<Geometry>(q_head, token, split, tokens)] = m0;
            partial_l[causal_partial_stat_index<Geometry>(q_head, token, split, tokens)] = l0;
        }
        if (row1 < row_count) {
            int q_head = 0;
            int token  = 0;
            causal_small_t_tc_row_to_qt<Geometry>(row1, tokens, kv_head, q_head, token);
            partial_m[causal_partial_stat_index<Geometry>(q_head, token, split, tokens)] = m1;
            partial_l[causal_partial_stat_index<Geometry>(q_head, token, split, tokens)] = l1;
        }
    }

    // MMA fragments hold each row in four-lane groups. Stage the final split-local
    // accumulator through shared memory so partial_acc is written as contiguous d-vector stores.
#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
        const int d0   = n * 8 + 2 * lid;
        const int d1   = d0 + 1;
        const int row0 = warp_row0 + gid;
        const int row1 = row0 + 8;
        if (row0 < row_count) {
            qkv_s[row0 * D + d0] = __float2bfloat16(acc[n][0]);
            qkv_s[row0 * D + d1] = __float2bfloat16(acc[n][1]);
        }
        if (row1 < row_count) {
            qkv_s[row1 * D + d0] = __float2bfloat16(acc[n][2]);
            qkv_s[row1 * D + d1] = __float2bfloat16(acc[n][3]);
        }
    }
    __syncthreads();

    for (int chunk = tid; chunk < row_count * (D / 8); chunk += Threads) {
        const int row = chunk / (D / 8);
        const int d   = (chunk - row * (D / 8)) * 8;
        int q_head    = 0;
        int token     = 0;
        causal_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
        if (causal_valid_q_head<Geometry>(kv_head, q_head)) {
            const std::int64_t dst =
                causal_partial_acc_index<Geometry>(q_head, d, token, split, tokens);
            store_vec(&partial_acc[dst], load_vec<int4>(&qkv_s[row * D + d]));
        }
    }

#else // __CUDA_ARCH__ < 800: Volta SIMT decode, no tensor cores.
    //
    // Same paged-KV addressing, GQA row mapping, and online-softmax recurrence as the
    // tensor-core path (this is a separate copy of that prologue, not a shared one — see
    // the comment above the #if for why). The tile-batched (Bc=32 keys/iteration) scoring
    // is replaced with one key at a time, so the split partitioning below is simpler
    // (plain contiguous ranges, not Bc-aligned) — both are valid, non-overlapping covers
    // of [0, window), which is all the reduce kernel requires. Each warp handles up to 16
    // rows sequentially (register cost: one row's state live at a time, not all 16), and
    // within a row all 32 lanes cooperate on one key: each lane owns an 8-wide d-chunk
    // (D=256=32*8) of Q/K/V, contributes a partial dot product, and warp_sum (an
    // all-reduce via shfl_xor) gives every lane the same score. No shared-memory K/V
    // staging (correctness-first Phase 1 milestone; each row re-reads K/V from
    // global/cache directly) — see docs/v100.md for the follow-up to add staging
    // shared across a warp's rows, matching how the tensor-core path reuses one key tile.
    constexpr int Wc            = WarpsPerCta;
    constexpr int D             = kCausalHeadDim;
    constexpr int Threads       = Wc * 32;
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;
    constexpr int kDPerLane     = D / 32;

    const int kv_head     = static_cast<int>(blockIdx.x);
    const int split       = static_cast<int>(blockIdx.y);
    const int batch       = MultiBatch ? static_cast<int>(blockIdx.z) : 0;
    const int split_count = static_cast<int>(gridDim.y);
    const int tid         = static_cast<int>(threadIdx.x);
    const int warp        = tid >> 5;
    const int lane        = tid & 31;
    int valid_tokens       = tokens;
    if constexpr (Masked) {
        const int remaining = valid_columns[batch] - column_begin;
        valid_tokens         = remaining <= 0 ? 0 : (remaining < tokens ? remaining : tokens);
    }
    const int row_count = tokens * Geometry::GroupSize;

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
        partial_acc += static_cast<std::int64_t>(batch) * kCausalHeadDim * Geometry::QHeads * tokens *
                       split_count;
        partial_m += static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * split_count;
        partial_l += static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * split_count;
    }

    auto write_neutral = [&]() {
        for (int row = tid; row < row_count; row += Threads) {
            int q_head = 0;
            int token  = 0;
            causal_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
            if (causal_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_m[causal_partial_stat_index<Geometry>(q_head, token, split, tokens)] =
                    -CUDART_INF_F;
                partial_l[causal_partial_stat_index<Geometry>(q_head, token, split, tokens)] = 0.0f;
            }
        }
        for (int idx = tid; idx < row_count * D; idx += Threads) {
            const int row = idx / D;
            const int d   = idx - row * D;
            int q_head    = 0;
            int token     = 0;
            causal_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
            if (causal_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_acc[causal_partial_acc_index<Geometry>(q_head, d, token, split, tokens)] =
                    __float2bfloat16(0.0f);
            }
        }
    };

    if (kv_head < 0 || kv_head >= Geometry::KVHeads || tokens < 1 || tokens > TokenTile ||
        row_count > Wc * 16 || split_count <= 0) {
        return;
    }
    if (valid_tokens == 0) {
        write_neutral();
        return;
    }

    const std::int32_t first_pos = pos[0];
    const std::int32_t last_pos  = pos[tokens - 1];
    if (first_pos < 0 || last_pos < 0 || last_pos >= logical_capacity) {
        write_neutral();
        return;
    }

    const int window = last_pos + 1;
    const int active_split_count =
        causal_small_t_active_splits<Geometry, false>(window, split_count, TokenTile);
    if (split >= active_split_count) { return; }

    const int units_per_split = div_up(window, active_split_count);
    const int split_start     = split * units_per_split;
    const int split_end_raw   = split_start + units_per_split;
    const int split_end       = (split_end_raw < window) ? split_end_raw : window;
    if (split_start >= split_end) {
        write_neutral();
        return;
    }

    if constexpr (CacheInput::writes_cache) {
        // Same bypass rationale as the tensor-core path's cache-write loop: current-step
        // tokens are read directly from `input` below, never from the cache we just wrote,
        // so no split depends on another split's write completing first.
        for (int chunk = tid; chunk < valid_tokens * (D / 8); chunk += Threads) {
            const int token = chunk / (D / 8);
            const int d      = (chunk - token * (D / 8)) * 8;
            const int p_tok  = pos[token];
            if (p_tok >= split_start && p_tok < split_end && p_tok >= 0 &&
                p_tok < logical_capacity) {
                const std::int64_t new_off = kv_cache_int8_new_index<Geometry>(kv_head, d, token);
                const int lane2             = tid & 31;
                int physical_page = lane2 == 0 ? paged_kv_physical_page(block_table, p_tok) : 0;
                physical_page      = __shfl_sync(FullMask, physical_page, 0);
                const std::int64_t cache_off =
                    causal_cache_index<Geometry>(physical_page, kv_head, d, p_tok & kPagedKVPageMask);
                store_vec(&cache_k[cache_off], load_vec<int4>(&input.k[new_off]));
                store_vec(&cache_v[cache_off], load_vec<int4>(&input.v[new_off]));
            }
        }
        __syncthreads();
    }

    for (int r = 0; r < 16; ++r) {
        const int row = warp * 16 + r;
        if (row >= row_count) { break; }
        int q_head = 0;
        int token  = 0;
        causal_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
        if (!causal_valid_q_head<Geometry>(kv_head, q_head)) { continue; }

        const int qabs = pos[token];

        float q_reg[kDPerLane];
        {
            const int4 qv =
                load_vec<int4>(&q[causal_q_index<Geometry>(q_head, lane * kDPerLane, token)]);
            const __nv_bfloat16* qp = reinterpret_cast<const __nv_bfloat16*>(&qv);
#pragma unroll
            for (int i = 0; i < kDPerLane; ++i) { q_reg[i] = __bfloat162float(qp[i]); }
        }

        float m = -CUDART_INF_F;
        float l = 0.0f;
        float acc[kDPerLane];
#pragma unroll
        for (int i = 0; i < kDPerLane; ++i) { acc[i] = 0.0f; }

        int cached_block   = -1;
        int physical_page  = -1;
        const int key_last = (qabs < split_end - 1) ? qabs : split_end - 1;
        for (int key = split_start; key <= key_last; ++key) {
            const __nv_bfloat16* k_ptr;
            const __nv_bfloat16* v_ptr;
            bool from_new = false;
            if constexpr (CacheInput::writes_cache) {
                const int new_token = key - first_pos;
                from_new = new_token >= 0 && new_token < valid_tokens && key >= first_pos;
                if (from_new) {
                    k_ptr = &input.k[kv_cache_int8_new_index<Geometry>(kv_head, lane * kDPerLane, new_token)];
                    v_ptr = &input.v[kv_cache_int8_new_index<Geometry>(kv_head, lane * kDPerLane, new_token)];
                }
            }
            if (!from_new) {
                const int block = key >> kPagedKVPageShift;
                if (block != cached_block) {
                    physical_page = block_table[block];
                    cached_block  = block;
                }
                const std::int64_t off = causal_cache_index<Geometry>(
                    physical_page, kv_head, lane * kDPerLane, key & kPagedKVPageMask);
                k_ptr = &cache_k[off];
                v_ptr = &cache_v[off];
            }

            const int4 kv           = load_vec<int4>(k_ptr);
            const __nv_bfloat16* kp = reinterpret_cast<const __nv_bfloat16*>(&kv);
            float partial           = 0.0f;
#pragma unroll
            for (int i = 0; i < kDPerLane; ++i) {
                partial = fmaf(q_reg[i], __bfloat162float(kp[i]), partial);
            }
            const float score = warp_sum(partial, FullMask) * scale;

            const float new_m = fmaxf(m, score);
            const float alpha = (m == -CUDART_INF_F) ? 0.0f : exp2_approx((m - new_m) * Log2E);
            const float p      = exp2_approx((score - new_m) * Log2E);
            l = l * alpha + p;
            m = new_m;

            const int4 vv           = load_vec<int4>(v_ptr);
            const __nv_bfloat16* vp = reinterpret_cast<const __nv_bfloat16*>(&vv);
#pragma unroll
            for (int i = 0; i < kDPerLane; ++i) {
                acc[i] = fmaf(acc[i], alpha, p * __bfloat162float(vp[i]));
            }
        }

        if (lane == 0) {
            partial_m[causal_partial_stat_index<Geometry>(q_head, token, split, tokens)] = m;
            partial_l[causal_partial_stat_index<Geometry>(q_head, token, split, tokens)] = l;
        }
        const std::int64_t dst_base =
            causal_partial_acc_index<Geometry>(q_head, lane * kDPerLane, token, split, tokens);
        __nv_bfloat16 out8[kDPerLane];
#pragma unroll
        for (int i = 0; i < kDPerLane; ++i) { out8[i] = __float2bfloat16(acc[i]); }
        store_vec(&partial_acc[dst_base], *reinterpret_cast<const int4*>(out8));
    }
#endif // __CUDA_ARCH__ >= 800
}

} // namespace ninfer::ops
