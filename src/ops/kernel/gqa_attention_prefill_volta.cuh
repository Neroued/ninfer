#pragma once

// ninfer::ops - Volta (sm_70) tensor-core GQA small-T attention partial kernel.
//
// Structurally mirrors gqa_attention_small_t_tc_partial_bf16_kernel's Ampere+ branch
// (gqa_attention_decode_bf16.cuh) -- same paged-KV addressing, GQA row mapping, and
// cache-write handling, reused via the shared helpers in gqa_attention_decode.cuh -- but
// the tile/warp topology and QK^T/PV compute core are rebuilt from scratch on Volta's
// mma.sync.m8n8k4 primitives (ops/common/volta_mma.cuh), since Volta has neither
// ldmatrix nor m16n8k16. Writes the same partial_acc/partial_m/partial_l format the
// existing reduce kernel already consumes, so no changes are needed anywhere else in the
// online-softmax merge pipeline.
//
// Why this kernel's tile topology differs from the Ampere+ kernel's (see the V100 performance summary
// for the full derivation -- this is the load-bearing design note, read it before changing
// any of the constants below):
//
//   1. Register wall. Ampere's ldmatrix-based PV tile packs 2 output rows x 2 columns per
//      thread per mma call; Volta's mma.m8n8k4 packs only 1 row x 8 columns per thread per
//      call. Holding the *entire* D=256 head-dim output resident across the whole key loop
//      (as the Ampere kernel does, ~128 registers) would cost Volta 32 chunks x 8 floats =
//      256 registers/thread for a *single* output row -- already at Volta's hard 255/thread
//      cap with nothing left for Q/K fragments or loop state. Fix: split the head dimension
//      across warps (DimSplit=4 below) instead of giving each warp the full D range, so each
//      warp only holds D/4=64 columns resident (8 chunks x 8 floats = 64 registers). Warps
//      that share a row-tile redundantly recompute the (cheap, register-light) QK^T +
//      online-softmax step identically and only diverge for the PV accumulate.
//
//   2. Shared-memory wall. This kernel reloads Q from shared memory every key-tile
//      iteration rather than keeping Q fragments resident in registers (that trade-off is
//      what keeps (1)'s register budget low -- persisting Q *and* a full-width accumulator
//      would both want ~128 registers, right back at the wall). That means Q must stay
//      resident in shared memory for the whole key loop, so -- unlike the Ampere kernel,
//      which stages Q into the same shared buffer it later reuses for K/V once Q has been
//      consumed into registers -- this kernel needs separate, simultaneously-live q_s/k_s/
//      v_s buffers. To fit that under the 48KB static-shared-memory default, Br is fixed at
//      one 32-row Volta tile (not scaled by warp count) and Bc is halved to 16 keys/tile;
//      q_s+k_s+v_s then costs 16KB+8KB+8KB = 32KB, comfortably under budget.
//
//   3. Row capacity. Fixing Br=32 means a single tile can't cover every (TokenTile x
//      GroupSize) row count this op needs (up to 6*8=48 for the widest GQA geometry this
//      codebase instantiates). Rather than scale Br back up (reintroducing the shared-memory
//      wall), the kernel loops over row-tile "passes": each pass restages Q for a different
//      32-row slice and re-walks the whole key range. At most 2 passes are ever needed for
//      any geometry this codebase instantiates. This trades some redundant K/V global/cache
//      reads across passes for a fixed, small, well-understood resource footprint --
//      consistent with this kernel's stated correctness-first phase 1 priority.

#include "ops/common/volta_mma.cuh"
#include "ops/kernel/gqa_attention_decode.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops {

template <typename Geometry, int TokenTile, int WarpsPerCta, bool MultiBatch, bool Masked,
          typename CacheInput>
__launch_bounds__(128, 2) __global__ void gqa_attention_small_t_tc_volta_partial_kernel(
    const __nv_bfloat16* q, CacheInput input, const std::int32_t* pos, __nv_bfloat16* cache_k,
    __nv_bfloat16* cache_v, const std::int32_t* block_tables, const std::int32_t* valid_columns,
    const std::int32_t* table_rows, std::int32_t table_stride, std::int32_t tokens,
    std::int32_t full_width, std::int32_t column_begin, std::int32_t logical_capacity, float scale,
    __nv_bfloat16* partial_acc, float* partial_m, float* partial_l) {
    static_assert(TokenTile >= 1 && TokenTile <= 6);
    static_assert(WarpsPerCta == 4, "this kernel always splits the head dim 4 ways -- see file comment");

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700
    constexpr int DimSplit      = WarpsPerCta; // warps split the head dim, not the row range
    constexpr int Br            = 32;          // one Volta tile's worth of rows per pass
    constexpr int Bc            = 16;          // keys per shared-memory tile
    constexpr int D             = kGqaHeadDim;
    constexpr int Threads       = DimSplit * 32;
    constexpr int DChunks       = D / 8;           // QK^T: full head-dim contraction, unsplit
    constexpr int PVChunks      = Bc / 8;          // key sub-groups per Bc tile
    constexpr int DSlice        = D / DimSplit;    // this warp's PV output width
    constexpr int DChunksLocal  = DSlice / 8;       // this warp's resident accumulator chunks
    constexpr int PageIds       = 64;
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;

    // Declared fp16 (not bf16): Volta's mma.sync.m8n8k4 only accepts fp16 operands. K/V
    // are cp.async'd here as raw bf16 bytes (matching the bf16 KV cache/input) and then
    // converted in place to fp16 immediately after staging -- see the conversion passes
    // below. Q is converted inline at the point of load, since it's read scalar-wise
    // rather than via cp.async.
    // Padded row stride -- see the identical note in gqa_attention_decode_i8_tc_volta.cuh. The
    // mma feed reads down a column of these tiles, so an unpadded 512-byte row stride puts every
    // row on the same shared-memory bank and serializes the load ~21 ways. Worth 2.56x on the
    // int8 sibling; this kernel had the same layout and the same bug.
    constexpr int SmemPad    = 8;
    constexpr int SmemStride = D + SmemPad;
    static_assert(SmemPad % 8 == 0, "pad must preserve 16-byte alignment of the staged copies");
    __shared__ __align__(16) half q_s[Br * SmemStride];
    __shared__ __align__(16) half k_s[Bc * SmemStride];
    __shared__ __align__(16) half v_s[Bc * SmemStride];
    __shared__ std::int32_t physical_pages_s[PageIds];

    const int kv_head     = static_cast<int>(blockIdx.x);
    const int split       = static_cast<int>(blockIdx.y);
    const int batch       = MultiBatch ? static_cast<int>(blockIdx.z) : 0;
    const int split_count = static_cast<int>(gridDim.y);
    const int tid         = static_cast<int>(threadIdx.x);
    const int warp        = tid >> 5;
    const int lane        = tid & 31;
    const int dim_warp    = warp; // DimSplit == WarpsPerCta: every warp is a dim-slice of the one row-tile
    int valid_tokens       = tokens;
    if constexpr (Masked) {
        const int remaining = valid_columns[batch] - column_begin;
        valid_tokens         = remaining <= 0 ? 0 : (remaining < tokens ? remaining : tokens);
    }
    const int row_count = tokens * Geometry::GroupSize;

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
        partial_acc += static_cast<std::int64_t>(batch) * kGqaHeadDim * Geometry::QHeads * tokens *
                       split_count;
        partial_m += static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * split_count;
        partial_l += static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * split_count;
    }

    auto write_neutral = [&]() {
        for (int row = tid; row < row_count; row += Threads) {
            int q_head = 0;
            int token  = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
            if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] =
                    -CUDART_INF_F;
                partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = 0.0f;
            }
        }
        for (int idx = tid; idx < row_count * D; idx += Threads) {
            const int row = idx / D;
            const int d   = idx - row * D;
            int q_head    = 0;
            int token     = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
            if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_acc[gqa_partial_acc_index<Geometry>(q_head, d, token, split, tokens)] =
                    __float2bfloat16(0.0f);
            }
        }
    };

    if (kv_head < 0 || kv_head >= Geometry::KVHeads || tokens < 1 || tokens > TokenTile ||
        split_count <= 0) {
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
        gqa_small_t_active_splits<Geometry, false>(window, split_count, TokenTile);
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
        for (int chunk = tid; chunk < valid_tokens * (D / 8); chunk += Threads) {
            const int token = chunk / (D / 8);
            const int d     = (chunk - token * (D / 8)) * 8;
            const int p_tok = pos[token];
            if (p_tok >= split_start && p_tok < split_end && p_tok >= 0 &&
                p_tok < logical_capacity) {
                const std::int64_t new_off = gqa_kv_new_index<Geometry>(kv_head, d, token);
                const int lane2             = tid & 31;
                int physical_page = lane2 == 0 ? paged_kv_physical_page(block_table, p_tok) : 0;
                physical_page      = __shfl_sync(FullMask, physical_page, 0);
                const std::int64_t cache_off =
                    gqa_cache_index<Geometry>(physical_page, kv_head, d, p_tok & kPagedKVPageMask);
                store_vec(&cache_k[cache_off], load_vec<int4>(&input.k[new_off]));
                store_vec(&cache_v[cache_off], load_vec<int4>(&input.v[new_off]));
            }
        }
        __syncthreads();
    }

    // One row-tile "pass" per Br=32 slice of this token tile's rows (at most 2 for any
    // geometry this codebase instantiates -- see file comment #3). Each pass restages Q
    // and re-walks the whole key range for its own 32-row slice.
    for (int row_base = 0; row_base < row_count; row_base += Br) {
        const int rows_here = (row_count - row_base < Br) ? (row_count - row_base) : Br;

        for (int idx = tid; idx < Br * D; idx += Threads) {
            const int row = idx / D;
            const int d   = idx - row * D;
            int q_head     = 0;
            int token      = 0;
            __nv_bfloat16 value = __float2bfloat16(0.0f);
            if (row < rows_here) {
                gqa_small_t_tc_row_to_qt<Geometry>(row_base + row, tokens, kv_head, q_head, token);
                if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
                    value = q[gqa_q_index<Geometry>(q_head, d, token)];
                }
            }
            q_s[row * SmemStride + d] = __float2half(__bfloat162float(value));
        }
        __syncthreads();

        int physical_page = physical_pages_s[0];

        // acc_f[c] holds this warp's DSlice-wide PV output for head-dim chunk c (columns
        // [dim_warp*DSlice + c*8, +8)), folded across key-tiles by the online-softmax
        // alpha rescale -- see file comment #1 for why this is DChunksLocal (not DChunks)
        // wide.
        float acc_f[DChunksLocal][8];
#pragma unroll
        for (int c = 0; c < DChunksLocal; ++c) {
#pragma unroll
            for (int i = 0; i < 8; ++i) { acc_f[c][i] = 0.0f; }
        }
        // Every thread's 8 D-tile registers straddle TWO distinct rows (r_lo, r_hi -- see
        // volta_mma.cuh), so their softmax state must be tracked as two independent series,
        // not collapsed into one: thread `lane` and thread `lane^2` both compute IDENTICAL
        // bm_lo/bm_hi (each is already a full 2-lane reduction), and both update m_lo/l_lo
        // and m_hi/l_hi identically from those identical inputs -- so every thread has a
        // fully correct, non-mixed view of both rows' state, and can normalize d_score's
        // row_lo group against m_lo and its row_hi group against m_hi (previously this used
        // a single shared max/alpha for both groups, silently corrupting whichever group
        // wasn't "this thread's own row" -- masked by the very first softmax update, since
        // then m_lo==m_hi==-inf makes alpha 0 regardless, but wrong from the second update
        // onward whenever the two rows' true maxima differ).
        float m_lo = -CUDART_INF_F, m_hi = -CUDART_INF_F;
        float l_lo = 0.0f, l_hi = 0.0f;

        for (int kb = 0; kb < key_blocks; ++kb) {
            const int k0 = first_tile + kb * Bc;
            if (kb != 0 && (k0 & kPagedKVPageMask) == 0) {
                physical_page = physical_pages_s[(k0 >> kPagedKVPageShift) - first_page];
            }

            for (int chunk = tid; chunk < Bc * (D / 8); chunk += Threads) {
                const int key_l      = chunk / (D / 8);
                const int d          = (chunk - key_l * (D / 8)) * 8;
                const int key = k0 + key_l;
                half* k_dst   = &k_s[key_l * SmemStride + d]; // raw bf16 bytes; converted below
                half* v_dst   = &v_s[key_l * SmemStride + d];
                if (key >= split_start && key < split_end) {
                    if constexpr (CacheInput::writes_cache) {
                        const int new_token = key - first_pos;
                        const bool from_new =
                            new_token >= 0 && new_token < valid_tokens && key >= first_pos;
                        if (from_new) {
                            const std::int64_t off = gqa_kv_new_index<Geometry>(kv_head, d, new_token);
                            ninfer::ops::cp_async<16>(k_dst, &input.k[off]);
                            ninfer::ops::cp_async<16>(v_dst, &input.v[off]);
                        } else {
                            const std::int64_t off = gqa_cache_index<Geometry>(
                                physical_page, kv_head, d, key & kPagedKVPageMask);
                            ninfer::ops::cp_async<16>(k_dst, &cache_k[off]);
                            ninfer::ops::cp_async<16>(v_dst, &cache_v[off]);
                        }
                    } else {
                        const std::int64_t off = gqa_cache_index<Geometry>(physical_page, kv_head, d,
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

            // cp.async above copied raw bf16 bytes into k_s/v_s (declared fp16 for the mma
            // primitives below). Convert those bytes to real fp16 in place: each element is
            // read and overwritten only by its own owning thread (idx is unique per thread
            // per grid-stride step), so there's no cross-thread hazard to guard against here
            // -- only the __syncthreads() after the loop, which ensures every thread's
            // conversion has landed before any thread starts reading k_s/v_s as fp16 below.
            // Walks the D real elements of each row and maps them through the padded stride, so
            // the pad columns -- which the staging copy never wrote and the mma never reads --
            // are left alone rather than converted from uninitialised shared memory.
            for (int idx = tid; idx < Bc * D; idx += Threads) {
                const int row  = idx / D;
                const int off  = row * SmemStride + (idx - row * D);
                const float kf = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(k_s)[off]);
                const float vf = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(v_s)[off]);
                k_s[off]       = __float2half(kf);
                v_s[off]       = __float2half(vf);
            }
            __syncthreads();

            // Bc=16 keys are staged together, but QK^T/softmax/PV operate on 8-key
            // sub-groups within that stage: each mma.sync.m8n8k4 call only covers an
            // 8-key "N" dimension, so each sub-group gets its own complete online-softmax
            // rescale step (same recurrence as the outer kb loop, one level finer). Every
            // dim-split warp sharing this row-tile computes this step identically and
            // redundantly -- see file comment #1.
#pragma unroll
            for (int sub = 0; sub < PVChunks; ++sub) {
                const int sub_k0 = k0 + sub * 8;

                // --- QK^T: accumulate over the full D=256 head dim, 8 real k-elements/call. ---
                float d_score[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
                for (int c = 0; c < DChunks; ++c) {
                    half2 qf[4];
                    volta_load_qp(qf, reinterpret_cast<const half2*>(&q_s[c * 8]), SmemStride / 2);
                    half2 kf[4];
                    volta_load_k(
                        kf, reinterpret_cast<const half2*>(&k_s[sub * 8 * SmemStride + c * 8]),
                        SmemStride / 2);
                    volta_mma_qk(d_score, qf, kf);
                }

                // --- Causal mask + scale. Row = volta_d_get_i(l) (0..31, this pass's local
                // row index); each thread's 8 registers span exactly two distinct rows
                // (l&2==0 vs l&2==2, see volta_mma.cuh), so two qabs lookups (not eight)
                // cover this thread's own rows. ---
                const int r_lo = volta_d_get_i(0) & ~2;
                const int r_hi = volta_d_get_i(0) | 2;
                int q_head_lo = 0, tok_lo = 0, q_head_hi = 0, tok_hi = 0;
                gqa_small_t_tc_row_to_qt<Geometry>(row_base + r_lo, tokens, kv_head, q_head_lo,
                                                   tok_lo);
                gqa_small_t_tc_row_to_qt<Geometry>(row_base + r_hi, tokens, kv_head, q_head_hi,
                                                   tok_hi);
                const int qabs_lo = (r_lo < rows_here) ? pos[tok_lo] : -1;
                const int qabs_hi = (r_hi < rows_here) ? pos[tok_hi] : -1;
#pragma unroll
                for (int l = 0; l < 8; ++l) {
                    const int row   = volta_d_get_i(l);
                    const int col   = volta_d_get_j(l);
                    const int key   = sub_k0 + col;
                    const bool lo   = (l & 2) == 0;
                    const int qabs  = lo ? qabs_lo : qabs_hi;
                    const bool ok =
                        row < rows_here && key >= split_start && key < split_end && key <= qabs;
                    d_score[l] = ok ? d_score[l] * scale : -CUDART_INF_F;
                }

                // --- Per-row online-softmax update. Threads tid and tid^2 together hold
                // the complete 8-key row for both rows this thread touches, so a single
                // offset-2 shfl_xor gives an exact per-row reduction. ---
                float bm_lo = fmaxf(fmaxf(d_score[0], d_score[1]), fmaxf(d_score[4], d_score[5]));
                float bm_hi = fmaxf(fmaxf(d_score[2], d_score[3]), fmaxf(d_score[6], d_score[7]));
                bm_lo        = fmaxf(bm_lo, __shfl_xor_sync(FullMask, bm_lo, 2, 32));
                bm_hi        = fmaxf(bm_hi, __shfl_xor_sync(FullMask, bm_hi, 2, 32));

                const float new_m_lo = fmaxf(m_lo, bm_lo);
                const float new_m_hi = fmaxf(m_hi, bm_hi);
                const float alpha_lo =
                    (m_lo == -CUDART_INF_F) ? 0.0f : exp2_approx((m_lo - new_m_lo) * Log2E);
                const float alpha_hi =
                    (m_hi == -CUDART_INF_F) ? 0.0f : exp2_approx((m_hi - new_m_hi) * Log2E);

#pragma unroll
                for (int l = 0; l < 8; ++l) {
                    const float new_m = ((l & 2) == 0) ? new_m_lo : new_m_hi;
                    d_score[l] = (new_m > -CUDART_INF_F && d_score[l] > -CUDART_INF_F)
                                     ? exp2_approx((d_score[l] - new_m) * Log2E)
                                     : 0.0f;
                }
                float bl_lo = d_score[0] + d_score[1] + d_score[4] + d_score[5];
                float bl_hi = d_score[2] + d_score[3] + d_score[6] + d_score[7];
                bl_lo        = bl_lo + __shfl_xor_sync(FullMask, bl_lo, 2, 32);
                bl_hi        = bl_hi + __shfl_xor_sync(FullMask, bl_hi, 2, 32);

                l_lo = l_lo * alpha_lo + bl_lo;
                l_hi = l_hi * alpha_hi + bl_hi;
                m_lo = new_m_lo;
                m_hi = new_m_hi;
                // Own-row selection: this thread's PV accumulator (acc_f) is tied to output
                // row=lane specifically, which is r_lo when lane&2==0 and r_hi otherwise --
                // see volta_mma.cuh's I_MAJOR addressing for the Q/P/PV-output tile.
                const float alpha = ((lane & 2) == 0) ? alpha_lo : alpha_hi;

                half2 p[4];
                volta_softmax_to_half2(p, d_score);

#pragma unroll
                for (int c = 0; c < DChunksLocal; ++c) {
                    half2 vf[4];
                    volta_load_v(vf,
                                 reinterpret_cast<const half2*>(
                                     &v_s[sub * 8 * SmemStride + dim_warp * DSlice + c * 8]),
                                 SmemStride / 2);
                    half2 pv[4] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
                    volta_mma_pv(pv, p, vf);
#pragma unroll
                    for (int n = 0; n < 4; ++n) {
                        const float2 contrib = __half22float2(pv[n]);
                        acc_f[c][2 * n + 0]   = acc_f[c][2 * n + 0] * alpha + contrib.x;
                        acc_f[c][2 * n + 1]   = acc_f[c][2 * n + 1] * alpha + contrib.y;
                    }
                }
            }
            __syncthreads();
        }

        // --- Write-out: partial_m/partial_l once per row-tile (dim_warp==0 only -- every
        // dim-split warp computed the same m/l redundantly), partial_acc per dim-split
        // warp's own DSlice-wide column range. ---
        const int row      = lane;
        const float own_m = ((lane & 2) == 0) ? m_lo : m_hi;
        const float own_l = ((lane & 2) == 0) ? l_lo : l_hi;
        if (dim_warp == 0 && row < rows_here) {
            int q_head = 0;
            int token  = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row_base + row, tokens, kv_head, q_head, token);
            if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = own_m;
                partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = own_l;
            }
        }
        if (row < rows_here) {
            int q_head = 0;
            int token  = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row_base + row, tokens, kv_head, q_head, token);
            if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
#pragma unroll
                for (int c = 0; c < DChunksLocal; ++c) {
                    const int d = dim_warp * DSlice + c * 8;
                    __nv_bfloat16 out8[8];
#pragma unroll
                    for (int i = 0; i < 8; ++i) { out8[i] = __float2bfloat16(acc_f[c][i]); }
                    const std::int64_t dst =
                        gqa_partial_acc_index<Geometry>(q_head, d, token, split, tokens);
                    store_vec(&partial_acc[dst], *reinterpret_cast<const int4*>(out8));
                }
            }
        }
        __syncthreads();
    }
#endif // !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700
}

} // namespace ninfer::ops
