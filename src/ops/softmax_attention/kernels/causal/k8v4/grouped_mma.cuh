#pragma once
#include "ops/kv_cache/kernels/write/k8v4.cuh"
#include "ops/kv_cache/kernels/hadamard.cuh"
#include "ops/kv_cache/kernels/layout.cuh"
#include "ops/softmax_attention/kernels/causal/operands.h"
#include "ops/common/quant/fp8.cuh"
#include "ops/common/quant/nvfp4.cuh"
#include "ops/softmax_attention/kernels/causal/k8v4/schedule.cuh"
#include "ops/softmax_attention/kernels/causal/partition.h"
#include "ops/softmax_attention/kernels/causal/epilogue.cuh"
#include "ops/softmax_attention/kernels/softmax.cuh"

namespace ninfer::ops::detail {
// Rotated FP8 Q/K use native FP8 MMA. Rotated NVFP4 V widens to FP16 for FP32 PV accumulation.
template <class Geometry, class Schedule, bool MultiBatch, bool Masked, class CacheInput,
          bool ParallelQueries = false>
__launch_bounds__(Schedule::kThreads, Schedule::kMinBlocks) __global__
    void k8v4_kv_grouped_mma_kernel(
        const __nv_bfloat16* q, CacheInput input, const std::int32_t* positions,
        typename K8V4KvCacheView<CacheInput::writes_cache>::Code* cache_k,
        typename K8V4KvCacheView<CacheInput::writes_cache>::Code* cache_v,
        typename K8V4KvCacheView<CacheInput::writes_cache>::KeyScale* cache_k_scale,
        typename K8V4KvCacheView<CacheInput::writes_cache>::ValueScale* cache_v_scale,
        const std::int64_t* block_tables, const std::int32_t* valid_columns,
        const std::int32_t* table_rows, std::int32_t table_stride, std::int32_t full_width,
        std::int32_t logical_capacity, CausalKvPartition partition, float attention_scale,
        float* partial_acc, float* partial_m, float* partial_l) {
    constexpr int TokenTile = Schedule::kTokenTile, WarpsPerCta = Schedule::kWarps;
    constexpr int KeyBlock             = Schedule::kKeyRows;
    constexpr bool DynamicArena        = Schedule::kDynamicArena;
    constexpr int Wc                   = WarpsPerCta;
    constexpr int RowCount             = TokenTile * Geometry::GroupSize;
    constexpr int RowTiles             = (RowCount + 15) / 16;
    constexpr int Br                   = RowTiles * 16;
    constexpr int Bc                   = KeyBlock;
    constexpr int D                    = 256;
    constexpr int DB16                 = D / 2;
    constexpr int Threads              = Wc * 32;
    constexpr int QKKs                 = D / 32;
    constexpr int QKNt                 = Bc / 8;
    constexpr int PStride              = Bc == 32 ? 64 : Bc;
    constexpr int ConsumerWarpsPerTile = Wc / RowTiles;
    constexpr int PVNtPerWarp          = D / (ConsumerWarpsPerTile * 8);
    constexpr int PVKs                 = Bc / 16;
    constexpr int ProducerThreads      = RowTiles * 32;
    constexpr int VLoaderThreads       = Threads - ProducerThreads;
    constexpr float Log2E              = common::kLog2E;
    constexpr unsigned FullMask        = 0xffffffffU;

    static_assert(TokenTile >= 1 && TokenTile * Geometry::GroupSize <= 64);
    static_assert(Bc == 32 || Bc == 64);
    static_assert(RowTiles >= 1 && RowTiles <= 4);
    static_assert(Wc > RowTiles && Wc % RowTiles == 0);
    static_assert(PVNtPerWarp == 4 || PVNtPerWarp == 8 || PVNtPerWarp == 16);
    static_assert(QKKs == 8);
    __shared__ __align__(16) std::uint8_t q_s[Br * D];
    __shared__ __align__(16) std::uint8_t static_arena[DynamicArena ? 16 : 7 * Bc * D / 2];
    extern __shared__ __align__(16) std::uint8_t dynamic_arena[];
    std::uint8_t* arena   = DynamicArena ? dynamic_arena : static_arena;
    float* q_scale_tmp    = reinterpret_cast<float*>(arena);
    std::uint8_t* k_fp8   = arena;
    std::uint8_t* v_nvfp4 = arena + Bc * D;
    __half* v_f16         = reinterpret_cast<__half*>(arena + 3 * Bc * D / 2);
    __nv_bfloat16* q_b16  = reinterpret_cast<__nv_bfloat16*>(q_s);
    __nv_bfloat16* k_b16  = reinterpret_cast<__nv_bfloat16*>(k_fp8);
    __shared__ __align__(16) __half p_s[Br * PStride];
    __shared__ float alpha_s[Br];
    __shared__ __align__(16) std::uint8_t k_scale_s[2 * Bc];
    __shared__ __align__(16) std::uint8_t v_scale_s[Bc * kv::kNvfp4Groups];

    static_assert(!ParallelQueries || !CacheInput::writes_cache);
    const int kv_head      = ParallelQueries ? blockIdx.x % Geometry::KVHeads : blockIdx.x;
    const int column_begin = ParallelQueries ? (blockIdx.x / Geometry::KVHeads) * TokenTile : 0;
    const int tile_tokens = ParallelQueries ? min(TokenTile, full_width - column_begin) : TokenTile;
    const int live_row_tiles = (tile_tokens * Geometry::GroupSize + 15) / 16;
    const int partial_width = full_width, partial_begin = column_begin;
    const int split       = static_cast<int>(blockIdx.y);
    const int batch       = MultiBatch ? static_cast<int>(blockIdx.z) : 0;
    const int split_count = static_cast<int>(gridDim.y);
    const int tid         = static_cast<int>(threadIdx.x);
    const int warp        = tid >> 5;
    const int lane        = tid & 31;

    int valid_tokens = tile_tokens;
    if constexpr (Masked) {
        const int remaining = valid_columns[batch] - column_begin;
        valid_tokens        = remaining <= 0 ? 0 : min(remaining, tile_tokens);
    }
    std::int64_t column_base = column_begin;
    if constexpr (MultiBatch) { column_base += static_cast<std::int64_t>(batch) * full_width; }
    q += static_cast<std::int64_t>(D) * Geometry::QHeads * column_base;
    const int last_pos = positions[(MultiBatch ? batch * full_width : 0) + full_width - 1];
    positions += column_base;
    if constexpr (CacheInput::writes_cache) {
        input.k += static_cast<std::int64_t>(D) * Geometry::KVHeads * column_base;
        input.v += static_cast<std::int64_t>(D) * Geometry::KVHeads * column_base;
    }
    const int table_row = table_rows == nullptr ? 0 : table_rows[batch];
    const std::int64_t* block_table =
        block_tables + static_cast<std::int64_t>(table_row) * table_stride;
    if constexpr (MultiBatch) {
        partial_acc +=
            static_cast<std::int64_t>(batch) * D * Geometry::QHeads * partial_width * split_count;
        partial_m +=
            static_cast<std::int64_t>(batch) * Geometry::QHeads * partial_width * split_count;
        partial_l +=
            static_cast<std::int64_t>(batch) * Geometry::QHeads * partial_width * split_count;
    }

    if (valid_tokens == 0) return;
    if (positions[0] < 0 || last_pos < 0 || last_pos >= logical_capacity) return;
    const int window             = last_pos + 1;
    const int active_split_count = partition.active(window);
    if (split >= active_split_count) return;
    const int logical_tiles = common::div_up(window, Bc);
    const int split_start   = (split * logical_tiles / active_split_count) * Bc;
    const int split_end     = min(((split + 1) * logical_tiles / active_split_count) * Bc, window);
    const int first_tile    = split_start;
    const int key_blocks    = common::div_up(split_end - first_tile, Bc);

    if constexpr (CacheInput::writes_cache) {
        float* append_scratch = reinterpret_cast<float*>(arena);
        for (int token = warp; token < valid_tokens; token += Wc) {
            const int position = positions[token];
            if (position < split_start || position >= split_end) continue;
            kv::write_k8v4_row<Geometry>(input.k, input.v, cache_k, cache_v, cache_k_scale,
                                         cache_v_scale, token, kv_head,
                                         block_table[position >> kv::kPageShift],
                                         position & kv::kPageMask, lane, append_scratch + warp * D);
        }
        __syncthreads();
    }

    for (int index = tid; index < Br * D; index += Threads) q_s[index] = 0;
    for (int row = tid; row < RowCount; row += Threads) q_scale_tmp[row] = 0.0F;
    __syncthreads();

    for (int row = warp; row < RowCount; row += Wc) {
        int q_head = 0;
        int token  = 0;
        causal_row_to_qt<Geometry>(row, kv_head, q_head, token);
        float values[8];
        float local_absmax = 0.0F;
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int d = lane + 32 * r;
            values[r]   = token < valid_tokens
                              ? __bfloat162float(q[causal_q_index<Geometry>(q_head, d, token)])
                              : 0.0F;
        }
        kv::normalized_hadamard_d256_inplace(values, lane);
#pragma unroll
        for (float value : values) local_absmax = fmaxf(local_absmax, fabsf(value));
        const float absmax = common::warp_max(local_absmax, FullMask);
        const float qs     = absmax > 0.0F ? absmax / common::kFp8E4M3MaxFinite : 0.0F;
        const float inv    = qs > 0.0F ? 1.0F / qs : 0.0F;
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int d = lane + 32 * r;
            causal_store_query_code(q_s, row, d, common::fp8_quant_code(values[r], inv));
        }
        if (lane == 0) q_scale_tmp[row] = qs;
    }
    __syncthreads();

    const int gid      = lane >> 2;
    const int lid      = lane & 3;
    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_rin    = lane & 7;
    const int b_koff   = ((lane >> 3) & 1) << 3;

    float q_scale_r0 = 0.0F;
    float q_scale_r1 = 0.0F;
    if (warp < RowTiles) {
        const int row0 = warp * 16 + gid;
        float qs0 =
            (lid == 0 && row0 < tile_tokens * Geometry::GroupSize) ? q_scale_tmp[row0] : 0.0F;
        float qs1  = (lid == 0 && row0 + 8 < RowCount) ? q_scale_tmp[row0 + 8] : 0.0F;
        q_scale_r0 = __shfl_sync(FullMask, qs0, gid * 4);
        q_scale_r1 = __shfl_sync(FullMask, qs1, gid * 4);
    }
    __syncthreads();

    float acc[PVNtPerWarp][4];
#pragma unroll
    for (int n = 0; n < PVNtPerWarp; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[n][i] = 0.0F;
    }
    float m0 = -CUDART_INF_F;
    float m1 = -CUDART_INF_F;
    float l0 = 0.0F;
    float l1 = 0.0F;

    auto issue_kv_tile = [&](int tile_k0, std::int64_t record_offset) {
        for (int key_l = tid; key_l < Bc; key_l += Threads) {
            const int key             = tile_k0 + key_l;
            std::uint8_t* v_scale_dst = v_scale_s + key_l * kv::kNvfp4Groups;
            if (key >= split_start && key < split_end) {
                const std::int64_t k_scale_offset =
                    kv::Fp8G128Layout::scale_offset(record_offset, kv_head, key & kv::kPageMask);
                const std::int64_t v_scale_offset = kv::Nvfp4G16Layout::scale_offset(
                    record_offset, kv_head, 0, key & kv::kPageMask);
                *reinterpret_cast<std::uint16_t*>(k_scale_s + 2 * key_l) =
                    *reinterpret_cast<const std::uint16_t*>(cache_k_scale + k_scale_offset);
                common::cp_async<16>(v_scale_dst, cache_v_scale + v_scale_offset);
            } else {
                *reinterpret_cast<std::uint16_t*>(k_scale_s + 2 * key_l) = 0;
                common::store_vec(v_scale_dst, make_int4(0, 0, 0, 0));
            }
        }
#pragma unroll 1
        for (int chunk = tid; chunk < Bc * (D / 16); chunk += Threads) {
            const int key_l = chunk / (D / 16);
            const int dc    = chunk - key_l * (D / 16);
            const int d     = dc * 16;
            const int key   = tile_k0 + key_l;
            std::uint8_t* k_dst =
                &k_fp8[(key_l * DB16 + common::swizzle_xor_64(key_l, dc * 8)) * 2];
            if (key >= split_start && key < split_end) {
                const std::int64_t code_offset =
                    kv::Fp8G128Layout::code_offset(record_offset, kv_head, d, key & kv::kPageMask);
                common::cp_async<16, Cache::cg>(k_dst, &cache_k[code_offset]);
            } else {
                common::store_vec(k_dst, make_int4(0, 0, 0, 0));
            }
        }
#pragma unroll 1
        for (int chunk = tid; chunk < Bc * (D / 32); chunk += Threads) {
            const int key_l     = chunk / (D / 32);
            const int dc        = chunk - key_l * (D / 32);
            const int d         = dc * 32;
            const int key       = tile_k0 + key_l;
            std::uint8_t* v_dst = &v_nvfp4[key_l * (D / 2) + d / 2];
            if (key >= split_start && key < split_end) {
                const std::int64_t code_offset =
                    kv::Nvfp4G16Layout::code_offset(record_offset, kv_head, d, key & kv::kPageMask);
                common::cp_async<16, Cache::cg>(v_dst, &cache_v[code_offset]);
            } else {
                common::store_vec(v_dst, make_int4(0, 0, 0, 0));
            }
        }
        common::cp_commit();
    };

    std::int64_t record_offset = block_table[first_tile >> kv::kPageShift];
    issue_kv_tile(first_tile, record_offset);
    common::cp_wait<0>();
    __syncthreads();

    for (int kb = 0; kb < key_blocks; ++kb) {
        const int k0 = first_tile + kb * Bc;
        if (warp < RowTiles) {
            if (warp < live_row_tiles) {
                const int row_base = warp * 16;
                __half* p_sw       = &p_s[row_base * PStride];
                float score[QKNt][4];
#pragma unroll
                for (int nt = 0; nt < QKNt; ++nt) {
                    score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0F;
                }
#pragma unroll
                for (int kk = 0; kk < QKKs; ++kk) {
                    const int acol = kk * 16 + a_coloff;
                    unsigned af[4];
                    common::ldmatrix_x4(
                        af[0], af[1], af[2], af[3],
                        common::smem_addr(
                            &q_b16[(row_base + a_rowoff) * DB16 +
                                   common::swizzle_xor_64(row_base + a_rowoff, acol)]));
#pragma unroll
                    for (int nt = 0; nt < QKNt; ++nt) {
                        const int brow = nt * 8 + b_rin;
                        const int bcol = kk * 16 + b_koff;
                        unsigned bf[2];
                        common::ldmatrix_x2(
                            bf[0], bf[1],
                            common::smem_addr(
                                &k_b16[brow * DB16 + common::swizzle_xor_64(brow, bcol)]));
                        common::mma_fp8_e4m3_scaled(score[nt][0], score[nt][1], score[nt][2],
                                                    score[nt][3], af[0], af[1], af[2], af[3], bf[0],
                                                    bf[1], 0x7fu,
                                                    k_scale_s[2 * (nt * 8 + gid) + kk / 4]);
                    }
                }
#pragma unroll
                for (int nt = 0; nt < QKNt; ++nt) {
                    score[nt][0] *= q_scale_r0;
                    score[nt][1] *= q_scale_r0;
                    score[nt][2] *= q_scale_r1;
                    score[nt][3] *= q_scale_r1;
                }

                const int row0 = row_base + gid;
                const int row1 = row0 + 8;
                int q_head0 = 0, token0 = 0, q_head1 = 0, token1 = 0;
                causal_row_to_qt<Geometry>(row0, kv_head, q_head0, token0);
                causal_row_to_qt<Geometry>(row1, kv_head, q_head1, token1);
                const int qabs0 = row0 < tile_tokens * Geometry::GroupSize ? positions[token0] : -1;
                const int qabs1 = row1 < tile_tokens * Geometry::GroupSize ? positions[token1] : -1;
                float bm0       = -CUDART_INF_F;
                float bm1       = -CUDART_INF_F;
#pragma unroll
                for (int nt = 0; nt < QKNt; ++nt) {
                    const int col0 = nt * 8 + 2 * lid;
                    const int key0 = k0 + col0;
                    const int key1 = key0 + 1;
                    score[nt][0]   = row0 < tile_tokens * Geometry::GroupSize &&
                                           key0 >= split_start && key0 < split_end && key0 <= qabs0
                                         ? score[nt][0] * attention_scale
                                         : -CUDART_INF_F;
                    score[nt][1]   = row0 < tile_tokens * Geometry::GroupSize &&
                                           key1 >= split_start && key1 < split_end && key1 <= qabs0
                                         ? score[nt][1] * attention_scale
                                         : -CUDART_INF_F;
                    score[nt][2]   = row1 < tile_tokens * Geometry::GroupSize &&
                                           key0 >= split_start && key0 < split_end && key0 <= qabs1
                                         ? score[nt][2] * attention_scale
                                         : -CUDART_INF_F;
                    score[nt][3]   = row1 < tile_tokens * Geometry::GroupSize &&
                                           key1 >= split_start && key1 < split_end && key1 <= qabs1
                                         ? score[nt][3] * attention_scale
                                         : -CUDART_INF_F;
                    bm0            = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
                    bm1            = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
                }
                bm0             = common::warp_max<4>(bm0, FullMask);
                bm1             = common::warp_max<4>(bm1, FullMask);
                const float nm0 = fmaxf(m0, bm0);
                const float nm1 = fmaxf(m1, bm1);
                const float alpha0 =
                    m0 == -CUDART_INF_F ? 0.0F : causal_exp_difference(m0, nm0, Log2E);
                const float alpha1 =
                    m1 == -CUDART_INF_F ? 0.0F : causal_exp_difference(m1, nm1, Log2E);
                float bl0 = 0.0F;
                float bl1 = 0.0F;
#pragma unroll
                for (int nt = 0; nt < QKNt; ++nt) {
                    const int col0  = nt * 8 + 2 * lid;
                    const float p00 = nm0 > -CUDART_INF_F && score[nt][0] > -CUDART_INF_F
                                          ? causal_exp_difference(score[nt][0], nm0, Log2E)
                                          : 0.0F;
                    const float p01 = nm0 > -CUDART_INF_F && score[nt][1] > -CUDART_INF_F
                                          ? causal_exp_difference(score[nt][1], nm0, Log2E)
                                          : 0.0F;
                    const float p10 = nm1 > -CUDART_INF_F && score[nt][2] > -CUDART_INF_F
                                          ? causal_exp_difference(score[nt][2], nm1, Log2E)
                                          : 0.0F;
                    const float p11 = nm1 > -CUDART_INF_F && score[nt][3] > -CUDART_INF_F
                                          ? causal_exp_difference(score[nt][3], nm1, Log2E)
                                          : 0.0F;
                    bl0 += p00 + p01;
                    bl1 += p10 + p11;
                    *reinterpret_cast<__half2*>(
                        &p_sw[gid * PStride + common::swizzle_xor_64(gid, col0)]) =
                        __floats2half2_rn(p00, p01);
                    *reinterpret_cast<__half2*>(
                        &p_sw[(gid + 8) * PStride + common::swizzle_xor_64(gid + 8, col0)]) =
                        __floats2half2_rn(p10, p11);
                }
                bl0 = common::warp_sum<4>(bl0, FullMask);
                bl1 = common::warp_sum<4>(bl1, FullMask);
                l0  = __fmaf_rn(l0, alpha0, bl0);
                l1  = __fmaf_rn(l1, alpha1, bl1);
                m0  = nm0;
                m1  = nm1;
                if (lid == 0) {
                    alpha_s[row0] = alpha0;
                    alpha_s[row1] = alpha1;
                }
            }
        } else {
            const int loader_tid = tid - ProducerThreads;
#pragma unroll 1
            for (int chunk = loader_tid; chunk < Bc * (D / 8); chunk += VLoaderThreads) {
                const int key_l = chunk / (D / 8);
                const int dc    = chunk - key_l * (D / 8);
                const int d     = dc * 8;
                const int key   = k0 + key_l;
                __half* dst     = &v_f16[key_l * D + common::swizzle_xor_64(key_l, d)];
                if (key >= split_start && key < split_end) {
                    common::store_vec(
                        dst, common::nvfp4_dequant_f16x8(
                                 &v_nvfp4[key_l * (D / 2) + d / 2],
                                 v_scale_s[key_l * kv::kNvfp4Groups + d / kv::kNvfp4Group]));
                } else {
                    common::store_vec(dst, make_int4(0, 0, 0, 0));
                }
            }
        }
        __syncthreads();

        const bool has_next = kb + 1 < key_blocks;
        if (has_next) {
            const int next_k0 = k0 + Bc;
            if ((next_k0 & kv::kPageMask) == 0) {
                record_offset = block_table[next_k0 >> kv::kPageShift];
            }
            issue_kv_tile(next_k0, record_offset);
        }

        if (warp % RowTiles < live_row_tiles) {
            const int consumer_tile     = warp % RowTiles;
            const int consumer_slice    = warp / RowTiles;
            const int consumer_row_base = consumer_tile * 16;
            __half* p_consumer          = &p_s[consumer_row_base * PStride];
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
                    common::ldmatrix_x4(
                        pf[0], pf[1], pf[2], pf[3],
                        common::smem_addr(&p_consumer[a_rowoff * PStride +
                                                      common::swizzle_xor_64(a_rowoff, pcol)]));
                    unsigned vf[2];
                    const int vrow = k * 16 + b_koff + b_rin;
                    const int vcol = global_n * 8;
                    common::ldmatrix_x2_t(
                        vf[0], vf[1],
                        common::smem_addr(&v_f16[vrow * D + common::swizzle_xor_64(vrow, vcol)]));
                    common::mma_f16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], pf[0], pf[1], pf[2],
                                    pf[3], vf[0], vf[1]);
                }
            }
        }
        if (has_next) { common::cp_wait<0>(); }
        __syncthreads();
    }

    if (warp < RowTiles && lid == 0) {
        const int row0 = warp * 16 + gid;
        const int row1 = row0 + 8;
        if (row0 < tile_tokens * Geometry::GroupSize) {
            int q_head = 0;
            int token  = 0;
            causal_row_to_qt<Geometry>(row0, kv_head, q_head, token);
            partial_m[causal_stat_index<Geometry>(q_head, partial_begin + token, split,
                                                  partial_width)] = m0;
            partial_l[causal_stat_index<Geometry>(q_head, partial_begin + token, split,
                                                  partial_width)] = l0;
        }
        if (row1 < tile_tokens * Geometry::GroupSize) {
            int q_head = 0;
            int token  = 0;
            causal_row_to_qt<Geometry>(row1, kv_head, q_head, token);
            partial_m[causal_stat_index<Geometry>(q_head, partial_begin + token, split,
                                                  partial_width)] = m1;
            partial_l[causal_stat_index<Geometry>(q_head, partial_begin + token, split,
                                                  partial_width)] = l1;
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
        if (row0 < tile_tokens * Geometry::GroupSize) {
            int q_head = 0;
            int token  = 0;
            causal_row_to_qt<Geometry>(row0, kv_head, q_head, token);
            const std::int64_t dst = causal_partial_index<Geometry>(
                q_head, d0, partial_begin + token, split, partial_width);
            causal_store_partial_pair(&partial_acc[dst], acc[n][0], acc[n][1]);
        }
        if (row1 < tile_tokens * Geometry::GroupSize) {
            int q_head = 0;
            int token  = 0;
            causal_row_to_qt<Geometry>(row1, kv_head, q_head, token);
            const std::int64_t dst = causal_partial_index<Geometry>(
                q_head, d0, partial_begin + token, split, partial_width);
            causal_store_partial_pair(&partial_acc[dst], acc[n][2], acc[n][3]);
        }
    }
}

} // namespace ninfer::ops::detail
