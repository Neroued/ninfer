#pragma once
// INT8-KV prefill attention with two ping-pong warp groups (template_launch.cuh: PV with FP16
// accumulation, NINFER_ATTN_PV_F16ACC=1).
//
// int8_kv_tiled_mma_kernel runs each 64-key tile in phases that never overlap: QK and the softmax
// on four producer warps (one dependent chain per SM sub-partition) while twelve warps stage V as
// FP16, then PV on all sixteen. Tensor work per tile and sub-partition is 64 IMMA + 128 HMMA at 16
// cycles = 3,072 cycles; the kernel takes ~7,300 (ncu, 5070 Ti). Spreading QK over all warps
// (bit-identical) was slower: every warp then went through QK (tensor), softmax (ALU) and PV
// (tensor) in lockstep.
//
// Here the 64 query rows are split into two groups of eight warps (32 rows each). A group's MMA
// section t runs PV of tile t-1 and QK of tile t; its softmax of tile t follows. The sections of
// the two groups strictly alternate (named barriers B0/B1), so one group's softmax runs while the
// other group's MMAs keep the tensor pipe busy. (Measured and slower: lockstep sections of both
// groups, groups free-running one section apart, PV and QK k-steps interleaved inside a section.)
//   QK + softmax: warp (row tile, key quarter) scores 16 rows x 16 keys with the per-key arithmetic
//   of the
//     tiled kernel; the four quarters of a row tile exchange row maxima through shared memory,
//     quarter 0 keeps the row sum and adds the other quarters' probability pair sums in the
//     original key order.
//   PV: warp gw of a group covers both its row tiles x dimensions 32 gw .. 32 gw + 31 (one INT8
//   scale group):
//     raw INT8 V through ldmatrix.trans, converted exactly to FP16 times the FP16 scale in
//     registers (pipe_helpers.cuh), even and odd dimensions as separate n-tiles; FP16 accumulation
//     per tile, then acc = fma(acc, alpha, hacc) as the tiled kernel compiles it.
//   K and V: raw codes and scales double-buffered; group 1 loads them with cp.async, completion
//   through
//     mbarriers (cp.async.mbarrier.arrive.noinc), each stage refilled once both groups have read
//     it.
// Scores, maxima, probabilities, row sums and every PV product and sum are those of
// int8_kv_tiled_mma_kernel with PvHalfAcc, in the same order: bit-identical output.
#include "ops/softmax_attention/dense/causal_cache/int8/tile_io.cuh"

#include "ops/common/mbarrier.cuh"
#include "ops/softmax_attention/dense/causal_cache/int8/pipe_helpers.cuh"
#include "ops/softmax_attention/common/causal_epilogue.cuh"
#include "ops/softmax_attention/common/causal_softmax.cuh"

namespace ninfer::ops::detail {

namespace int8_kv_pp {
inline constexpr int kBr           = 64;
inline constexpr int kBc           = 64;
inline constexpr int kD            = 256;
inline constexpr int kGroups       = 4;
inline constexpr int kWarps        = 16;
inline constexpr int kThreads      = kWarps * 32;
inline constexpr int kGroupThreads = kThreads / 2;
inline constexpr int kTailSlots = 6; // probability pair sums of key quarters 1..3, two n-tiles each
// Shared memory.
inline constexpr int kQBytes      = kBr * kD;
inline constexpr int kQScaleBytes = kBr * kGroups * 4;
inline constexpr int kKvBytes     = kBc * kD;
inline constexpr int kScaleBytes  = kBc * kGroups * 2;
inline constexpr int kPBytes      = kBr * kBc * 2;
inline constexpr int kRowBytes    = kBr * 4;
inline constexpr int kMPartBytes  = 4 * kBr * 4;
inline constexpr int kSPartBytes  = kTailSlots * kBr * 4 * 4;
inline constexpr int kOffQScale   = kQBytes;
inline constexpr int kOffK        = kOffQScale + kQScaleBytes;
inline constexpr int kOffV        = kOffK + 2 * kKvBytes;
inline constexpr int kOffKScale   = kOffV + 2 * kKvBytes;
inline constexpr int kOffVScale   = kOffKScale + 2 * kScaleBytes;
inline constexpr int kOffP        = kOffVScale + 2 * kScaleBytes;
inline constexpr int kOffAlpha    = kOffP + kPBytes;
inline constexpr int kOffFinalL   = kOffAlpha + kRowBytes;
inline constexpr int kOffMPart    = kOffFinalL + kRowBytes;
inline constexpr int kOffSPart    = kOffMPart + kMPartBytes;
inline constexpr int kOffBarriers = kOffSPart + kSPartBytes;
inline constexpr int kSharedBytes = kOffBarriers + 4 * 8;
static_assert(kSharedBytes <= 99 * 1024);
static_assert(kOffBarriers % 8 == 0);
// Named barriers (0 is __syncthreads).
inline constexpr int kB0     = 1; // group 1 finished an MMA section: group 0 may start its next one
inline constexpr int kB1     = 2; // group 0 finished an MMA section: group 1 may start its own
inline constexpr int kGroup0 = 3; // + group: the group's 256 threads
inline constexpr int kLoad   = 5; // group 1's warps finished reading before it refills a stage

__device__ __forceinline__ void cp_async_mbarrier_arrive(std::uint64_t* barrier) {
    asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(smem_addr(barrier))
                 : "memory");
}
} // namespace int8_kv_pp

template <typename Geometry, typename Metadata>
__global__ __launch_bounds__(int8_kv_pp::kThreads, 1) void int8_kv_tiled_pp_mma_kernel(
    const __nv_bfloat16* __restrict__ q, const std::int8_t* __restrict__ cache_k,
    const std::int8_t* __restrict__ cache_v, const __half* __restrict__ cache_k_scale,
    const __half* __restrict__ cache_v_scale, Metadata metadata,
    const std::int32_t* __restrict__ positions, float scale, __nv_bfloat16* __restrict__ out,
    std::int32_t width) {
    using namespace int8_kv_pp;
    using int8_kv_pipe::arrive;
    using int8_kv_pipe::sync;
    constexpr int D             = kD;
    constexpr int Br            = kBr;
    constexpr int Bc            = kBc;
    constexpr int DB16          = D / 2;
    constexpr int Groups        = kGroups;
    constexpr int GroupKc       = kKVCacheInt8Group / 32;
    constexpr int QuarterNt     = 2;
    constexpr int PVKs          = Bc / 16;
    constexpr int CodeChunks    = Bc * (D / 16) / kGroupThreads;
    constexpr float Log2E       = kLog2E;
    constexpr unsigned FullMask = 0xffffffffu;
    static_assert(GroupKc == 2 && Groups == kKVCacheInt8Groups);
    static_assert(Bc == kPagedKVPageSize && Bc * (D / 16) % kGroupThreads == 0);

    extern __shared__ __align__(16) unsigned char smem_raw[];
    std::int8_t* q_i8     = reinterpret_cast<std::int8_t*>(smem_raw);
    float* q_scale        = reinterpret_cast<float*>(smem_raw + kOffQScale);
    std::int8_t* k_buf    = reinterpret_cast<std::int8_t*>(smem_raw + kOffK); // [2]
    std::int8_t* v_buf    = reinterpret_cast<std::int8_t*>(smem_raw + kOffV); // [2]
    __half* k_scale_buf   = reinterpret_cast<__half*>(smem_raw + kOffKScale); // [2]
    __half* v_scale_buf   = reinterpret_cast<__half*>(smem_raw + kOffVScale); // [2]
    __half* p_s           = reinterpret_cast<__half*>(smem_raw + kOffP);
    float* alpha_s        = reinterpret_cast<float*>(smem_raw + kOffAlpha);
    float* final_l_s      = reinterpret_cast<float*>(smem_raw + kOffFinalL);
    float* m_part_s       = reinterpret_cast<float*>(smem_raw + kOffMPart);
    float* s_part_s       = reinterpret_cast<float*>(smem_raw + kOffSPart);
    std::uint64_t* full_k = reinterpret_cast<std::uint64_t*>(smem_raw + kOffBarriers); // [2]
    std::uint64_t* full_v = full_k + 2;                                                // [2]
    __nv_bfloat16* q_b16  = reinterpret_cast<__nv_bfloat16*>(q_i8);

    // Query tiles in reverse: the last tiles of a chunk see the most keys and start first.
    const int q_block = static_cast<int>(gridDim.x) - 1 - static_cast<int>(blockIdx.x);
    const int q_head  = static_cast<int>(blockIdx.y);
    const int tid     = static_cast<int>(threadIdx.x);
    const int warp    = tid >> 5;
    const int lane    = tid & 31;
    const int q0      = q_block * Br;
    const int kv_head = q_head / Geometry::GroupSize;
    const int tokens  = metadata.valid_tokens(width);
    if (q_head >= Geometry::QHeads || q0 >= width) { return; }
    if (q0 >= tokens) {
        causal_zero_rows<Geometry>(out, q_head, q0, min(q0 + Br, width), tid, kThreads);
        return;
    }
    const int base_pos              = positions[0];
    const std::int32_t* block_table = metadata.block_table();

    const int tile_rows     = min(Br, tokens - q0);
    const int max_query_abs = base_pos + q0 + tile_rows - 1;
    const int key_blocks    = max_query_abs / Bc + 1;

    const int group = warp >> 3;
    const int gw    = warp & 7;

    // Raw codes and scales of tile `tile` (K or V) into its stage; group 1's threads only.
    auto load_tile = [&](int tile, bool keys) {
        const int tile_k0       = tile * Bc;
        const int page          = block_table[tile_k0 >> kPagedKVPageShift];
        const int stage         = tile & 1;
        const std::int8_t* src  = keys ? cache_k : cache_v;
        const __half* scale_src = keys ? cache_k_scale : cache_v_scale;
        std::int8_t* codes      = (keys ? k_buf : v_buf) + stage * kKvBytes;
        __half* scales          = (keys ? k_scale_buf : v_scale_buf) + stage * Bc * Groups;
        const int lt            = tid - kGroupThreads;
        if (lt < Bc) {
            const int key = tile_k0 + lt;
            const std::int64_t off =
                kv_cache_int8_quant_scale_index<Geometry>(page, kv_head, 0, key & kPagedKVPageMask);
            cp_async_zfill<8>(&scales[lt * Groups], &scale_src[off], key <= max_query_abs ? 8 : 0);
        }
#pragma unroll
        for (int i = 0; i < CodeChunks; ++i) {
            const int chunk  = lt + i * kGroupThreads;
            const int key_l  = chunk / (D / 16);
            const int dc     = chunk - key_l * (D / 16);
            const int key    = tile_k0 + key_l;
            std::int8_t* dst = keys ? &codes[(key_l * DB16 + causal_swizzle(key_l, dc * 8)) * 2]
                                    : &codes[int8_kv_pipe::v_offset(key_l, dc)];
            const std::int64_t off = kv_cache_int8_quant_code_index<Geometry>(
                page, kv_head, dc * 16, key & kPagedKVPageMask);
            cp_async_zfill<16, Cache::cg>(dst, &src[off], key <= max_query_abs ? 16 : 0);
        }
        cp_async_mbarrier_arrive((keys ? full_k : full_v) + stage);
    };

    if (tid == 0) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { cta_mbarrier_init(full_k + i, kGroupThreads); }
        cta_mbarrier_fence_init();
    }
    __syncthreads();
    if (group == 1) {
        load_tile(0, true);
        if (key_blocks > 1) { load_tile(1, true); }
        load_tile(0, false);
    }

    // Quantize Q cooperatively. One full warp rotates and encodes one D256 row at a time.
    for (int row = warp; row < Br; row += kWarps) {
        float q_values[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int d = lane + 32 * r;
            q_values[r] = 0.0f;
            if (row < tile_rows) {
                q_values[r] = __bfloat162float(q[causal_q_index<Geometry>(q_head, d, q0 + row)]);
            }
        }
        normalized_hadamard_d256_inplace(q_values, lane);

#pragma unroll
        for (int grp = 0; grp < Groups; ++grp) {
            const int d0    = grp * kKVCacheInt8Group + lane;
            const int d1    = d0 + 32;
            const float x0  = q_values[2 * grp];
            const float x1  = q_values[2 * grp + 1];
            float absmax    = fmaxf(fabsf(x0), fabsf(x1));
            absmax          = warp_max(absmax, FullMask);
            const float qs  = absmax > 0.0f ? absmax / 127.0f : 0.0f;
            const float inv = qs > 0.0f ? 1.0f / qs : 0.0f;
            causal_store_query_code(q_i8, row, d0, kv_cache_int8_quant_code(x0, inv));
            causal_store_query_code(q_i8, row, d1, kv_cache_int8_quant_code(x1, inv));
            if (lane == 0) { q_scale[row * Groups + grp] = qs; }
        }
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

    // QK + softmax role: row tile and 16-key quarter. Quarter 0 (which keeps the row sum) sits on
    // two different sub-partitions per group.
    const int qk_row_tile = 2 * group + (gw & 1);
    const int quarter     = (gw >> 1) ^ (gw & 1);
    const int qk_row_base = qk_row_tile * 16;
    const int qk_row0     = qk_row_base + gid;
    const int qk_row1     = qk_row0 + 8;
    // PV role: both row tiles of the group x 32 dimensions of scale group gw / 2.
    const int pv_row_base = group * 32;
    const int pv_scale    = gw >> 1;
    const int v_key       = (lane & 7) + (((lane >> 3) & 1) << 3);
    const int v_chunk     = 2 * gw + (lane >> 4);

    // Keep two group scales live; reload the other groups from the shared Q tile.
    float q_scale_r0[Groups - 2];
    float q_scale_r1[Groups - 2];
#pragma unroll
    for (int grp = 0; grp < Groups - 2; ++grp) {
        float qs0       = lid == 0 ? q_scale[qk_row0 * Groups + grp] : 0.0f;
        float qs1       = lid == 0 ? q_scale[qk_row1 * Groups + grp] : 0.0f;
        q_scale_r0[grp] = __shfl_sync(FullMask, qs0, gid * 4);
        q_scale_r1[grp] = __shfl_sync(FullMask, qs1, gid * 4);
    }

    float acc[2][4][4];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
#pragma unroll
        for (int n = 0; n < 4; ++n) {
#pragma unroll
            for (int i = 0; i < 4; ++i) { acc[r][n][i] = 0.0f; }
        }
    }
    float running_m0     = -CUDART_INF_F;
    float running_m1     = -CUDART_INF_F;
    float running_l0     = 0.0f;
    float running_l1     = 0.0f;
    float qk_alpha0      = 0.0f;
    float qk_alpha1      = 0.0f;
    float bl0            = 0.0f;
    float bl1            = 0.0f;
    const float scale_l2 = scale * Log2E;

    // Row sum of the previous tile in the original key order: quarter 0's own pairs, then the other
    // quarters'.
    auto update_row_sum = [&]() {
        if (quarter != 0) return;
#pragma unroll
        for (int slot = 0; slot < kTailSlots; ++slot) {
            bl0 += s_part_s[(slot * Br + qk_row0) * 4 + lid];
            bl1 += s_part_s[(slot * Br + qk_row1) * 4 + lid];
        }
        bl0        = warp_sum<4>(bl0, FullMask);
        bl1        = warp_sum<4>(bl1, FullMask);
        running_l0 = __fmaf_rn(running_l0, qk_alpha0, bl0);
        running_l1 = __fmaf_rn(running_l1, qk_alpha1, bl1);
    };

    float score[QuarterNt][4];
    for (int t = 0; t <= key_blocks; ++t) {
        // ---- MMA section t: PV(t - 1), QK(t). ----
        if (group == 0) {
            if (t > 0) { sync(kB0, kThreads); }
        } else {
            sync(kB1, kThreads);
        }

        if (t > 0) {
            const int tp = t - 1;
            cta_mbarrier_wait(full_v + (tp & 1), (tp >> 1) & 1);
            const std::int8_t* vb = v_buf + (tp & 1) * kKvBytes;
            const __half* vs      = v_scale_buf + (tp & 1) * Bc * Groups;
            unsigned hacc[2][4][2];
#pragma unroll
            for (int r = 0; r < 2; ++r) {
#pragma unroll
                for (int n = 0; n < 4; ++n) { hacc[r][n][0] = hacc[r][n][1] = 0u; }
            }

            // Fragments of one k-step: P of both row tiles, raw V (ldmatrix.trans) and the V scales
            // of this thread's B-fragment keys (2 lid, 2 lid + 1; + 8, + 9). Loaded one step ahead
            // of their MMAs.
            struct PvLoad {
                unsigned pf[2][4];
                unsigned v[4];
                __half2 s_lo;
                __half2 s_hi;
            };

            auto pv_load = [&](int k) {
                PvLoad l;
#pragma unroll
                for (int r = 0; r < 2; ++r) {
                    const int prow = pv_row_base + r * 16 + a_rowoff;
                    const int pcol = k * 16 + a_coloff;
                    ldmatrix_x4(
                        l.pf[r][0], l.pf[r][1], l.pf[r][2], l.pf[r][3],
                        smem_addr(&p_s[prow * Bc + causal_probability_swizzle<Bc>(prow, pcol)]));
                }
                const int kb0 = k * 16 + 2 * lid;
                l.s_lo =
                    __halves2half2(vs[kb0 * Groups + pv_scale], vs[(kb0 + 1) * Groups + pv_scale]);
                l.s_hi = __halves2half2(vs[(kb0 + 8) * Groups + pv_scale],
                                        vs[(kb0 + 9) * Groups + pv_scale]);
                int8_kv_pipe::ldmatrix_x4_trans_b16(
                    l.v[0], l.v[1], l.v[2], l.v[3],
                    smem_addr(&vb[int8_kv_pipe::v_offset(k * 16 + v_key, v_chunk)]));
                return l;
            };
            PvLoad next = pv_load(0);
#pragma unroll
            for (int k = 0; k < PVKs; ++k) {
                const PvLoad cur = next;
                if (k + 1 < PVKs) { next = pv_load(k + 1); }
                // v[0]: keys 0-7 x dims 32 gw + 0-15, v[1]: keys 8-15 x the same, v[2] / v[3]: dims
                // + 16.
                const unsigned x0 = cur.v[0] ^ 0x80808080u, x1 = cur.v[1] ^ 0x80808080u;
                const unsigned x2 = cur.v[2] ^ 0x80808080u, x3 = cur.v[3] ^ 0x80808080u;
                // n-tile 2 b + e: even (e = 0) / odd (e = 1) dimensions of 16-dimension block b.
                unsigned bf[4][2];
                bf[0][0] = int8_kv_pipe::v_pair(x0, 0x4240u, cur.s_lo);
                bf[0][1] = int8_kv_pipe::v_pair(x1, 0x4240u, cur.s_hi);
                bf[1][0] = int8_kv_pipe::v_pair(x0, 0x4341u, cur.s_lo);
                bf[1][1] = int8_kv_pipe::v_pair(x1, 0x4341u, cur.s_hi);
                bf[2][0] = int8_kv_pipe::v_pair(x2, 0x4240u, cur.s_lo);
                bf[2][1] = int8_kv_pipe::v_pair(x3, 0x4240u, cur.s_hi);
                bf[3][0] = int8_kv_pipe::v_pair(x2, 0x4341u, cur.s_lo);
                bf[3][1] = int8_kv_pipe::v_pair(x3, 0x4341u, cur.s_hi);
#pragma unroll
                for (int r = 0; r < 2; ++r) {
#pragma unroll
                    for (int n = 0; n < 4; ++n) {
                        mma_f16_hacc(hacc[r][n][0], hacc[r][n][1], cur.pf[r][0], cur.pf[r][1],
                                     cur.pf[r][2], cur.pf[r][3], bf[n][0], bf[n][1]);
                    }
                }
            }
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const float alpha0 = alpha_s[pv_row_base + r * 16 + gid];
                const float alpha1 = alpha_s[pv_row_base + r * 16 + gid + 8];
#pragma unroll
                for (int n = 0; n < 4; ++n) {
                    const float2 lo =
                        __half22float2(*reinterpret_cast<const __half2*>(&hacc[r][n][0]));
                    const float2 hi =
                        __half22float2(*reinterpret_cast<const __half2*>(&hacc[r][n][1]));
                    acc[r][n][0] = __fmaf_rn(acc[r][n][0], alpha0, lo.x);
                    acc[r][n][1] = __fmaf_rn(acc[r][n][1], alpha0, lo.y);
                    acc[r][n][2] = __fmaf_rn(acc[r][n][2], alpha1, hi.x);
                    acc[r][n][3] = __fmaf_rn(acc[r][n][3], alpha1, hi.y);
                }
            }
        }

        if (t < key_blocks) {
            cta_mbarrier_wait(full_k + (t & 1), (t >> 1) & 1);
            const __nv_bfloat16* k_b16 =
                reinterpret_cast<const __nv_bfloat16*>(k_buf + (t & 1) * kKvBytes);
            const __half* k_scale_s = k_scale_buf + (t & 1) * Bc * Groups;
#pragma unroll
            for (int j = 0; j < QuarterNt; ++j) {
                score[j][0] = score[j][1] = score[j][2] = score[j][3] = 0.0f;
            }
#pragma unroll
            for (int grp = 0; grp < Groups; ++grp) {
                float qs0;
                float qs1;
                if (grp < Groups - 2) {
                    qs0 = q_scale_r0[grp];
                    qs1 = q_scale_r1[grp];
                } else {
                    qs0 = lid == 0 ? q_scale[qk_row0 * Groups + grp] : 0.0f;
                    qs1 = lid == 0 ? q_scale[qk_row1 * Groups + grp] : 0.0f;
                    qs0 = __shfl_sync(FullMask, qs0, gid * 4);
                    qs1 = __shfl_sync(FullMask, qs1, gid * 4);
                }

                unsigned af[GroupKc][4];
#pragma unroll
                for (int kk = 0; kk < GroupKc; ++kk) {
                    const int k    = grp * GroupKc + kk;
                    const int acol = k * 16 + a_coloff;
                    ldmatrix_x4(af[kk][0], af[kk][1], af[kk][2], af[kk][3],
                                smem_addr(&q_b16[(qk_row_base + a_rowoff) * DB16 +
                                                 causal_swizzle(qk_row_base + a_rowoff, acol)]));
                }

#pragma unroll
                for (int j = 0; j < QuarterNt; ++j) {
                    const int nt = quarter * QuarterNt + j;
                    int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
#pragma unroll
                    for (int kk = 0; kk < GroupKc; ++kk) {
                        const int k    = grp * GroupKc + kk;
                        const int brow = nt * 8 + b_rin;
                        const int bcol = k * 16 + b_koff;
                        unsigned bf[2];
                        ldmatrix_x2(bf[0], bf[1],
                                    smem_addr(&k_b16[brow * DB16 + causal_swizzle(brow, bcol)]));
                        mma_s8(c0, c1, c2, c3, af[kk][0], af[kk][1], af[kk][2], af[kk][3], bf[0],
                               bf[1]);
                    }
                    const int keya  = nt * 8 + 2 * lid;
                    const int keyb  = keya + 1;
                    const float ks0 = __half2float(k_scale_s[keya * Groups + grp]);
                    const float ks1 = __half2float(k_scale_s[keyb * Groups + grp]);
                    score[j][0]     = __fmaf_rn(qs0 * ks0, static_cast<float>(c0), score[j][0]);
                    score[j][1]     = __fmaf_rn(qs0 * ks1, static_cast<float>(c1), score[j][1]);
                    score[j][2]     = __fmaf_rn(qs1 * ks0, static_cast<float>(c2), score[j][2]);
                    score[j][3]     = __fmaf_rn(qs1 * ks1, static_cast<float>(c3), score[j][3]);
                }
            }
        }

        // ---- End of section t: hand the tensor pipe over; group 1 refills the stages both groups
        // read. ----
        if (group == 0) {
            arrive(kB1, kThreads);
        } else if (t < key_blocks) {
            arrive(kB0, kThreads);
        }
        if (group == 1) {
            // Every group 1 warp is done with K(t) and V(t - 1); group 0 finished section t before
            // group 1 began it (B1).
            sync(kLoad, kGroupThreads);
            if (t + 2 < key_blocks) { load_tile(t + 2, true); }
            if (t + 1 < key_blocks) { load_tile(t + 1, false); }
        }

        // ---- Softmax t. ----
        if (t < key_blocks) {
            if (t > 0) { update_row_sum(); }
            const int k0               = t * Bc;
            const int qabs0            = qk_row0 < tile_rows ? base_pos + q0 + qk_row0 : -1;
            const int qabs1            = qk_row1 < tile_rows ? base_pos + q0 + qk_row1 : -1;
            const bool full_score_tile = q0 + Br <= tokens && k0 + Bc - 1 <= base_pos + q0;
            float bm0                  = -CUDART_INF_F;
            float bm1                  = -CUDART_INF_F;
#pragma unroll
            for (int j = 0; j < QuarterNt; ++j) {
                const int nt   = quarter * QuarterNt + j;
                const int key0 = k0 + nt * 8 + 2 * lid;
                const int key1 = key0 + 1;
                if (!full_score_tile) {
                    score[j][0] = key0 <= qabs0 ? score[j][0] : -CUDART_INF_F;
                    score[j][1] = key1 <= qabs0 ? score[j][1] : -CUDART_INF_F;
                    score[j][2] = key0 <= qabs1 ? score[j][2] : -CUDART_INF_F;
                    score[j][3] = key1 <= qabs1 ? score[j][3] : -CUDART_INF_F;
                }
                bm0 = fmaxf(bm0, fmaxf(score[j][0], score[j][1]));
                bm1 = fmaxf(bm1, fmaxf(score[j][2], score[j][3]));
            }
            bm0 = warp_max<4>(bm0, FullMask);
            bm1 = warp_max<4>(bm1, FullMask);
            if (lid == 0) {
                m_part_s[quarter * Br + qk_row0] = bm0;
                m_part_s[quarter * Br + qk_row1] = bm1;
            }
            // Every warp of the group has written its maxima and finished section t (its reads of
            // P(t - 1), alpha and the pair sums), so P, alpha and the pair sums may be rewritten
            // below.
            sync(kGroup0 + group, kGroupThreads);
            bm0 = m_part_s[qk_row0];
            bm1 = m_part_s[qk_row1];
#pragma unroll
            for (int qq = 1; qq < 4; ++qq) {
                bm0 = fmaxf(bm0, m_part_s[qq * Br + qk_row0]);
                bm1 = fmaxf(bm1, m_part_s[qq * Br + qk_row1]);
            }

            const float nm0        = fmaxf(running_m0, bm0);
            const float nm1        = fmaxf(running_m1, bm1);
            const float nm0_scaled = nm0 * scale_l2;
            const float nm1_scaled = nm1 * scale_l2;
            qk_alpha0              = running_m0 == -CUDART_INF_F
                                         ? 0.0f
                                         : causal_exp_scaled(running_m0, nm0_scaled, scale_l2);
            qk_alpha1              = running_m1 == -CUDART_INF_F
                                         ? 0.0f
                                         : causal_exp_scaled(running_m1, nm1_scaled, scale_l2);
            bl0                    = 0.0f;
            bl1                    = 0.0f;
#pragma unroll
            for (int j = 0; j < QuarterNt; ++j) {
                const int nt    = quarter * QuarterNt + j;
                const int col0  = nt * 8 + 2 * lid;
                const int col1  = col0 + 1;
                const float p00 = score[j][0] > -CUDART_INF_F
                                      ? causal_exp_scaled(score[j][0], nm0_scaled, scale_l2)
                                      : 0.0f;
                const float p01 = score[j][1] > -CUDART_INF_F
                                      ? causal_exp_scaled(score[j][1], nm0_scaled, scale_l2)
                                      : 0.0f;
                const float p10 = score[j][2] > -CUDART_INF_F
                                      ? causal_exp_scaled(score[j][2], nm1_scaled, scale_l2)
                                      : 0.0f;
                const float p11 = score[j][3] > -CUDART_INF_F
                                      ? causal_exp_scaled(score[j][3], nm1_scaled, scale_l2)
                                      : 0.0f;
                const float s0  = p00 + p01;
                const float s1  = p10 + p11;
                if (quarter == 0) {
                    bl0 += s0;
                    bl1 += s1;
                } else {
                    const int slot                            = (quarter - 1) * QuarterNt + j;
                    s_part_s[(slot * Br + qk_row0) * 4 + lid] = s0;
                    s_part_s[(slot * Br + qk_row1) * 4 + lid] = s1;
                }
                p_s[qk_row0 * Bc + causal_probability_swizzle<Bc>(qk_row0, col0)] =
                    __float2half_rn(p00);
                p_s[qk_row0 * Bc + causal_probability_swizzle<Bc>(qk_row0, col1)] =
                    __float2half_rn(p01);
                p_s[qk_row1 * Bc + causal_probability_swizzle<Bc>(qk_row1, col0)] =
                    __float2half_rn(p10);
                p_s[qk_row1 * Bc + causal_probability_swizzle<Bc>(qk_row1, col1)] =
                    __float2half_rn(p11);
            }
            running_m0 = nm0;
            running_m1 = nm1;
            if (quarter == 0 && lid == 0) {
                alpha_s[qk_row0] = qk_alpha0;
                alpha_s[qk_row1] = qk_alpha1;
            }
        }
    }
    // The pair sums of the last tile were written before section key_blocks, which began with a
    // group-wide barrier.
    update_row_sum();
    if (quarter == 0 && lid == 0) {
        final_l_s[qk_row0] = running_l0;
        final_l_s[qk_row1] = running_l1;
    }
    sync(kGroup0 + group, kGroupThreads);

    // n-tile 2 b + e of row tile r: column j is dimension 32 gw + 16 b + 2 j + e, so a thread's
    // c0/c1 of the even and odd tiles are dimensions 4 lid + {0, 1} and 4 lid + {2, 3} of the
    // block.
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const int row0     = pv_row_base + r * 16 + gid;
        const int row1     = row0 + 8;
        const float inv_l0 = final_l_s[row0] > 0.0f ? __frcp_rn(final_l_s[row0]) : 0.0f;
        const float inv_l1 = final_l_s[row1] > 0.0f ? __frcp_rn(final_l_s[row1]) : 0.0f;
#pragma unroll
        for (int b = 0; b < 2; ++b) {
            const int d       = gw * 32 + b * 16 + 4 * lid;
            const float* even = acc[r][2 * b];
            const float* odd  = acc[r][2 * b + 1];
            if (row0 < tile_rows) {
                causal_store_output_pair<Geometry>(out, q_head, d, q0 + row0, even[0] * inv_l0,
                                                   odd[0] * inv_l0);
                causal_store_output_pair<Geometry>(out, q_head, d + 2, q0 + row0, even[1] * inv_l0,
                                                   odd[1] * inv_l0);
            }
            if (row1 < tile_rows) {
                causal_store_output_pair<Geometry>(out, q_head, d, q0 + row1, even[2] * inv_l1,
                                                   odd[2] * inv_l1);
                causal_store_output_pair<Geometry>(out, q_head, d + 2, q0 + row1, even[3] * inv_l1,
                                                   odd[3] * inv_l1);
            }
        }
    }
    causal_zero_rows<Geometry>(out, q_head, tokens, min(q0 + Br, width), tid, kThreads);
}

} // namespace ninfer::ops::detail
