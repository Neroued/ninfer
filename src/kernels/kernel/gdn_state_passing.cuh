#pragma once

#include "kernels/common/mma.cuh"
#include "kernels/kernel/gdn_chunked_common.cuh"

#include <cmath>
#include <cstdio>
#include <utility>

// Stage 3: chunk-sequential state passing.
//
// Math + I/O layouts: see gdn chunked state_passing_config.
// Block / smem total: 8 warps (256 t) at S=128 with __launch_bounds__(256, 2);
//                     ~33 KB smem (W_half + UVD alias + k_half + snap + g).


#include <cstdio>

namespace qus::kernels::detail::gdn_state_passing {

namespace {

using gdn_chunked::BT;
using gdn_chunked::MMA_M;
using gdn_chunked::MMA_N;
using gdn_chunked::MMA_K;
using gdn_chunked::bh_decode_t;
using gdn_chunked::zero_frag;
using qus::kernels::SmemTile;
using qus::kernels::mma_tf32;
using qus::kernels::mma_tf32_bits;
using qus::kernels::ldmatrix_x4;
using qus::kernels::ldmatrix_x2;
using qus::kernels::smem_addr;
using qus::kernels::exp2_approx;
using qus::kernels::RCP_LN2_F;

static_assert(gdn_chunked::kChunkSize == 64, "stage_state_passing: kChunkSize must be 64");

// ---------------------------------------------------------------------------
// Per-S kernel dimensions for the d-strip + BT-split design.
//
//   N_STRIP_PER_BLOCK  = min(S, 16)         d-cols owned by a block
//   D_STRIPS           = S / N_STRIP_PER_BLOCK
//   DT_TILES_PER_BLOCK = N_STRIP_PER_BLOCK / MMA_N        (1 or 2)
//   BT_SPLITS          = chosen so total warps <= 8
//   N_WARPS            = DT_TILES_PER_BLOCK * BT_SPLITS
//   THREADS            = N_WARPS * 32
//   BT_PER_WARP        = BT / BT_SPLITS                   (matmul1 M-rows)
//   S_PER_WARP         = S  / BT_SPLITS                   (matmul2 M-rows)
//   M_TILES_MM1_PW     = BT_PER_WARP / MMA_M
//   M_TILES_H_PW       = S_PER_WARP  / MMA_M
//   M_TILES_H_GLOB     = S / MMA_M
//   K_TILES_MM2_PER_LOAD = (BT / K_HALVES) / MMA_K
//   W_HALVES           = how many K-axis halves W is loaded in (1 -> full)
//   K_HALVES           = how many BT-axis halves k is loaded in (1 -> full)
//
// S=128 (M1, mainloop-rewrite): both halves merged into single full loads,
// so Phase B has no inner W-half loop and Phase E is one mma2 sweep over
// `BT/MMA_K` tiles. Smem grows ~32 KB (W) + 16 KB (k) but stays under the
// 99 KB / SM cap (one block / SM was already the smem-bound today).
// ---------------------------------------------------------------------------
template <int S>
struct kernel_dims;

template <>
struct kernel_dims<128> {
    // EXP-HALVES2 was tried. Predictions on MIO Throttle (5.66 -> 2.98,
    // -47%) and Long Scoreboard (1.56 -> 4.52, +190%) both validated,
    // confirming FLA's diagnosis was directionally correct. BUT wall
    // regressed 362 -> 408us (+46us) because our existing W_HALVES=2 /
    // K_HALVES=2 code path is a "split-smem-to-save-space" design
    // (mid-Phase-B refill with synchronous wait_all), NOT FLA's
    // "double-smem-to-deepen-pipeline" design (two half buffers
    // co-resident, two independent commit groups, no mid-iter refill).
    //
    // To truly match FLA's pipeline depth we need a structural
    // rewrite: W_half1_smem + W_half2_smem both live, both prefetched
    // up-front with independent commit groups, mm1 reads from the
    // appropriate half without ever refilling. Same for k. That is
    // ~1-2 days of work; track separately.
    //
    // Side observation: HALVES=2 also surfaced 3.3M l1tex bank
    // conflicts at STRIDE=64 (was 0 at STRIDE=128) -- swizzle for
    // SmemTile<64> needs validation against the actual mm1 lane
    // mapping at S=128.
    static constexpr int N_STRIP_PER_BLOCK  = 32;
    static constexpr int D_STRIPS           = 4;
    static constexpr int DT_TILES_PER_BLOCK = 4;
    static constexpr int BT_SPLITS          = 2;
    static constexpr int N_WARPS            = DT_TILES_PER_BLOCK * BT_SPLITS; // 8
    static constexpr int THREADS            = N_WARPS * qus::kernels::kWarpSize;
    static constexpr int W_HALVES           = 1;
    static constexpr int K_HALVES           = 1;
};

template <>
struct kernel_dims<64> {
    static constexpr int N_STRIP_PER_BLOCK  = 16;
    static constexpr int D_STRIPS           = 4;
    static constexpr int DT_TILES_PER_BLOCK = 2;
    static constexpr int BT_SPLITS          = 4;
    static constexpr int N_WARPS            = DT_TILES_PER_BLOCK * BT_SPLITS;
    static constexpr int THREADS            = N_WARPS * qus::kernels::kWarpSize;
    static constexpr int W_HALVES           = 2;
    static constexpr int K_HALVES           = 2;
};

template <>
struct kernel_dims<32> {
    static constexpr int N_STRIP_PER_BLOCK  = 16;
    static constexpr int D_STRIPS           = 2;
    static constexpr int DT_TILES_PER_BLOCK = 2;
    static constexpr int BT_SPLITS          = 2;
    static constexpr int N_WARPS            = DT_TILES_PER_BLOCK * BT_SPLITS;
    static constexpr int THREADS            = N_WARPS * qus::kernels::kWarpSize;
    static constexpr int W_HALVES           = 2;
    static constexpr int K_HALVES           = 2;
};

template <>
struct kernel_dims<16> {
    static constexpr int N_STRIP_PER_BLOCK  = 16;
    static constexpr int D_STRIPS           = 1;
    static constexpr int DT_TILES_PER_BLOCK = 2;
    static constexpr int BT_SPLITS          = 1;
    static constexpr int N_WARPS            = DT_TILES_PER_BLOCK * BT_SPLITS;
    static constexpr int THREADS            = N_WARPS * qus::kernels::kWarpSize;
    // S=16 W half stride would be 8 -- SmemTile requires >= 16. Load full W.
    static constexpr int W_HALVES = 1;
    static constexpr int K_HALVES = 2;
};

// Smem floats per block, derived purely from kernel_dims<S>.
//
// `_LOAD_` slot sizes are per shared-memory load stage. W/U/k receive bf16
// global data converted to float shared memory. With HALVES=1 a single load
// brings the full slab; with HALVES=2 each load brings a half and Phase E
// performs a second synchronous k load.
template <int S>
struct smem_layout {
    using D                             = kernel_dims<S>;
    static constexpr int W_STRIDE       = S / D::W_HALVES;
    static constexpr int W_LOAD_FLT     = BT * W_STRIDE;
    static constexpr int UVD_FLT        = BT * D::N_STRIP_PER_BLOCK; // U-vd alias
    static constexpr int K_LOAD_ROWS    = BT / D::K_HALVES;
    static constexpr int K_LOAD_FLT     = K_LOAD_ROWS * S;
    static constexpr int M_TILES_H_PW   = (S / D::BT_SPLITS) / MMA_M;
    static constexpr int SNAP_K_ROWS    = MMA_M * M_TILES_H_PW;
    static constexpr int SNAP_FLT       = SNAP_K_ROWS * D::N_STRIP_PER_BLOCK;
    static constexpr int M_TILES_H_GLOB = S / MMA_M;
    static constexpr int N_SNAP_ITERS   = M_TILES_H_GLOB / M_TILES_H_PW;
    // One buffer per sit so all warps can scatter their owned h_frag once at
    // chunk start (Phase A) and then read from any sit's buffer in the unified
    // matmul1 / coop_write paths without per-sit re-scatter+sync.
    static constexpr int N_SNAP_BUF = N_SNAP_ITERS;
    static constexpr int SMEM_FLOATS =
        W_LOAD_FLT + UVD_FLT + K_LOAD_FLT + SNAP_FLT * N_SNAP_BUF + BT;
};

// ---------------------------------------------------------------------------
// SnapView<STRIDE>: custom swizzle for snap_smem (transposed h[d][k] layout).
// Different from SmemTile<32>'s default swizzle to keep the 4-d-row scatter
// conflict-free. Do NOT merge with qus::kernels::SmemTile.
//
// STRIDE=64 case (S=128 wide-block design): with row*64/4=row*16 mod 32, even
// rows start at bank 0, odd rows at bank 16. For the 4-d-row scatter (d takes
// 4 same-parity values per warp, e.g. {0,2,4,6} or {16,18,20,22}), all 4 rows
// land at the same bank without swizzle. The `(row & 7) << 2` swizzle splits
// d ∈ {0,2,4,6} into swz {0,8,16,24} → banks {0,8,16,24} (distinct). For odd
// d ∈ {1,3,5,7}, base bank=16 + swz {4,12,20,28} mod 32 = {20,28,4,12}, also
// distinct. The 8-d-row MM1 B-read (d ∈ {warp_d_local..warp_d_local+7}) maps
// to 8 distinct banks too. For STRIDE=32 the same formula still gives
// {0,8,16,24}.
// ---------------------------------------------------------------------------
template <int STRIDE>
struct SnapView {
    float* __restrict__ base;
    static_assert(STRIDE == 16 || STRIDE == 32 || STRIDE == 64,
                  "SnapView: only STRIDE in {16, 32, 64} supported");

    __device__ __forceinline__ int swz_xor(int row) const {
        if constexpr (STRIDE == 64 || STRIDE == 32) {
            return (row & 7) << 2; // {0, 4, 8, ..., 28}
        } else {
            return ((row >> 1) & 3) << 2; // fall back to default for STRIDE=16
        }
    }

    __device__ __forceinline__ float& at(int row, int col) const {
        return base[row * STRIDE + (col ^ swz_xor(row))];
    }

    __device__ __forceinline__ float4& vec4_at(int row, int col) const {
        return *reinterpret_cast<float4*>(&base[row * STRIDE + (col ^ swz_xor(row))]);
    }
};

// __launch_bounds__: 256 threads + min_blocks=1 -> ~250 reg cap. With R2
// (ldmatrix.x4 for mm1 A) the compiler needs to preload multiple A frags
// concurrently to hide ldmatrix's data-ready latency (~80 cycles vs ~24 for
// LDS.32). The 128-reg cap from LB(2) blocked preload pipelining and made
// ldmatrix net-negative (R2 wall +31us at LB(2)); LB(1) lets ptxas use enough
// regs for in-flight overlap. Occupancy already only 17% in either mode, so
// dropping min_blocks costs nothing.
template <int S>
__launch_bounds__(kernel_dims<S>::THREADS, 1) __global__
    void state_passing_gdn_kernel(const __nv_bfloat16* __restrict__ W_in,
                                  const __nv_bfloat16* __restrict__ U_in,
                                  const __nv_bfloat16* __restrict__ k_in,
                                  const float* __restrict__ g_cumsum, const float* state_in,
                                  __nv_bfloat16* __restrict__ v_new,
                                  __nv_bfloat16* __restrict__ h_chunk, float* state_out,
                                  int64_t T, int64_t H_v,
                                  qus::kernels::head_map qk_map,
                                  // Token-axis stride for k (in floats).
                                  // Caller passes the materialised value
                                  // (launcher handles 0 -> H_qk * S).
                                  int64_t k_stride_t, int NT) {
    using D                         = kernel_dims<S>;
    using L                         = smem_layout<S>;
    constexpr int N_STRIP_PER_BLOCK = D::N_STRIP_PER_BLOCK;
    constexpr int D_STRIPS          = D::D_STRIPS;
    constexpr int BT_SPLITS         = D::BT_SPLITS;
    constexpr int THREADS_K         = D::THREADS;
    constexpr int W_HALVES          = D::W_HALVES;
    constexpr int K_HALVES          = D::K_HALVES;
    constexpr int W_STRIDE          = L::W_STRIDE;

    constexpr int BT_PER_WARP          = BT / BT_SPLITS;
    constexpr int S_PER_WARP           = S / BT_SPLITS;
    constexpr int M_TILES_MM1_PW       = BT_PER_WARP / MMA_M;
    constexpr int M_TILES_H_PW         = S_PER_WARP / MMA_M;
    constexpr int M_TILES_H_GLOB       = S / MMA_M;
    constexpr int K_TILES_MM2_PER_LOAD = (BT / K_HALVES) / MMA_K;

    static_assert(M_TILES_H_PW >= 1, "S_PER_WARP must yield >= 1 M-tile per warp");
    static_assert(M_TILES_MM1_PW >= 1, "BT_PER_WARP must yield >= 1 M-tile per warp");
    static_assert(M_TILES_H_GLOB == BT_SPLITS * M_TILES_H_PW,
                  "BT_SPLITS partition of S must be exact");
    static_assert(W_STRIDE >= 16, "SmemTile<W_STRIDE> requires stride >= 16");

    constexpr int SNAP_K_ROWS           = L::SNAP_K_ROWS;
    constexpr int N_SNAP_ITERS          = L::N_SNAP_ITERS;
    constexpr int N_SNAP_HALF           = N_SNAP_ITERS / W_HALVES;
    constexpr int K_TILES_PER_SNAP_ITER = SNAP_K_ROWS / MMA_K;

    static_assert(N_SNAP_ITERS == BT_SPLITS, "design assumes one snap iter per s_idx");
    static_assert(N_SNAP_HALF >= 1, "W_HALVES must divide N_SNAP_ITERS");

    constexpr int W_LOAD_FLT  = L::W_LOAD_FLT;
    constexpr int UVD_FLT     = L::UVD_FLT;
    constexpr int K_LOAD_ROWS = L::K_LOAD_ROWS;
    constexpr int K_LOAD_FLT  = L::K_LOAD_FLT;
    constexpr int SNAP_FLT    = L::SNAP_FLT;
    constexpr int N_SNAP_BUF  = L::N_SNAP_BUF;

    // Smem partition. U and vd alias (`uvd_smem`) -- disjoint phases. snap is
    // per-sit (one buffer per sit) so the chunk-start unified scatter feeds
    // every sit's read without re-scatter.
    extern __shared__ float smem[];
    float* const W_smem    = smem;                              // W_LOAD_FLT
    float* const uvd_smem  = W_smem + W_LOAD_FLT;               // UVD_FLT (U & vd alias)
    float* const k_smem    = uvd_smem + UVD_FLT;                // K_LOAD_FLT
    float* const snap_smem = k_smem + K_LOAD_FLT;               // SNAP_FLT * N_SNAP_BUF
    float* const g_smem    = snap_smem + SNAP_FLT * N_SNAP_BUF; // BT

    SmemTile<W_STRIDE> W_view{W_smem};
    SmemTile<N_STRIP_PER_BLOCK> vd_view{uvd_smem};
    SmemTile<S> k_view{k_smem};
    SmemTile<N_STRIP_PER_BLOCK> U_view{uvd_smem};
    // One SnapView per sit so each owning warp scatters into a unique buffer
    // (Phase 2.a unified-scatter design). Initialised inline via a lambda so
    // the array is sized on N_SNAP_BUF (= N_SNAP_ITERS) rather than hard-coded.
    SnapView<SNAP_K_ROWS> snap_views[N_SNAP_BUF];
#pragma unroll
    for (int b_ = 0; b_ < N_SNAP_BUF; ++b_) {
        snap_views[b_] = SnapView<SNAP_K_ROWS>{snap_smem + b_ * SNAP_FLT};
    }

    // Block / lane indexing.
    //   grid.x = bhd in [0, B*H_v*D_STRIPS).
    //   warp = dt_idx * BT_SPLITS + s_idx.
    const int tid    = threadIdx.x;
    const auto lanes = qus::kernels::mma_lane_t::decode(tid);
    const int warp   = lanes.warp;
    const int lane_g = lanes.lane_g;
    const int lane_t = lanes.lane_t;

    const int s_idx        = warp % BT_SPLITS;
    const int dt_idx       = warp / BT_SPLITS;
    const int warp_d_local = dt_idx * MMA_N;

    const int bhd           = blockIdx.x;
    const int chain_idx     = bhd / D_STRIPS;
    const int strip_idx     = bhd - chain_idx * D_STRIPS;
    const auto bh           = bh_decode_t::of(chain_idx, (int)H_v);
    const int b             = bh.b;
    const int h_v           = bh.h_v;
    const int d_off         = strip_idx * N_STRIP_PER_BLOCK;
    const int warp_d_global = d_off + warp_d_local;

    // === Phase 0: load state_in (AR-transposed) -> per-warp h_frag ===
    float h_frag[M_TILES_H_PW][4];
    {
        const int64_t st_base = ((int64_t)b * H_v + h_v) * S * S;
        const int row_off     = s_idx * S_PER_WARP;
#pragma unroll
        for (int m = 0; m < M_TILES_H_PW; ++m) {
            const int row_g0 = row_off + m * MMA_M + lane_g;
            const int row_g1 = row_g0 + 8;
            const int col_d0 = warp_d_global + 2 * lane_t;
            const int col_d1 = col_d0 + 1;
            h_frag[m][0] = load_ldg<float>(state_in + st_base + (int64_t)col_d0 * S + row_g0);
            h_frag[m][1] = load_ldg<float>(state_in + st_base + (int64_t)col_d1 * S + row_g0);
            h_frag[m][2] = load_ldg<float>(state_in + st_base + (int64_t)col_d0 * S + row_g1);
            h_frag[m][3] = load_ldg<float>(state_in + st_base + (int64_t)col_d1 * S + row_g1);
        }
    }

    // Cross-chunk prefetch. Chunk 0's W_half_0 + U + k_half_0 are loaded
    // before the loop; subsequent chunks load W at end of Phase B and U/k at
    // end of Phase E.
    //
    // R7.1: All per-chunk gmem bases are precomputed as `*_block_base`
    // (chunk-0 value) + `*_chunk_stride` (delta), and advanced ADDITIVELY at
    // end of each iter -- saves 4 IMAD.WIDE per chunk that the old lambdas
    // ran for current/next W and current/next k bases.
    // R7.3: g_cumsum index splits into per-thread invariant base
    // (`g_thread_base`) + per-chunk delta (`g_cs_offset`), advanced by a
    // single int64 add per chunk per thread.
    const int64_t W_stride        = (int64_t)H_v * S;
    const int64_t k_stride        = k_stride_t;
    const int64_t W_chunk_stride  = (int64_t)BT * W_stride; // = BT*H_v*S
    const int64_t k_chunk_stride  = (int64_t)BT * k_stride; // = BT*k_stride_t
    const int64_t hc_chunk_stride = (int64_t)H_v * S * S;
    const int64_t vn_stride       = W_stride; // = H_v*S
    const int64_t vn_chunk_stride = (int64_t)BT * vn_stride;
    const int64_t g_chunk_step    = (int64_t)BT * H_v;

    const int64_t W_block_base  = (int64_t)b * T * W_stride + (int64_t)h_v * S;
    const int64_t k_block_base  = (int64_t)b * T * k_stride + (int64_t)qk_map.qk_head(h_v) * S;
    const int64_t hc_block_base = ((int64_t)b * NT * H_v + h_v) * S * S;
    const int64_t vn_block_base = (int64_t)b * T * vn_stride + (int64_t)h_v * S;
    const int64_t g_block_base  = (int64_t)b * T * H_v + h_v;
    const int64_t g_thread_base = g_block_base + (int64_t)tid * H_v;

    // W/U/k are bf16 in global workspace/boundary storage and are converted
    // into float shared memory before the math phases consume them.
    {
        const int64_t ce0_64 = BT; // chunk_0 end = BT
        const int cl0        = (ce0_64 < T) ? (int)ce0_64 : (int)T;
        qus::kernels::issue_load_bf16_to_float_vec4<BT, W_STRIDE, THREADS_K>(
            W_view, W_in + W_block_base, W_stride, cl0, tid);
        qus::kernels::issue_load_bf16_to_float_vec4<BT, N_STRIP_PER_BLOCK, THREADS_K>(
            U_view, U_in + W_block_base + d_off, W_stride, cl0, tid);
        const int cl_k0 = (cl0 < K_LOAD_ROWS) ? cl0 : K_LOAD_ROWS;
        qus::kernels::issue_load_bf16_to_float_vec4<K_LOAD_ROWS, S, THREADS_K>(
            k_view, k_in + k_block_base, k_stride, cl_k0, tid);
    }

    // === Main chunk loop ===
    //
    // R7.1: cs / W_base / k_base / hc_base / vn_base are LOOP-CARRIED
    // additive accumulators. R7.2: next-chunk bounds (cs_next / cl_next /
    // W_base_next / k_base_next / cl_k_next) are computed ONCE per iter and
    // reused by both Phase B and Phase E prefetch sites.
    int64_t cs          = 0;
    int64_t W_base      = W_block_base;
    int64_t k_base      = k_block_base;
    int64_t hc_base     = hc_block_base;
    int64_t vn_base     = vn_block_base;
    int64_t g_cs_offset = 0;
    for (int chunk = 0; chunk < NT; ++chunk) {
        const int64_t ce_64 = cs + BT;
        const int64_t ce    = (ce_64 < T) ? ce_64 : T;
        const int cl        = (int)(ce - cs);

        // Hoisted next-chunk metadata (used by Phase B + Phase E loads).
        // Variables are unconditionally computed; the load sites still gate
        // global memory traffic on `chunk + 1 < NT`.
        const int64_t cs_next     = cs + BT;
        const int64_t ce_next_64  = cs_next + BT;
        const int64_t ce_next     = (ce_next_64 < T) ? ce_next_64 : T;
        const int cl_next         = (int)(ce_next - cs_next);
        const int cl_k_next       = (cl_next < K_LOAD_ROWS) ? cl_next : K_LOAD_ROWS;
        const int64_t W_base_next = W_base + W_chunk_stride;
        const int64_t k_base_next = k_base + k_chunk_stride;

        // === Phase A: drain early group (W + U) + scatter h_frag to snap ===
        //
        // Phase 2.a unified-scatter: every warp scatters its owned h_frag into
        // snap_views[s_idx] BEFORE the Phase A drain sync. The single sync
        // below covers W/U/g_smem AND snap visibility, so Phase B no longer
        // needs per-sit scatter+sync (saves 2 syncs/chunk on the BT_SPLITS=2
        // S=128 path, 4 syncs/chunk on the BT_SPLITS=4 S=64 path).
        if (tid < BT) {
            float val = 0.0f;
            if (tid < cl) { val = g_cumsum[g_thread_base + g_cs_offset]; }
            g_smem[tid] = val;
        }

        {
            SnapView<SNAP_K_ROWS> snap = snap_views[s_idx];
#pragma unroll
            for (int m = 0; m < M_TILES_H_PW; ++m) {
                const int k_g0    = m * MMA_M + lane_g;
                const int k_g1    = k_g0 + 8;
                const int d0      = warp_d_local + 2 * lane_t;
                const int d1      = d0 + 1;
                snap.at(d0, k_g0) = h_frag[m][0];
                snap.at(d1, k_g0) = h_frag[m][1];
                snap.at(d0, k_g1) = h_frag[m][2];
                snap.at(d1, k_g1) = h_frag[m][3];
            }
        }

        __syncthreads(); // gates W+U+g_smem STS and scatter visibility

        // === Phase B: per-sit coop_write + matmul1 (no per-sit scatter sync) ===
        //
        // The unified scatter in Phase A populated snap_views[s_idx] for all
        // s_idx in one shot, so each sit just consumes snap[sit] without
        // re-scattering or syncing. We retain the per-sit interleave between
        // coop_write (LDS+STG to gmem) and mma (LDS) because that was the
        // crucial latency-hiding pattern in the per-sit baseline -- moving
        // all coop_writes ahead of all mmas costs ~15 us on L=4096.
        float vnew_frag[M_TILES_MM1_PW][4];
        zero_frag(vnew_frag);

#pragma unroll 1
        for (int half = 0; half < W_HALVES; ++half) {
            if (half >= 1) {
                __syncthreads(); // gates prev half's mma reads of W_smem
                qus::kernels::issue_load_bf16_to_float_vec4<BT, W_STRIDE, THREADS_K>(
                    W_view, W_in + W_base + (int64_t)(half * W_STRIDE), W_stride, cl, tid);
                __syncthreads();
            }

            const int sit_lo = half * N_SNAP_HALF;
            const int sit_hi = sit_lo + N_SNAP_HALF;

#pragma unroll
            for (int sit = sit_lo; sit < sit_hi; ++sit) {
                const int k_row_off        = sit * SNAP_K_ROWS;
                SnapView<SNAP_K_ROWS> snap = snap_views[sit];

                // Coop float4 gmem write of h_chunk for this snap block.
                // No sync above: snap was populated by the Phase A
                // unified-scatter, and snap[sit] is read-only from now on.
                {
                    constexpr int K_VEC_PER_D = SNAP_K_ROWS / 4;
                    constexpr int N_VEC_SNAP  = SNAP_FLT / 4;
#pragma unroll
                    for (int v = tid; v < N_VEC_SNAP; v += THREADS_K) {
                        const int d_local  = v / K_VEC_PER_D;
                        const int kvec     = v - d_local * K_VEC_PER_D;
                        const int k_off    = kvec * 4;
                        float4 val         = snap.vec4_at(d_local, k_off);
                        const int d_global = d_off + d_local;
                        __nv_bfloat16* out =
                            &h_chunk[hc_base + (int64_t)d_global * S + k_row_off + k_off];
                        store_vec(out, __floats2bfloat162_rn(val.x, val.y));
                        store_vec(out + 2, __floats2bfloat162_rn(val.z, val.w));
                    }
                }

                // matmul1 inner: K_TILES_PER_SNAP_ITER mma K-tiles.
                //
                // A/B operand loads use ldmatrix.x4 / x2 .b16 to fold what was
                // 4+2 LDS.32 per mma into 1+1 issues going through the smem
                // crossbar. Per-lane address ownership (PTX docs Fig.70):
                //   A x4: lanes 0..7 source rows 0..7  of A-frag, col 0..3
                //         lanes 8..15        rows 0..7,           col 4..7
                //         lanes 16..23       rows 8..15,          col 0..3
                //         lanes 24..31       rows 8..15,          col 4..7
                //   B x2: lanes 0..7 source rows 0..7 of B-frag (mma-N row, =
                //         snap row d=warp_d_local..+7), col 0..3 of B-frag
                //         (mma-K col, = snap col snap_k..+3)
                //         lanes 8..15 source rows 0..7, col 4..7.
                // snap stores B as smem[N, K] which is exactly the no-trans
                // ldmatrix layout for tf32 B (b0 = snap[t/4, t%4]). W_view
                // stores A as smem[M, K] (no-trans tf32 A: a0 = W_view[t/4,
                // t%4]).
                // R3-minimal: for sit == s_idx (own-sit) the mm1 B operand
                // lives in THIS warp's h_frag. Use a per-lane register-shuffle
                // instead of ldmatrix.x2 from snap — skips MIO pipe entirely
                // for half of the B reads. snap STS is unchanged (other warps'
                // cross-sit reads still need it). Cross-sit (sit != s_idx)
                // stays on the ldmatrix.x2 path.
                //
                // Per-thread B-frag wants:
                //   b0 = h[k=snap_k+lane_t, d=warp_d_local+lane_g]
                //   b1 = h[k=snap_k+lane_t+4, d=warp_d_local+lane_g]
                // Our h_frag[m_h][i_h] for lane (lane_g_o, lane_t_o) stores:
                //   h[k=lane_g_o + m_h*16 + ((i_h>>1)&1)*8, d=warp_d_local + 2*lane_t_o + (i_h&1)]
                // Solving for own-sit:
                //   m_h         = kt >> 1
                //   i_h         = (kt&1)*2 + (lane_g & 1)
                //   src_b0_lane = (lane_t << 2) | (lane_g >> 1)
                //   src_b1_lane = src_b0_lane + 16
                // i_h picks {0,1} (kt even) or {2,3} (kt odd) of h_frag[m_h]
                // based on lane_g parity — selected branch-free via FSEL.
                // R2: matmul1 inner. A operand via ldmatrix.x4 (16x8 tf32 from
                // W_view), B operand via ldmatrix.x2 (8x8 tf32 from snap). Both
                // use the no-trans .b16 layout because W_view stores A as
                // smem[M, K] and snap stores B as smem[N, K] (mma B-frag's
                // per-lane b0 = source[t/4, t%4] matches the no-trans output).
                //
                // R3-minimal (own-sit shuffle) was tried: replace own-sit
                // ldmatrix.x2 with 4 __shfl_sync + 2 SEL per kt. Wall-time
                // 364us -> 392us (+28us, -7.7%). The shuffle EU is also
                // 1-issue/cycle, so 4 shfls cost more than 1 LDSM.x2 going
                // through MIO. Reverted; see analysis/root_cause.md.
                const int lane_in_8   = lanes.lane & 7;
                const int which_horiz = (lanes.lane >> 3) & 1;
                const int which_vert  = (lanes.lane >> 4) & 1;
#pragma unroll
                for (int kt = 0; kt < K_TILES_PER_SNAP_ITER; ++kt) {
                    const int W_k_local = (k_row_off + kt * MMA_K) - half * W_STRIDE;
                    const int snap_k    = kt * MMA_K;

                    const int b_row       = warp_d_local + lane_in_8;
                    const int b_col       = snap_k + which_horiz * 4;
                    const unsigned b_addr = smem_addr(&snap.at(b_row, b_col));
                    unsigned ub0, ub1;
                    ldmatrix_x2(ub0, ub1, b_addr);

#pragma unroll
                    for (int m_mm1 = 0; m_mm1 < M_TILES_MM1_PW; ++m_mm1) {
                        const int row_base    = s_idx * BT_PER_WARP + m_mm1 * MMA_M;
                        const int a_row       = row_base + which_vert * 8 + lane_in_8;
                        const int a_col       = W_k_local + which_horiz * 4;
                        const unsigned a_addr = smem_addr(&W_view.at(a_row, a_col));
                        unsigned ua0, ua1, ua2, ua3;
                        // GDN's TF32 fragment consumes ldmatrix registers in 0,2,1,3 order.
                        ldmatrix_x4(ua0, ua2, ua1, ua3, a_addr);

                        mma_tf32_bits(vnew_frag[m_mm1][0], vnew_frag[m_mm1][1],
                                             vnew_frag[m_mm1][2], vnew_frag[m_mm1][3], ua0, ua1,
                                             ua2, ua3, ub0, ub1);
                    }
                }
            }
        }

        // Cross-chunk W load right after Phase B finishes consuming
        // W_smem. The __syncthreads gates the half=1
        // sit's mma reads of W_smem (matmul1 inner LDS.32 in slow warps)
        // BEFORE this thread overwrites W_smem with the next chunk's
        // W_half_0. Without it, racecheck reports a write vs
        // f32_to_tf32-read race on W_smem at sm_120, which
        // surfaces as ~25% flaky v_new at S=128/L=256 (the chunk-end barrier
        // before next iter's Phase A only fences the *future* read of the
        // arriving W, not the *prior* read of the outgoing W).
        if (chunk + 1 < NT) {
            __syncthreads();
            qus::kernels::issue_load_bf16_to_float_vec4<BT, W_STRIDE, THREADS_K>(
                W_view, W_in + W_base_next, W_stride, cl_next, tid);
        }

// === Phase C: subtract U from U_smem (no global wait, U landed in A) ===
#pragma unroll
        for (int m_mm1 = 0; m_mm1 < M_TILES_MM1_PW; ++m_mm1) {
            const int row_g0    = s_idx * BT_PER_WARP + m_mm1 * MMA_M + lane_g;
            const int row_g1    = row_g0 + 8;
            const int col_d0    = warp_d_local + 2 * lane_t;
            const float2 u_top  = load_vec<float2>(&U_view.at(row_g0, col_d0));
            const float2 u_bot  = load_vec<float2>(&U_view.at(row_g1, col_d0));
            vnew_frag[m_mm1][0] = u_top.x - vnew_frag[m_mm1][0];
            vnew_frag[m_mm1][1] = u_top.y - vnew_frag[m_mm1][1];
            vnew_frag[m_mm1][2] = u_bot.x - vnew_frag[m_mm1][2];
            vnew_frag[m_mm1][3] = u_bot.y - vnew_frag[m_mm1][3];
        }

        // === Phase D: STG vnew (UNDECAYED), STS v_decay -> vd_view, scale h_frag ===
        const float g_C     = g_smem[cl - 1];
        const float gamma_C = exp2_approx(g_C * RCP_LN2_F);

#pragma unroll
        for (int m_mm1 = 0; m_mm1 < M_TILES_MM1_PW; ++m_mm1) {
            const int row_g0 = s_idx * BT_PER_WARP + m_mm1 * MMA_M + lane_g;
            const int row_g1 = row_g0 + 8;
            const int col_d0 = warp_d_global + 2 * lane_t;

            const bool top_in_chunk = (row_g0 < cl);
            const bool bot_in_chunk = (row_g1 < cl);

            const float g_top   = top_in_chunk ? g_smem[row_g0] : g_C;
            const float g_bot   = bot_in_chunk ? g_smem[row_g1] : g_C;
            const float dec_top = top_in_chunk ? exp2_approx((g_C - g_top) * RCP_LN2_F) : 0.0f;
            const float dec_bot = bot_in_chunk ? exp2_approx((g_C - g_bot) * RCP_LN2_F) : 0.0f;

            const float v0 = vnew_frag[m_mm1][0];
            const float v1 = vnew_frag[m_mm1][1];
            const float v2 = vnew_frag[m_mm1][2];
            const float v3 = vnew_frag[m_mm1][3];

            if (top_in_chunk) {
                const __nv_bfloat162 out = __floats2bfloat162_rn(v0, v1);
                store_vec(&v_new[vn_base + (int64_t)row_g0 * vn_stride + col_d0], out);
            }
            if (bot_in_chunk) {
                const __nv_bfloat162 out = __floats2bfloat162_rn(v2, v3);
                store_vec(&v_new[vn_base + (int64_t)row_g1 * vn_stride + col_d0], out);
            }

            const int row_g0_loc = s_idx * BT_PER_WARP + m_mm1 * MMA_M + lane_g;
            const int row_g1_loc = row_g0_loc + 8;
            const int col_d0_loc = warp_d_local + 2 * lane_t;
            store_vec(&vd_view.at(row_g0_loc, col_d0_loc),
                      make_float2(v0 * dec_top, v1 * dec_top));
            store_vec(&vd_view.at(row_g1_loc, col_d0_loc),
                      make_float2(v2 * dec_bot, v3 * dec_bot));
        }

#pragma unroll
        for (int m = 0; m < M_TILES_H_PW; ++m) {
#pragma unroll
            for (int e = 0; e < 4; ++e) { h_frag[m][e] *= gamma_C; }
        }

        // k is loaded synchronously. The barrier below gates matmul2's
        // reads of vd_view (Phase D write -> Phase E read) without draining
        // a possible W_next async prefetch.
        __syncthreads();

        // === Phase E: matmul2 over BT rows of k ===
        //
        // K_HALVES=1 (S=128 mainloop-rewrite): full k already in smem from
        // the prologue / chunk-end prefetch -- one mma2 sweep over BT/MMA_K
        // tiles, no mid-phase k reload, no extra sync.
        //
        // K_HALVES=2 (S=64/32/16): split the sweep at K_LOAD_ROWS, refill
        // k_smem with the second half mid-loop. Saves K_LOAD_FLT smem at
        // the cost of one extra synchronous reload + sync.
        if constexpr (K_HALVES == 1) {
            constexpr int K_TILES_MM2 = BT / MMA_K;
#pragma unroll
            for (int kt = 0; kt < K_TILES_MM2; ++kt) {
                const int k_off_local = kt * MMA_K;

                const int row_t0 = k_off_local + lane_t;
                const int row_t1 = row_t0 + 4;
                const int col_g  = warp_d_local + lane_g;
                const float b0   = vd_view.at(row_t0, col_g);
                const float b1   = vd_view.at(row_t1, col_g);

#pragma unroll
                for (int m = 0; m < M_TILES_H_PW; ++m) {
                    const int row_a0    = k_off_local + lane_t;
                    const int row_a1    = row_a0 + 4;
                    const int col_a_top = s_idx * S_PER_WARP + m * MMA_M + lane_g;
                    const int col_a_bot = col_a_top + 8;

                    const float a0 = k_view.at(row_a0, col_a_top);
                    const float a1 = k_view.at(row_a0, col_a_bot);
                    const float a2 = k_view.at(row_a1, col_a_top);
                    const float a3 = k_view.at(row_a1, col_a_bot);

                    mma_tf32(h_frag[m][0], h_frag[m][1], h_frag[m][2], h_frag[m][3], a0, a1,
                                     a2, a3, b0, b1);
                }
            }
        } else {
#pragma unroll
            for (int kt = 0; kt < K_TILES_MM2_PER_LOAD; ++kt) {
                const int k_off_local = kt * MMA_K;

                const int row_t0 = k_off_local + lane_t;
                const int row_t1 = row_t0 + 4;
                const int col_g  = warp_d_local + lane_g;
                const float b0   = vd_view.at(row_t0, col_g);
                const float b1   = vd_view.at(row_t1, col_g);

#pragma unroll
                for (int m = 0; m < M_TILES_H_PW; ++m) {
                    const int row_a0    = k_off_local + lane_t;
                    const int row_a1    = row_a0 + 4;
                    const int col_a_top = s_idx * S_PER_WARP + m * MMA_M + lane_g;
                    const int col_a_bot = col_a_top + 8;

                    const float a0 = k_view.at(row_a0, col_a_top);
                    const float a1 = k_view.at(row_a0, col_a_bot);
                    const float a2 = k_view.at(row_a1, col_a_top);
                    const float a3 = k_view.at(row_a1, col_a_bot);

                    mma_tf32(h_frag[m][0], h_frag[m][1], h_frag[m][2], h_frag[m][3], a0, a1,
                                     a2, a3, b0, b1);
                }
            }

            __syncthreads(); // before k_half_1 overwrites k_smem

            const int64_t k_base_h1 = k_base + (int64_t)K_LOAD_ROWS * k_stride;
            const int cl_kh1        = (cl > K_LOAD_ROWS) ? (cl - K_LOAD_ROWS) : 0;
            qus::kernels::issue_load_bf16_to_float_vec4<K_LOAD_ROWS, S, THREADS_K>(
                k_view, k_in + k_base_h1, k_stride, cl_kh1, tid);
            __syncthreads();

#pragma unroll
            for (int kt = 0; kt < K_TILES_MM2_PER_LOAD; ++kt) {
                const int k_off_local  = kt * MMA_K;
                const int k_off_global = k_off_local + K_LOAD_ROWS; // vd row

                const int row_t0 = k_off_global + lane_t;
                const int row_t1 = row_t0 + 4;
                const int col_g  = warp_d_local + lane_g;
                const float b0   = vd_view.at(row_t0, col_g);
                const float b1   = vd_view.at(row_t1, col_g);

#pragma unroll
                for (int m = 0; m < M_TILES_H_PW; ++m) {
                    const int row_a0    = k_off_local + lane_t;
                    const int row_a1    = row_a0 + 4;
                    const int col_a_top = s_idx * S_PER_WARP + m * MMA_M + lane_g;
                    const int col_a_bot = col_a_top + 8;

                    const float a0 = k_view.at(row_a0, col_a_top);
                    const float a1 = k_view.at(row_a0, col_a_bot);
                    const float a2 = k_view.at(row_a1, col_a_top);
                    const float a3 = k_view.at(row_a1, col_a_bot);

                    mma_tf32(h_frag[m][0], h_frag[m][1], h_frag[m][2], h_frag[m][3], a0, a1,
                                     a2, a3, b0, b1);
                }
            }
        }

        __syncthreads(); // before chunk-end loads overwrite k/U smem

        // Cross-chunk load: chunk t+1 U/k are converted synchronously into
        // shared memory.
        if (chunk + 1 < NT) {
            qus::kernels::issue_load_bf16_to_float_vec4<BT, N_STRIP_PER_BLOCK, THREADS_K>(
                U_view, U_in + W_base_next + d_off, W_stride, cl_next, tid);
            qus::kernels::issue_load_bf16_to_float_vec4<K_LOAD_ROWS, S, THREADS_K>(
                k_view, k_in + k_base_next, k_stride, cl_k_next, tid);
        }

        // R7.1: advance loop-carried accumulators for next iter.
        cs += BT;
        W_base = W_base_next;
        k_base = k_base_next;
        hc_base += hc_chunk_stride;
        vn_base += vn_chunk_stride;
        g_cs_offset += g_chunk_step;
    }

    // === Phase Z: store h_frag -> state_out (AR-transposed) ===
    const int64_t st_base = ((int64_t)b * H_v + h_v) * S * S;

#pragma unroll
    for (int m = 0; m < M_TILES_H_PW; ++m) {
        const int k_g0                              = s_idx * S_PER_WARP + m * MMA_M + lane_g;
        const int k_g1                              = k_g0 + 8;
        const int d0                                = warp_d_global + 2 * lane_t;
        const int d1                                = d0 + 1;
        state_out[st_base + (int64_t)d0 * S + k_g0] = h_frag[m][0];
        state_out[st_base + (int64_t)d1 * S + k_g0] = h_frag[m][1];
        state_out[st_base + (int64_t)d0 * S + k_g1] = h_frag[m][2];
        state_out[st_base + (int64_t)d1 * S + k_g1] = h_frag[m][3];
    }
}

template <int S>
cudaError_t launch_typed(const gdn_chunked::state_passing_config& cfg,
                         qus::kernels::head_map qk_map, int NT) {
    using D                  = kernel_dims<S>;
    constexpr int smem_bytes = smem_layout<S>::SMEM_FLOATS * (int)sizeof(float);

    cudaError_t err = cudaFuncSetAttribute(state_passing_gdn_kernel<S>,
                                           cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    if (err != cudaSuccess) return err;

    const dim3 grid((unsigned)(cfg.B * cfg.H_v * D::D_STRIPS), 1, 1);
    const dim3 block(D::THREADS, 1, 1);

    const int64_t k_stride_t =
        (cfg.k_stride_t_floats != 0) ? cfg.k_stride_t_floats : (int64_t)cfg.H_qk * cfg.S;

    state_passing_gdn_kernel<S><<<grid, block, smem_bytes, cfg.stream>>>(
        cfg.W, cfg.U, cfg.k, cfg.g_cumsum, cfg.state_in, cfg.v_new, cfg.h_chunk, cfg.state_out,
        cfg.L, cfg.H_v, qk_map, k_stride_t, NT);
    return cudaGetLastError();
}

} // namespace

cudaError_t launch_state_passing(const gdn_chunked::state_passing_config& cfg) {
    gdn_chunked::stage_validator v{"launch_state_passing", cfg.S, cfg.H_qk, cfg.H_v, cfg.L, cfg.B};
    QUS_GDN_PROPAGATE(v.check_shape());
    QUS_GDN_PROPAGATE(v.check_gdn_full_chunks());
    if (cfg.W == nullptr || cfg.U == nullptr || cfg.k == nullptr || cfg.g_cumsum == nullptr ||
        cfg.state_in == nullptr || cfg.v_new == nullptr || cfg.h_chunk == nullptr ||
        cfg.state_out == nullptr) {
        return cudaErrorInvalidValue;
    }

    const auto qk_map = qus::kernels::head_map::of((int)cfg.H_qk, (int)cfg.H_v);
    const int64_t NT  = div_up(cfg.L, static_cast<int64_t>(BT));

    // grid_x = B * H_v * D_STRIPS depends on S; check inside each case.
    auto check_grid_for = [&](int d_strips) -> cudaError_t {
        return v.check_grid(cfg.B * cfg.H_v * d_strips, /*grid_y=*/1, /*grid_z=*/1);
    };

    switch (cfg.S) {
    case 16:
        QUS_GDN_PROPAGATE(check_grid_for(kernel_dims<16>::D_STRIPS));
        return launch_typed<16>(cfg, qk_map, (int)NT);
    case 32:
        QUS_GDN_PROPAGATE(check_grid_for(kernel_dims<32>::D_STRIPS));
        return launch_typed<32>(cfg, qk_map, (int)NT);
    case 64:
        QUS_GDN_PROPAGATE(check_grid_for(kernel_dims<64>::D_STRIPS));
        return launch_typed<64>(cfg, qk_map, (int)NT);
    case 128:
        QUS_GDN_PROPAGATE(check_grid_for(kernel_dims<128>::D_STRIPS));
        return launch_typed<128>(cfg, qk_map, (int)NT);
    default:
        return cudaErrorInvalidValue; // check_shape already filtered
    }
}

} // namespace qus::kernels::detail::gdn_state_passing
