#pragma once

#include "kernels/common/mma.cuh"
#include "kernels/gated_delta_rule/chunked_common.cuh"

#include <cmath>
#include <cstdio>
#include <utility>

// Stage 2 (fused): build T_inv in smem, immediately consume it to produce
// W and U. T_inv never crosses HBM (workspace T_inv region is 0 bytes).
//
// Math + I/O layouts: see gdn chunked prepare_wy_wu_config.
// Block / smem total: 4 warps (128 t), smem ~49 KB at S=128 (16 KB T_inv +
//                     32 KB VK alias + 0.75 KB g/beta/bg).


#include <cstdio>
#include <utility>

namespace ninfer::kernels::detail::gdn_prepare_wy_wu {

namespace {

using gdn_chunked::BT;
using gdn_chunked::BC;
using gdn_chunked::MMA_M;
using gdn_chunked::MMA_N;
using gdn_chunked::MMA_K;
using gdn_chunked::bh_decode_t;
using gdn_chunked::zero_frag;
using ninfer::kernels::SmemTile;
using ninfer::kernels::mma_tf32;

static_assert(gdn_chunked::kChunkSize == 64,
              "stage_prepare_wy_wu: kChunkSize must be 64 (kernel hard-codes "
              "BT=64 = 4 * BC=16)");

constexpr int N_SUB     = BT / BC;                           // 4
constexpr int N_WARPS   = N_SUB;                             // 4 warps
constexpr int THREADS   = N_WARPS * ninfer::kernels::kWarpSize; // 128
constexpr int N_K_TILES = BT / MMA_K;                        // 8 (recompute_wu)

static_assert(MMA_M == BC, "kernel assumes MMA m == BC");

// Phase D scratch row stride. Smallest value >=16 (per-warp data needs
// 16 cols) that preserves the 2-way-write / 0-way-read property:
//   * stride%32 = 20 -> (20*lane_g + 2*lane_t) lands all 32 lanes on 16
//     distinct even banks (strict 2-way write); cumulative shifts
//     {0,20,8,28,16,4,24,12} are all distinct mod 32.
//   * Reads (20*lane_g + lane_t) span 32 distinct banks (0-way read).
// Equivalent BC to stride=36 but saves 1024B of scratch_floats, which at
// S=128 lets the kernel fit 3 blocks/SM in the 100 KB smem pool.
constexpr int SCR_STRIDE = 20;

// ---------------------------------------------------------------------------
// kernel_dims<S>: K-chunking, output N-chunking, fused smem layout.
//
// VK_floats = max(BT*S, scratch). At S<=16 the natural BT*S=1024 floats is
// LESS than scratch=1280 -- without max() Phase D scatter would silently
// overflow into g_smem/beta_smem.
// ---------------------------------------------------------------------------
template <int S>
struct kernel_dims {
    // K-chunk dims for the prepare_wy half (K loaded in chunks of <= 64 cols).
    static constexpr int K_CHUNK       = (S > 64) ? 64 : S;
    static constexpr int N_K_CHUNKS    = S / K_CHUNK;
    static constexpr int K_TILES_CHUNK = K_CHUNK / MMA_K;
    static_assert(S % K_CHUNK == 0, "S must be a multiple of the K chunk size");

    // recompute_wu N-axis chunking (caps D fragment at 16 fp32/lane).
    static constexpr int N_TILES_TOTAL     = S / MMA_N;
    static constexpr int N_TILES_PER_CHUNK = (N_TILES_TOTAL < 4) ? N_TILES_TOTAL : 4;
    static constexpr int N_CHUNKS          = N_TILES_TOTAL / N_TILES_PER_CHUNK;
    static_assert(N_CHUNKS * N_TILES_PER_CHUNK == N_TILES_TOTAL, "must divide");

    // Fused smem layout (floats).
    static constexpr int T_inv_floats   = BT * BT; // 16 KB
    static constexpr int natural_VK     = BT * S;
    static constexpr int scratch_floats = N_WARPS * BC * SCR_STRIDE; // 1280
    static constexpr int VK_floats      = natural_VK > scratch_floats ? natural_VK : scratch_floats;
    static constexpr int g_floats       = BT;
    static constexpr int beta_floats    = BT;
    static constexpr int bg_floats      = BT;
    static constexpr int SMEM_FLOATS =
        T_inv_floats + VK_floats + g_floats + beta_floats + bg_floats;
};

// ---------------------------------------------------------------------------
// Phase D helpers. Refactored from in-kernel lambdas to namespace-scope
// __device__ functions to drop the spill-via-lambda-capture pattern that
// nvcc's register allocator could not see through.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void
scatter_frag_to_scr(const float frag[8], float* __restrict__ scr_smem, int warp, int lane) {
    float* Sptr                                  = scr_smem + warp * BC * SCR_STRIDE;
    const int lane_g                             = lane >> 2;
    const int col_2t                             = (lane & 3) << 1;
    Sptr[lane_g * SCR_STRIDE + col_2t]           = frag[0];
    Sptr[lane_g * SCR_STRIDE + col_2t + 1]       = frag[1];
    Sptr[(lane_g + 8) * SCR_STRIDE + col_2t]     = frag[2];
    Sptr[(lane_g + 8) * SCR_STRIDE + col_2t + 1] = frag[3];
    Sptr[lane_g * SCR_STRIDE + col_2t + 8]       = frag[4];
    Sptr[lane_g * SCR_STRIDE + col_2t + 9]       = frag[5];
    Sptr[(lane_g + 8) * SCR_STRIDE + col_2t + 8] = frag[6];
    Sptr[(lane_g + 8) * SCR_STRIDE + col_2t + 9] = frag[7];
}

// 16x16x16 mma: A from raw row-major scratch (stride SCR_STRIDE),
// B from swizzled M_view at offset (M_row_off, M_col_off).
__device__ __forceinline__ void mma16_raw_x_swiz(float D[8], int lane,
                                                 const float* __restrict__ A_buf,
                                                 SmemTile<BT> M_view, int M_row_off,
                                                 int M_col_off) {
    const int lane_g = lane >> 2;
    const int lane_t = lane & 3;
#pragma unroll
    for (int kt = 0; kt < 2; ++kt) {
        const int k_off = kt * MMA_K;
        const float a0  = A_buf[lane_g * SCR_STRIDE + (k_off + lane_t)];
        const float a1  = A_buf[(lane_g + 8) * SCR_STRIDE + (k_off + lane_t)];
        const float a2  = A_buf[lane_g * SCR_STRIDE + (k_off + lane_t + 4)];
        const float a3  = A_buf[(lane_g + 8) * SCR_STRIDE + (k_off + lane_t + 4)];
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const int n_off = nt * MMA_N;
            const float b0  = M_view.at(M_row_off + k_off + lane_t, M_col_off + n_off + lane_g);
            const float b1  = M_view.at(M_row_off + k_off + lane_t + 4, M_col_off + n_off + lane_g);
            mma_tf32(D[nt * 4 + 0], D[nt * 4 + 1], D[nt * 4 + 2], D[nt * 4 + 3], a0, a1, a2,
                             a3, b0, b1);
        }
    }
}

// 16x16x16 mma: A from swizzled M_view, B from raw row-major scratch.
__device__ __forceinline__ void mma16_swiz_x_raw(float D[8], int lane, SmemTile<BT> M_view,
                                                 int M_row_off, int M_col_off,
                                                 const float* __restrict__ B_buf) {
    const int lane_g = lane >> 2;
    const int lane_t = lane & 3;
#pragma unroll
    for (int kt = 0; kt < 2; ++kt) {
        const int k_off = kt * MMA_K;
        const float a0  = M_view.at(M_row_off + lane_g, M_col_off + k_off + lane_t);
        const float a1  = M_view.at(M_row_off + lane_g + 8, M_col_off + k_off + lane_t);
        const float a2  = M_view.at(M_row_off + lane_g, M_col_off + k_off + lane_t + 4);
        const float a3  = M_view.at(M_row_off + lane_g + 8, M_col_off + k_off + lane_t + 4);
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const int n_off = nt * MMA_N;
            const float b0  = B_buf[(k_off + lane_t) * SCR_STRIDE + (n_off + lane_g)];
            const float b1  = B_buf[(k_off + lane_t + 4) * SCR_STRIDE + (n_off + lane_g)];
            mma_tf32(D[nt * 4 + 0], D[nt * 4 + 1], D[nt * 4 + 2], D[nt * 4 + 3], a0, a1, a2,
                             a3, b0, b1);
        }
    }
}

// Templated on (MY_W, MY_J) so `A_reg[k]` resolves to compile-time indices
// after unrolling; otherwise nvcc parks A_reg in local memory.
template <int MY_W, int MY_J>
__device__ __forceinline__ void
compute_off_diag(float out[8], int warp, int lane, const float A_reg[N_SUB][8],
                 float* __restrict__ scr_smem, SmemTile<BT> M_view) {
    static_assert(0 <= MY_J && MY_J < MY_W && MY_W <= N_SUB);

    float sum[8] = {};

#pragma unroll
    for (int k = MY_J; k < MY_W; ++k) {
        scatter_frag_to_scr(A_reg[k], scr_smem, warp, lane);
        __syncwarp();
        const float* A_buf = scr_smem + warp * BC * SCR_STRIDE;
        mma16_raw_x_swiz(sum, lane, A_buf, M_view, k * BC, MY_J * BC);
        __syncwarp();
        if (k == MY_J) { // diagonal-block correction (folded after unroll)
#pragma unroll
            for (int e = 0; e < 8; ++e) sum[e] += A_reg[k][e];
        }
    }

    scatter_frag_to_scr(sum, scr_smem, warp, lane);
    __syncwarp();
    const float* B_buf = scr_smem + warp * BC * SCR_STRIDE;

    float prod[8] = {};
    mma16_swiz_x_raw(prod, lane, M_view, MY_W * BC, MY_W * BC, B_buf);

#pragma unroll
    for (int e = 0; e < 8; ++e) out[e] = prod[e] + sum[e];
}

__device__ __forceinline__ void store_frag_to_M(const float frag[8], int my_w, int my_j, int lane_g,
                                                int lane_t, SmemTile<BT> M_view) {
    const int row_g0                = my_w * BC + lane_g;
    const int row_g1                = row_g0 + 8;
    const int col_base              = my_j * BC + 2 * lane_t;
    M_view.at(row_g0, col_base)     = frag[0];
    M_view.at(row_g0, col_base + 1) = frag[1];
    M_view.at(row_g1, col_base)     = frag[2];
    M_view.at(row_g1, col_base + 1) = frag[3];
    M_view.at(row_g0, col_base + 8) = frag[4];
    M_view.at(row_g0, col_base + 9) = frag[5];
    M_view.at(row_g1, col_base + 8) = frag[6];
    M_view.at(row_g1, col_base + 9) = frag[7];
}

// __launch_bounds__ NOT applied: nvcc's natural reg=203 is the sweet spot.
// Capping at 256 (min_blocks=2) just spills the overflow.
template <int S>
__global__ void
prepare_wy_wu_gdn_kernel(const __nv_bfloat16* __restrict__ k_in,
                         const __nv_bfloat16* __restrict__ v_in, const float* __restrict__ g_in,
                         const float* __restrict__ beta_in, __nv_bfloat16* __restrict__ W,
                         __nv_bfloat16* __restrict__ U, float* __restrict__ g_cumsum_out, int64_t T,
                         int64_t H_v, ninfer::kernels::head_map qk_map,
                         // Token-axis strides (in floats) for k / v. Caller
                         // passes materialised values (launch_typed handles
                         // 0 -> packed defaults H_qk*S / H_v*S).
                         int64_t k_stride_t, int64_t v_stride_t) {
    using KD                        = kernel_dims<S>;
    constexpr int K_STRIDE          = KD::K_CHUNK;
    constexpr int K_CHUNK           = KD::K_CHUNK;
    constexpr int K_TILES_CHUNK     = KD::K_TILES_CHUNK;
    constexpr int N_K_CHUNKS        = KD::N_K_CHUNKS;
    constexpr int N_TILES_PER_CHUNK = KD::N_TILES_PER_CHUNK;
    constexpr int N_CHUNKS          = KD::N_CHUNKS;

    extern __shared__ float smem[];
    float* const T_inv_smem = smem;
    float* const VK_smem    = smem + KD::T_inv_floats;
    float* const g_smem     = VK_smem + KD::VK_floats;
    float* const beta_smem  = g_smem + BT;
    float* const bg_smem    = beta_smem + BT;

    // VK_smem aliases: K (Phase WY-B) -> scr (Phase WY-D) -> V (Phase WU-A)
    // -> K (Phase WU-C). Each transition is gated by an existing barrier.
    float* const K_smem   = VK_smem;
    float* const scr_smem = VK_smem;

    SmemTile<BT> M_view{T_inv_smem};   // T_inv from Phase WY-C onward
    SmemTile<BT> T_view{T_inv_smem};   // recompute_wu reads via T_view
    SmemTile<K_STRIDE> K_view{K_smem}; // prepare_wy Phase B
    SmemTile<S> VK_view{VK_smem};      // recompute_wu Phase A..D

    const int tid    = threadIdx.x;
    const auto lanes = ninfer::kernels::mma_lane_t::decode(tid);
    const int lane   = lanes.lane;
    const int warp   = lanes.warp;
    const int lane_g = lanes.lane_g;
    const int lane_t = lanes.lane_t;

    const int chunk = blockIdx.x;
    // Grouped mapping: adjacent CTAs already share the same qk head under
    // identity cta_h_v (0..G-1 -> qk_head 0, etc.).
    const auto bh = bh_decode_t::of(blockIdx.y, qk_map);
    const int b   = bh.b;
    const int h_v = bh.h_v;

    const auto cb    = ninfer::kernels::chunk_bounds_t::of(chunk, T, BT);
    const int64_t cs = cb.cs;
    const int cl     = cb.cl;

    // === Phase WY-A: cooperative load of beta + warp-0 in-place scan of g ===
    //
    // beta load: 64 threads (warp 0 + 1) load BT entries.
    // g scan: only warp 0 participates -- each lane handles 2 consecutive
    // tokens (a[2L], a[2L+1]) of g_in for this (b, chunk, h_v), runs a
    // Hillis-Steele inclusive scan via shfl, and writes both g_smem[]
    // (consumed by Phase WY-C / WU-A) AND HBM g_cumsum_out (consumed
    // by stages 3/4 unchanged). This folds the old standalone g_cumsum
    // kernel (~10 us, 1.3% of e2e) into here at zero added latency: the
    // scan is hidden behind Phase WY-B's K load + KKT mma.
    //
    // No __syncthreads here -- per-chunk K loader below issues one before
    // any read of g_smem / beta_smem can race the scan stores.
    if (tid < BT) {
        const int64_t boff = ((int64_t)b * T + cs + tid) * H_v + h_v;
        beta_smem[tid]     = (tid < cl) ? beta_in[boff] : 0.0f;
    }

    if (warp == 0) {
        const int64_t g_row_base = ((int64_t)b * T + cs) * H_v + h_v;
        const int t0             = 2 * lane; // 0, 2, ..., 62
        const int t1             = t0 + 1;   // 1, 3, ..., 63

        // Out-of-chunk tokens contribute 0 to the prefix sum (matches
        // the standalone g_cumsum kernel).
        const float a  = (t0 < cl) ? g_in[g_row_base + (int64_t)t0 * H_v] : 0.0f;
        const float bv = (t1 < cl) ? g_in[g_row_base + (int64_t)t1 * H_v] : 0.0f;

        // Hillis-Steele inclusive scan over per-lane partials (a + bv).
        float partial = a + bv;
#pragma unroll
        for (int o = 1; o < ninfer::kernels::kWarpSize; o <<= 1) {
            const float n = __shfl_up_sync(0xffffffffu, partial, o);
            if (lane >= o) partial += n;
        }
        // Inclusive -> exclusive shift: lane 0's prefix is 0.
        const float prev_inc  = __shfl_up_sync(0xffffffffu, partial, 1);
        const float ex_prefix = (lane == 0) ? 0.0f : prev_inc;

        const float c0 = ex_prefix + a;
        const float c1 = c0 + bv;

        // Always populate g_smem (zero past cl) so Phase WY-C reg's
        // unconditional reads stay valid for partial-chunk shapes.
        g_smem[t0] = (t0 < cl) ? c0 : 0.0f;
        g_smem[t1] = (t1 < cl) ? c1 : 0.0f;

        // Skip HBM stores past cl -- matches the standalone g_cumsum kernel
        // and avoids touching out-of-chunk slots.
        if (t0 < cl) g_cumsum_out[g_row_base + (int64_t)t0 * H_v] = c0;
        if (t1 < cl) g_cumsum_out[g_row_base + (int64_t)t1 * H_v] = c1;
    }

    // === Phase WY-B: KKT via TF32 MMA on lower-tri 4x4 sub-block grid ===
    float A_reg[N_SUB][8];
    zero_frag(A_reg);

    const int row_g0 = warp * BC + lane_g; // strip rows 0..7
    const int row_g1 = row_g0 + 8;         // strip rows 8..15

    const int64_t k_base = ((int64_t)b * T + cs) * k_stride_t + (int64_t)qk_map.qk_head(h_v) * S;
    constexpr int VEC_PER_ROW_CHUNK = K_CHUNK / 4;
    constexpr int N_VEC_CHUNK       = BT * VEC_PER_ROW_CHUNK;

    auto kkt_strip = [&]<int N_OWNED>() {
#pragma unroll
        for (int k_tile = 0; k_tile < K_TILES_CHUNK; ++k_tile) {
            const int k_off  = k_tile * MMA_K;
            const int col_t0 = k_off + lane_t;
            const int col_t1 = col_t0 + 4;

            const float a0 = K_view.at(row_g0, col_t0);
            const float a1 = K_view.at(row_g1, col_t0);
            const float a2 = K_view.at(row_g0, col_t1);
            const float a3 = K_view.at(row_g1, col_t1);

#pragma unroll
            for (int j_sub = 0; j_sub < N_OWNED; ++j_sub) {
#pragma unroll
                for (int n_tile = 0; n_tile < 2; ++n_tile) {
                    const int n_off = n_tile * MMA_N;
                    const int row_b = j_sub * BC + n_off + lane_g;
                    const float b0  = K_view.at(row_b, col_t0);
                    const float b1  = K_view.at(row_b, col_t1);

                    mma_tf32(A_reg[j_sub][n_tile * 4 + 0], A_reg[j_sub][n_tile * 4 + 1],
                                     A_reg[j_sub][n_tile * 4 + 2], A_reg[j_sub][n_tile * 4 + 3], a0,
                                     a1, a2, a3, b0, b1);
                }
            }
        }
    };

#pragma unroll
    for (int kc = 0; kc < N_K_CHUNKS; ++kc) {
        const int chunk_col = kc * K_CHUNK;

#pragma unroll
        for (int v = tid; v < N_VEC_CHUNK; v += THREADS) {
            const int row  = v / VEC_PER_ROW_CHUNK;
            const int col4 = v - row * VEC_PER_ROW_CHUNK;
            float4 val     = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (row < cl) {
                val = ninfer::kernels::load_bf16_vec4_as_float4(
                    k_in + k_base + (int64_t)row * k_stride_t + chunk_col + col4 * 4);
            }
            K_view.vec4_at(row, col4 * 4) = val;
        }
        __syncthreads();

        switch (warp) {
        case 0:
            kkt_strip.template operator()<1>();
            break;
        case 1:
            kkt_strip.template operator()<2>();
            break;
        case 2:
            kkt_strip.template operator()<3>();
            break;
        case 3:
            kkt_strip.template operator()<4>();
            break;
        }

        if (kc + 1 < N_K_CHUNKS) { __syncthreads(); }
    }

    // === Phase WY-C reg: gate decay + beta scaling on fragments ===
    const int r_g0       = warp * BC + lane_g;
    const int r_g1       = r_g0 + 8;
    const float beta_r0  = beta_smem[r_g0];
    const float beta_r1  = beta_smem[r_g1];
    const float g_r0     = g_smem[r_g0];
    const float g_r1     = g_smem[r_g1];
    const float nbeta_r0 = -beta_r0;
    const float nbeta_r1 = -beta_r1;

#pragma unroll
    for (int j_sub = 0; j_sub < N_SUB; ++j_sub) {
        if (j_sub > warp) continue;

        const int c_base = j_sub * BC + 2 * lane_t;
        const int c0     = c_base;
        const int c1     = c_base + 1;
        const int c2     = c_base + 8;
        const int c3     = c_base + 9;

        const float g_c0 = g_smem[c0];
        const float g_c1 = g_smem[c1];
        const float g_c2 = g_smem[c2];
        const float g_c3 = g_smem[c3];

        const bool is_diag = (j_sub == warp);

        // Diagonal block: keep strict lower triangle (r > c), zero the rest.
        // Off-diagonal blocks (j_sub < warp): keep all entries.
        //
        // Footgun: g_cumsum is monotone decreasing, so on the diagonal block's
        // strict-upper triangle (r < c) g_r - g_c is large positive and expf
        // can saturate to +inf. A `* mask` (mask = 0/1 float) would then
        // produce inf * 0 = NaN. The conditional select below overwrites the
        // bad product with 0.0f instead, so no NaN escapes (same pattern as
        // stage_chunk_output.cu Phase C).
        A_reg[j_sub][0] =
            (!is_diag || r_g0 > c0) ? nbeta_r0 * A_reg[j_sub][0] * expf(g_r0 - g_c0) : 0.0f;
        A_reg[j_sub][1] =
            (!is_diag || r_g0 > c1) ? nbeta_r0 * A_reg[j_sub][1] * expf(g_r0 - g_c1) : 0.0f;
        A_reg[j_sub][2] =
            (!is_diag || r_g1 > c0) ? nbeta_r1 * A_reg[j_sub][2] * expf(g_r1 - g_c0) : 0.0f;
        A_reg[j_sub][3] =
            (!is_diag || r_g1 > c1) ? nbeta_r1 * A_reg[j_sub][3] * expf(g_r1 - g_c1) : 0.0f;
        A_reg[j_sub][4] =
            (!is_diag || r_g0 > c2) ? nbeta_r0 * A_reg[j_sub][4] * expf(g_r0 - g_c2) : 0.0f;
        A_reg[j_sub][5] =
            (!is_diag || r_g0 > c3) ? nbeta_r0 * A_reg[j_sub][5] * expf(g_r0 - g_c3) : 0.0f;
        A_reg[j_sub][6] =
            (!is_diag || r_g1 > c2) ? nbeta_r1 * A_reg[j_sub][6] * expf(g_r1 - g_c2) : 0.0f;
        A_reg[j_sub][7] =
            (!is_diag || r_g1 > c3) ? nbeta_r1 * A_reg[j_sub][7] * expf(g_r1 - g_c3) : 0.0f;
    }

    // === Phase WY-C diag: scatter A_reg[w][w] -> M_view, in-place forward sub ===
    __syncthreads(); // gates K_smem (Phase B reads) -> M_smem alias handover

    {
        constexpr int N = BT * BT;
#pragma unroll
        for (int idx = tid; idx < N; idx += THREADS) T_inv_smem[idx] = 0.0f;
    }
    __syncthreads();

    // Switch on `warp` so A_reg's row index is compile-time per case;
    // see compute_off_diag comment for rationale.
    switch (warp) {
    case 0:
        store_frag_to_M(A_reg[0], 0, 0, lane_g, lane_t, M_view);
        break;
    case 1:
        store_frag_to_M(A_reg[1], 1, 1, lane_g, lane_t, M_view);
        break;
    case 2:
        store_frag_to_M(A_reg[2], 2, 2, lane_g, lane_t, M_view);
        break;
    case 3:
        store_frag_to_M(A_reg[3], 3, 3, lane_g, lane_t, M_view);
        break;
    }
    __syncwarp();

    {
        const int diag_off = warp * BC;
        const int wcol     = lane & 15;
        for (int i = 1; i < BC; ++i) {
            const int row_i = diag_off + i;
            const int col   = diag_off + wcol;
            float sum       = 0.0f;
#pragma unroll
            for (int j = 0; j < BC - 1; ++j) {
                if (j < i) { sum += M_view.at(row_i, diag_off + j) * M_view.at(diag_off + j, col); }
            }
            __syncwarp();
            if (lane < 16 && wcol < i) { M_view.at(row_i, col) += sum; }
            __syncwarp();
        }
    }
    __syncthreads();

    // === Phase WY-D: block-Schur off-diagonal completion (3 waves) ===
    // Switch on `warp` so (MY_W, MY_J) are compile-time per case.
    auto wave_compute_store = [&]<int MY_W, int MY_J>() {
        float out[8];
        compute_off_diag<MY_W, MY_J>(out, warp, lane, A_reg, scr_smem, M_view);
        store_frag_to_M(out, MY_W, MY_J, lane_g, lane_t, M_view);
    };

    switch (warp) {
    case 1:
        wave_compute_store.template operator()<1, 0>();
        break;
    case 2:
        wave_compute_store.template operator()<2, 1>();
        break;
    case 3:
        wave_compute_store.template operator()<3, 2>();
        break;
    }
    __syncthreads();

    switch (warp) {
    case 2:
        wave_compute_store.template operator()<2, 0>();
        break;
    case 3:
        wave_compute_store.template operator()<3, 1>();
        break;
    }
    __syncthreads();

    if (warp == 3) { wave_compute_store.template operator()<3, 0>(); }
    __syncthreads();

    // === Phase WY-E: +I on diagonal of T_inv (no HBM write; sync fused into WU-A) ===
    if (tid < cl) { M_view.at(tid, tid) += 1.0f; }

    // === Phase WU-A: load V into VK_smem; pre-compute bg = beta * exp(g) ===
    {
        const int64_t row_base =
            (int64_t)b * T * v_stride_t + (int64_t)cs * v_stride_t + (int64_t)h_v * S;
        const int64_t row_stride = v_stride_t;
        ninfer::kernels::issue_load_bf16_to_float_vec4<BT, S, THREADS>(VK_view, v_in + row_base,
                                                                    row_stride, cl, tid);
    }

    if (tid < BT) { bg_smem[tid] = beta_smem[tid] * expf(g_smem[tid]); }

    __syncthreads(); // gates +I, bg writes, and V smem visibility

    // Per-thread fragment indexing for the recompute_wu matmuls.
    const int t_row_g0 = warp * MMA_M + lane_g;
    const int t_row_g1 = t_row_g0 + 8;
    const int col_d0   = (lane & 3) << 1;

    const int64_t out_base       = ((int64_t)b * T + cs) * H_v * S + (int64_t)h_v * S;
    const int64_t out_row_stride = (int64_t)H_v * S;

    // 16x32 (or smaller at S<128) mma over one N-chunk of the fully-loaded
    // VK_view. N_CHUNK is compile-time so the column offset folds away and
    // each chunk's D[N_TILES_PER_CHUNK][4] = 16 fp32/lane stays in registers
    // for the chunk's lifetime only (vs lifting across all chunks).
    auto matmul_chunk = [&]<int N_CHUNK>(__nv_bfloat16* __restrict__ out_gmem,
                                         const float* __restrict__ scale_smem) {
        constexpr int n_chunk_off = N_CHUNK * (N_TILES_PER_CHUNK * MMA_N);

        float D[N_TILES_PER_CHUNK][4];
        zero_frag(D);

#pragma unroll
        for (int k_tile = 0; k_tile < N_K_TILES; ++k_tile) {
            const int k_off  = k_tile * MMA_K;
            const int col_t0 = k_off + lane_t;
            const int col_t1 = col_t0 + 4;
            const float a0   = T_view.at(t_row_g0, col_t0);
            const float a1   = T_view.at(t_row_g1, col_t0);
            const float a2   = T_view.at(t_row_g0, col_t1);
            const float a3   = T_view.at(t_row_g1, col_t1);

            // Inline scale_smem reads (vs a pre-loaded scale_kt[2*N_K_TILES]
            // register array, which adds 16 reg/lane of pressure unnecessarily
            // -- LDS latency is hidden behind the following mma's TC pipe).
            const int row_t0 = k_off + lane_t;
            const int row_t1 = row_t0 + 4;
            const float s0   = scale_smem[row_t0];
            const float s1   = scale_smem[row_t1];

#pragma unroll
            for (int t = 0; t < N_TILES_PER_CHUNK; ++t) {
                const int n_off = n_chunk_off + t * MMA_N;
                const int col_g = n_off + lane_g;
                const float b0  = VK_view.at(row_t0, col_g) * s0;
                const float b1  = VK_view.at(row_t1, col_g) * s1;

                mma_tf32(D[t][0], D[t][1], D[t][2], D[t][3], a0, a1, a2, a3, b0, b1);
            }
        }

#pragma unroll
        for (int t = 0; t < N_TILES_PER_CHUNK; ++t) {
            const int col = n_chunk_off + t * MMA_N + col_d0;
            if (t_row_g0 < cl) {
                const __nv_bfloat162 v0 = __floats2bfloat162_rn(D[t][0], D[t][1]);
                store_vec(&out_gmem[(int64_t)t_row_g0 * out_row_stride + col], v0);
            }
            if (t_row_g1 < cl) {
                const __nv_bfloat162 v1 = __floats2bfloat162_rn(D[t][2], D[t][3]);
                store_vec(&out_gmem[(int64_t)t_row_g1 * out_row_stride + col], v1);
            }
        }
    };

    auto matmul_two_chunks = [&]<int C0, int C1>(__nv_bfloat16* __restrict__ out_gmem,
                                                 const float* __restrict__ scale_smem) {
        constexpr int c0_off = C0 * (N_TILES_PER_CHUNK * MMA_N);
        constexpr int c1_off = C1 * (N_TILES_PER_CHUNK * MMA_N);

        float D0[N_TILES_PER_CHUNK][4];
        float D1[N_TILES_PER_CHUNK][4];
        zero_frag(D0);
        zero_frag(D1);

#pragma unroll
        for (int k_tile = 0; k_tile < N_K_TILES; ++k_tile) {
            const int k_off  = k_tile * MMA_K;
            const int col_t0 = k_off + lane_t;
            const int col_t1 = col_t0 + 4;
            const float a0   = T_view.at(t_row_g0, col_t0);
            const float a1   = T_view.at(t_row_g1, col_t0);
            const float a2   = T_view.at(t_row_g0, col_t1);
            const float a3   = T_view.at(t_row_g1, col_t1);

            const int row_t0 = k_off + lane_t;
            const int row_t1 = row_t0 + 4;
            const float s0   = scale_smem[row_t0];
            const float s1   = scale_smem[row_t1];

#pragma unroll
            for (int t = 0; t < N_TILES_PER_CHUNK; ++t) {
                const int n_off = c0_off + t * MMA_N;
                const int col_g = n_off + lane_g;
                const float b0  = VK_view.at(row_t0, col_g) * s0;
                const float b1  = VK_view.at(row_t1, col_g) * s1;

                mma_tf32(D0[t][0], D0[t][1], D0[t][2], D0[t][3], a0, a1, a2, a3, b0, b1);
            }

#pragma unroll
            for (int t = 0; t < N_TILES_PER_CHUNK; ++t) {
                const int n_off = c1_off + t * MMA_N;
                const int col_g = n_off + lane_g;
                const float b0  = VK_view.at(row_t0, col_g) * s0;
                const float b1  = VK_view.at(row_t1, col_g) * s1;

                mma_tf32(D1[t][0], D1[t][1], D1[t][2], D1[t][3], a0, a1, a2, a3, b0, b1);
            }
        }

#pragma unroll
        for (int t = 0; t < N_TILES_PER_CHUNK; ++t) {
            const int col = c0_off + t * MMA_N + col_d0;
            if (t_row_g0 < cl) {
                const __nv_bfloat162 v0 = __floats2bfloat162_rn(D0[t][0], D0[t][1]);
                store_vec(&out_gmem[(int64_t)t_row_g0 * out_row_stride + col], v0);
            }
            if (t_row_g1 < cl) {
                const __nv_bfloat162 v1 = __floats2bfloat162_rn(D0[t][2], D0[t][3]);
                store_vec(&out_gmem[(int64_t)t_row_g1 * out_row_stride + col], v1);
            }
        }

#pragma unroll
        for (int t = 0; t < N_TILES_PER_CHUNK; ++t) {
            const int col = c1_off + t * MMA_N + col_d0;
            if (t_row_g0 < cl) {
                const __nv_bfloat162 v0 = __floats2bfloat162_rn(D1[t][0], D1[t][1]);
                store_vec(&out_gmem[(int64_t)t_row_g0 * out_row_stride + col], v0);
            }
            if (t_row_g1 < cl) {
                const __nv_bfloat162 v1 = __floats2bfloat162_rn(D1[t][2], D1[t][3]);
                store_vec(&out_gmem[(int64_t)t_row_g1 * out_row_stride + col], v1);
            }
        }
    };

    // T_inv @ (scale * VK) -> out_gmem. C++20 fold-expression dispatch so
    // each n_chunk's D fragment is born and dies in its own scope.
    auto matmul_and_store = [&](__nv_bfloat16* __restrict__ out_gmem,
                                const float* __restrict__ scale_smem) {
        if constexpr (N_CHUNKS == 4) {
            matmul_two_chunks.template operator()<0, 1>(out_gmem, scale_smem);
            matmul_two_chunks.template operator()<2, 3>(out_gmem, scale_smem);
        } else {
            [&]<int... Is>(std::integer_sequence<int, Is...>) {
                (matmul_chunk.template operator()<Is>(out_gmem, scale_smem), ...);
            }(std::make_integer_sequence<int, N_CHUNKS>{});
        }
    };

    // === Phase WU-B: U = T_inv @ (beta * V) ===
    matmul_and_store(U + out_base, beta_smem);

    // === Phase WU-C: load K -> VK_smem (overwrite V) ===
    __syncthreads(); // V reads done before K load overwrites
    {
        const int64_t row_base =
            ((int64_t)b * T + cs) * k_stride_t + (int64_t)qk_map.qk_head(h_v) * S;
        ninfer::kernels::issue_load_bf16_to_float_vec4<BT, S, THREADS>(VK_view, k_in + row_base,
                                                                    k_stride_t, cl, tid);
    }
    __syncthreads();

    // === Phase WU-D: W = T_inv @ (bg * K) ===
    matmul_and_store(W + out_base, bg_smem);
}

template <int S>
cudaError_t launch_typed(const gdn_chunked::prepare_wy_wu_config& cfg, dim3 grid, dim3 block,
                         ninfer::kernels::head_map qk_map) {
    constexpr int smem_bytes = kernel_dims<S>::SMEM_FLOATS * (int)sizeof(float);

    cudaError_t err = cudaFuncSetAttribute(prepare_wy_wu_gdn_kernel<S>,
                                           cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    if (err != cudaSuccess) return err;

    // 0 means "use packed default" -- materialise here so the kernel only
    // ever sees a real stride.
    const int64_t k_stride_t =
        (cfg.k_stride_t_floats != 0) ? cfg.k_stride_t_floats : (int64_t)cfg.H_qk * cfg.S;
    const int64_t v_stride_t =
        (cfg.v_stride_t_floats != 0) ? cfg.v_stride_t_floats : (int64_t)cfg.H_v * cfg.S;

    prepare_wy_wu_gdn_kernel<S><<<grid, block, smem_bytes, cfg.stream>>>(
        cfg.k, cfg.v, cfg.g_in, cfg.beta, cfg.W, cfg.U, cfg.g_cumsum_out, cfg.L, cfg.H_v, qk_map,
        k_stride_t, v_stride_t);
    return cudaGetLastError();
}

} // namespace

cudaError_t launch_prepare_wy_wu(const gdn_chunked::prepare_wy_wu_config& cfg) {
    gdn_chunked::stage_validator v{"launch_prepare_wy_wu", cfg.S, cfg.H_qk, cfg.H_v, cfg.L, cfg.B};
    NINFER_GDN_PROPAGATE(v.check_shape());
    NINFER_GDN_PROPAGATE(v.check_gdn_full_chunks());
    if (cfg.k == nullptr || cfg.v == nullptr || cfg.g_in == nullptr || cfg.beta == nullptr ||
        cfg.W == nullptr || cfg.U == nullptr || cfg.g_cumsum_out == nullptr) {
        return cudaErrorInvalidValue;
    }

    const auto qk_map = ninfer::kernels::head_map::of((int)cfg.H_qk, (int)cfg.H_v);
    const int64_t NT  = div_up(cfg.L, static_cast<int64_t>(BT));
    const int64_t bh  = cfg.B * cfg.H_v;
    NINFER_GDN_PROPAGATE(v.check_grid(NT, bh));

    const dim3 grid((unsigned)NT, (unsigned)bh, 1);
    const dim3 block(THREADS, 1, 1);

    switch (cfg.S) {
    case 16:
        return launch_typed<16>(cfg, grid, block, qk_map);
    case 32:
        return launch_typed<32>(cfg, grid, block, qk_map);
    case 64:
        return launch_typed<64>(cfg, grid, block, qk_map);
    case 128:
        return launch_typed<128>(cfg, grid, block, qk_map);
    default:
        return cudaErrorInvalidValue; // check_shape already filtered
    }
}

} // namespace ninfer::kernels::detail::gdn_prepare_wy_wu
