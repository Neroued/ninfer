#pragma once

#include "ops/common/mma.cuh"
#include "ops/linear_attention/gated_delta_net/chunked/common.cuh"

// Stage 3: chunk-sequential state passing.
//
// Math + I/O layouts: see the Gated DeltaNet chunked state_passing_config.
// Two fixed geometries share one kernel body:
//   value strip 16: 8 warps, ~44 KB dynamic smem, two CTAs/SM;
//   value strip 32: 16 warps, ~56 KB dynamic smem, one CTA/SM.

namespace ninfer::ops::detail::gated_delta_net::chunked::state_passing {

using ninfer::ops::mma_tf32_bits;
using ninfer::ops::ldmatrix_x2;
using ninfer::ops::ldmatrix_x2_t;
using ninfer::ops::smem_addr;
using ninfer::ops::exp2_approx;

static_assert(kChunkSize == 64, "stage_state_passing: kChunkSize must be 64");
static_assert(kStateDim == 128);

template <int NStrip>
struct kernel_dims;

template <>
struct kernel_dims<16> {
    static constexpr int N_STRIP_PER_BLOCK  = 16;
    static constexpr int D_STRIPS           = kStateDim / N_STRIP_PER_BLOCK;
    static constexpr int DT_TILES_PER_BLOCK = N_STRIP_PER_BLOCK / MMA_N;
    static constexpr int BT_SPLITS          = 4;
    static constexpr int N_WARPS            = DT_TILES_PER_BLOCK * BT_SPLITS; // 8
    static constexpr int THREADS            = N_WARPS * ninfer::ops::kWarpSize;
    static constexpr int MIN_BLOCKS         = 2;
};

template <>
struct kernel_dims<32> {
    static constexpr int N_STRIP_PER_BLOCK  = 32;
    static constexpr int D_STRIPS           = kStateDim / N_STRIP_PER_BLOCK;
    static constexpr int DT_TILES_PER_BLOCK = N_STRIP_PER_BLOCK / MMA_N;
    static constexpr int BT_SPLITS          = 4;
    static constexpr int N_WARPS            = DT_TILES_PER_BLOCK * BT_SPLITS; // 16
    static constexpr int THREADS            = N_WARPS * ninfer::ops::kWarpSize;
    static constexpr int MIN_BLOCKS         = 1;
};

template <int NStrip>
struct smem_layout {
    using D                             = kernel_dims<NStrip>;
    static constexpr int W_STRIDE       = kStateDim;
    static constexpr int W_LOAD_BF16    = BT * W_STRIDE;
    static constexpr int W_STORAGE_FLT  = W_LOAD_BF16 / 2;
    static constexpr int UVD_FLT        = BT * D::N_STRIP_PER_BLOCK; // U-vd alias
    static constexpr int K_LOAD_ROWS    = BT;
    static constexpr int K_LOAD_BF16    = K_LOAD_ROWS * kStateDim;
    static constexpr int K_STORAGE_FLT  = K_LOAD_BF16 / 2;
    static constexpr int M_TILES_H_PW   = (kStateDim / D::BT_SPLITS) / MMA_M;
    static constexpr int SNAP_K_ROWS    = MMA_M * M_TILES_H_PW;
    static constexpr int SNAP_FLT       = SNAP_K_ROWS * D::N_STRIP_PER_BLOCK;
    static constexpr int M_TILES_H_GLOB = kStateDim / MMA_M;
    static constexpr int N_SNAP_ITERS   = M_TILES_H_GLOB / M_TILES_H_PW;
    // One buffer per sit so all warps can scatter their owned h_frag once at
    // chunk start (Phase A) and then read from any sit's buffer in the unified
    // matmul1 / coop_write paths without per-sit re-scatter+sync.
    static constexpr int N_SNAP_BUF = N_SNAP_ITERS;
    static constexpr int SMEM_FLOATS =
        W_STORAGE_FLT + UVD_FLT + K_STORAGE_FLT + SNAP_FLT * N_SNAP_BUF + BT;
};

// Snapshot stores the FP32 state operand as [value, state] with a stride-32
// swizzle shared by the scatter, cooperative h_chunk write, and MM1 LDSM read.
struct SnapView {
    float* __restrict__ base;
    static constexpr int kStride = kStateDim / 4;
    static_assert(kStride == 32);

    __device__ __forceinline__ int swz_xor(int row) const { return (row & 7) << 2; }

    __device__ __forceinline__ float& at(int row, int col) const {
        return base[row * kStride + (col ^ swz_xor(row))];
    }

    __device__ __forceinline__ float4& vec4_at(int row, int col) const {
        return *reinterpret_cast<float4*>(&base[row * kStride + (col ^ swz_xor(row))]);
    }
};

// prepare_wy_wu stores W's private workspace in the native-BF16 fragment
// order {0,4,1,5,2,6,3,7} within each group of eight logical columns.
struct Bf16WView {
    __nv_bfloat16* __restrict__ base;

    __device__ __forceinline__ int swizzled_col(int row, int col) const {
        return (col & ~63) + ((((col & 63) >> 3) ^ (row & 7)) << 3) + (col & 7);
    }

    __device__ __forceinline__ __nv_bfloat16* ptr(int row, int col) const {
        return &base[row * kStateDim + swizzled_col(row, col)];
    }
};

template <int THREADS>
__device__ __forceinline__ void issue_load_w_bf16(Bf16WView view,
                                                  const __nv_bfloat16* gmem_base_row0,
                                                  int64_t gmem_row_stride, int tid) {
    constexpr int ELEMS_PER_COPY = 8;
    constexpr int COPIES_PER_ROW = kStateDim / ELEMS_PER_COPY;
    constexpr int COPIES         = BT * COPIES_PER_ROW;
#pragma unroll
    for (int copy = tid; copy < COPIES; copy += THREADS) {
        const int row = copy / COPIES_PER_ROW;
        const int col = (copy - row * COPIES_PER_ROW) * ELEMS_PER_COPY;
        cp_async<16>(view.ptr(row, col),
                     gmem_base_row0 + static_cast<int64_t>(row) * gmem_row_stride + col);
    }
}

// K is staged in native BF16 because it is already representable exactly as
// a TF32 operand. Within each group of eight token rows, physical rows are
// ordered {0,4,1,5,2,6,3,7}. ldmatrix.x2.trans then packs token columns
// {t,t+4} into the two halves of each register, matching the TF32 A fragment.
struct Bf16KView {
    __nv_bfloat16* __restrict__ base;

    __device__ __forceinline__ int physical_row(int logical_row) const {
        const int row8 = logical_row & 7;
        return (logical_row & ~7) + ((row8 & 3) << 1) + (row8 >> 2);
    }

    __device__ __forceinline__ int swizzled_col(int physical_row_, int col) const {
        return (col & ~63) + ((((col & 63) >> 3) ^ (physical_row_ & 7)) << 3) + (col & 7);
    }

    __device__ __forceinline__ __nv_bfloat16* logical_ptr(int row, int col) const {
        const int row_phys = physical_row(row);
        return &base[row_phys * kStateDim + swizzled_col(row_phys, col)];
    }

    __device__ __forceinline__ __nv_bfloat16* physical_ptr(int row, int col) const {
        return &base[row * kStateDim + swizzled_col(row, col)];
    }
};

template <int THREADS>
__device__ __forceinline__ void issue_load_k_bf16(Bf16KView view,
                                                  const __nv_bfloat16* gmem_base_row0,
                                                  int64_t gmem_row_stride, int tid) {
    constexpr int ELEMS_PER_COPY = 8;
    constexpr int COPIES_PER_ROW = kStateDim / ELEMS_PER_COPY;
    constexpr int COPIES         = BT * COPIES_PER_ROW;
#pragma unroll
    for (int copy = tid; copy < COPIES; copy += THREADS) {
        const int row = copy / COPIES_PER_ROW;
        const int col = (copy - row * COPIES_PER_ROW) * ELEMS_PER_COPY;
        cp_async<16>(view.logical_ptr(row, col),
                     gmem_base_row0 + static_cast<int64_t>(row) * gmem_row_stride + col);
    }
}

// The narrow geometry targets two 128-register CTAs per SM. The wide geometry
// uses one 512-thread CTA; both expose 16 resident warps without local spills.
template <int NStrip>
__launch_bounds__(kernel_dims<NStrip>::THREADS, kernel_dims<NStrip>::MIN_BLOCKS) __global__
    void state_passing_kernel(const __nv_bfloat16* __restrict__ W_in,
                              const __nv_bfloat16* __restrict__ U_in,
                              const __nv_bfloat16* __restrict__ k_in,
                              const float* __restrict__ g_cumsum, const float* state_in,
                              __nv_bfloat16* __restrict__ v_new,
                              __nv_bfloat16* __restrict__ h_chunk, float* state_out,
                              head_map qk_map, int chunks) {
    using D                         = kernel_dims<NStrip>;
    using L                         = smem_layout<NStrip>;
    constexpr int N_STRIP_PER_BLOCK = D::N_STRIP_PER_BLOCK;
    constexpr int D_STRIPS          = D::D_STRIPS;
    constexpr int BT_SPLITS         = D::BT_SPLITS;
    constexpr int THREADS_K         = D::THREADS;
    constexpr int W_STRIDE          = L::W_STRIDE;

    constexpr int BT_PER_WARP    = BT / BT_SPLITS;
    constexpr int S_PER_WARP     = kStateDim / BT_SPLITS;
    constexpr int M_TILES_MM1_PW = BT_PER_WARP / MMA_M;
    constexpr int M_TILES_H_PW   = S_PER_WARP / MMA_M;
    constexpr int M_TILES_H_GLOB = kStateDim / MMA_M;
    constexpr int K_TILES_MM2    = BT / MMA_K;

    static_assert(M_TILES_H_PW >= 1, "S_PER_WARP must yield >= 1 M-tile per warp");
    static_assert(M_TILES_MM1_PW >= 1, "BT_PER_WARP must yield >= 1 M-tile per warp");
    static_assert(M_TILES_H_GLOB == BT_SPLITS * M_TILES_H_PW,
                  "BT_SPLITS partition of state dim must be exact");
    static_assert(W_STRIDE >= 16, "SmemTile<W_STRIDE> requires stride >= 16");

    constexpr int SNAP_K_ROWS           = L::SNAP_K_ROWS;
    constexpr int N_SNAP_ITERS          = L::N_SNAP_ITERS;
    constexpr int K_TILES_PER_SNAP_ITER = SNAP_K_ROWS / MMA_K;

    static_assert(N_SNAP_ITERS == BT_SPLITS, "design assumes one snap iter per s_idx");

    constexpr int W_STORAGE_FLT = L::W_STORAGE_FLT;
    constexpr int UVD_FLT       = L::UVD_FLT;
    constexpr int K_STORAGE_FLT = L::K_STORAGE_FLT;
    constexpr int SNAP_FLT      = L::SNAP_FLT;
    constexpr int N_SNAP_BUF    = L::N_SNAP_BUF;

    // Smem partition. U and vd alias (`uvd_smem`) in disjoint phases. snap
    // is per-sit so the chunk-start unified scatter feeds every sit's MM1.
    extern __shared__ float smem[];
    auto* const W_smem     = reinterpret_cast<__nv_bfloat16*>(smem); // W_LOAD_BF16
    float* const uvd_smem  = smem + W_STORAGE_FLT;                   // UVD_FLT
    auto* const k_smem     = reinterpret_cast<__nv_bfloat16*>(uvd_smem + UVD_FLT);
    float* const snap_smem = uvd_smem + UVD_FLT + K_STORAGE_FLT; // SNAP_FLT * N_SNAP_BUF
    float* const g_smem    = snap_smem + SNAP_FLT * N_SNAP_BUF;  // BT

    Bf16WView W_view{W_smem};
    SmemTile<N_STRIP_PER_BLOCK> vd_view{uvd_smem};
    Bf16KView k_view{k_smem};
    SmemTile<N_STRIP_PER_BLOCK> U_view{uvd_smem};
    // One SnapView per sit so each owning warp scatters into a unique buffer
    // (Phase 2.a unified-scatter design). The array is sized on N_SNAP_BUF
    // (= N_SNAP_ITERS) rather than hard-coded.
    SnapView snap_views[N_SNAP_BUF];
#pragma unroll
    for (int b_ = 0; b_ < N_SNAP_BUF; ++b_) {
        snap_views[b_] = SnapView{snap_smem + b_ * SNAP_FLT};
    }

    // Block / lane indexing.
    //   grid.x = hd in [0, H_v*D_STRIPS).
    //   warp = dt_idx * BT_SPLITS + s_idx.
    const int tid    = static_cast<int>(threadIdx.x);
    const int lane   = tid & (kWarpSize - 1);
    const int warp   = tid / kWarpSize;
    const int lane_g = lane >> 2;
    const int lane_t = lane & 3;

    const int s_idx        = warp % BT_SPLITS;
    const int dt_idx       = warp / BT_SPLITS;
    const int warp_d_local = dt_idx * MMA_N;

    const int hd            = static_cast<int>(blockIdx.x);
    const int h_v           = hd / D_STRIPS;
    const int strip_idx     = hd - h_v * D_STRIPS;
    const int d_off         = strip_idx * N_STRIP_PER_BLOCK;
    const int warp_d_global = d_off + warp_d_local;
    const std::int64_t H_v  = qk_map.H_v;

    // === Phase 0: load state_in (AR-transposed) -> per-warp h_frag ===
    float h_frag[M_TILES_H_PW][4];
    {
        const int64_t st_base = static_cast<int64_t>(h_v) * kStateDim * kStateDim;
        const int row_off     = s_idx * S_PER_WARP;
#pragma unroll
        for (int m = 0; m < M_TILES_H_PW; ++m) {
            const int row_g0 = row_off + m * MMA_M + lane_g;
            const int row_g1 = row_g0 + 8;
            const int col_d0 = warp_d_global + 2 * lane_t;
            const int col_d1 = col_d0 + 1;
            h_frag[m][0] =
                load_ldg<float>(state_in + st_base + (int64_t)col_d0 * kStateDim + row_g0);
            h_frag[m][1] =
                load_ldg<float>(state_in + st_base + (int64_t)col_d1 * kStateDim + row_g0);
            h_frag[m][2] =
                load_ldg<float>(state_in + st_base + (int64_t)col_d0 * kStateDim + row_g1);
            h_frag[m][3] =
                load_ldg<float>(state_in + st_base + (int64_t)col_d1 * kStateDim + row_g1);
        }
    }

    // Chunk 0 is staged before the loop. Later W/K async groups and converted
    // U are issued together at the preceding chunk's Phase-E boundary. All
    // global bases advance additively across the sequential chunk loop.
    const int64_t W_stride        = H_v * kStateDim;
    const int64_t k_stride        = static_cast<int64_t>(qk_map.H_qk) * kStateDim;
    const int64_t W_chunk_stride  = (int64_t)BT * W_stride;
    const int64_t k_chunk_stride  = (int64_t)BT * k_stride;
    const int64_t hc_chunk_stride = H_v * kStateDim * kStateDim;
    const int64_t vn_stride       = W_stride;
    const int64_t vn_chunk_stride = (int64_t)BT * vn_stride;
    const int64_t g_chunk_step    = (int64_t)BT * H_v;

    const int64_t W_block_base  = static_cast<int64_t>(h_v) * kStateDim;
    const int64_t k_block_base  = static_cast<int64_t>(qk_map.qk_head(h_v)) * kStateDim;
    const int64_t hc_block_base = static_cast<int64_t>(h_v) * kStateDim * kStateDim;
    const int64_t vn_block_base = static_cast<int64_t>(h_v) * kStateDim;
    const int64_t g_block_base  = h_v;
    const int64_t g_thread_base = g_block_base + (int64_t)tid * H_v;

    // W/K stay native BF16 in shared memory. W is committed first for MM1;
    // K is the later group and remains in flight until MM2. U is expanded
    // synchronously while both async groups make progress.
    {
        issue_load_w_bf16<THREADS_K>(W_view, W_in + W_block_base, W_stride, tid);
        cp_commit();
        issue_load_k_bf16<THREADS_K>(k_view, k_in + k_block_base, k_stride, tid);
        cp_commit();
        issue_load_bf16_to_float_vec4<BT, N_STRIP_PER_BLOCK, THREADS_K>(
            U_view, U_in + W_block_base + d_off, W_stride, tid);
    }

    // === Main chunk loop ===
    int64_t W_base      = W_block_base;
    int64_t k_base      = k_block_base;
    int64_t hc_base     = hc_block_base;
    int64_t vn_base     = vn_block_base;
    int64_t g_cs_offset = 0;
    for (int chunk = 0; chunk < chunks; ++chunk) {
        const int64_t W_base_next = W_base + W_chunk_stride;
        const int64_t k_base_next = k_base + k_chunk_stride;

        // === Phase A: drain W + scatter h_frag to snap ===
        //
        // Phase 2.a unified-scatter: every warp scatters its owned h_frag into
        // snap_views[s_idx] BEFORE the Phase A drain sync. The single sync
        // below covers W/U/g_smem AND snap visibility, so Phase B no longer
        // needs per-sit scatter+sync.
        if (tid < BT) { g_smem[tid] = g_cumsum[g_thread_base + g_cs_offset]; }

        {
            SnapView snap = snap_views[s_idx];
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

        cp_wait<1>();    // drain W; leave K in flight
        __syncthreads(); // gates W/U/g_smem STS and scatter visibility

        // === Phase B: per-sit coop_write + matmul1 (no per-sit scatter sync) ===
        //
        // The unified scatter in Phase A populated snap_views[s_idx] for all
        // s_idx in one shot, so each sit just consumes snap[sit] without
        // re-scattering or syncing. We retain the per-sit interleave between
        // coop_write (LDS+STG to gmem) and mma (LDS) because that was the
        // crucial latency-hiding pattern in the per-sit baseline -- moving
        // all coop_writes ahead of all mmas costs ~15 us on L=4096.
        float vnew_frag[M_TILES_MM1_PW][4] = {};

#pragma unroll
        for (int sit = 0; sit < N_SNAP_ITERS; ++sit) {
            const int k_row_off = sit * SNAP_K_ROWS;
            SnapView snap       = snap_views[sit];

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
                        &h_chunk[hc_base + (int64_t)d_global * kStateDim + k_row_off + k_off];
                    store_vec(out, __floats2bfloat162_rn(val.x, val.y));
                    store_vec(out + 2, __floats2bfloat162_rn(val.z, val.w));
                }
            }

            // W's producer-interleaved BF16 layout lets ldmatrix.x2 form the
            // four exact FP32/TF32 A operands. The FP32 state snapshot uses
            // ldmatrix.x2 as a b16 transport for the two B operand bit patterns.
            const int lane_in_8   = lane & 7;
            const int matrix_half = (lane >> 3) & 1;
#pragma unroll
            for (int kt = 0; kt < K_TILES_PER_SNAP_ITER; ++kt) {
                const int W_k_local = k_row_off + kt * MMA_K;
                const int snap_k    = kt * MMA_K;

                const int b_row       = warp_d_local + lane_in_8;
                const int b_col       = snap_k + matrix_half * 4;
                const unsigned b_addr = smem_addr(&snap.at(b_row, b_col));
                unsigned ub0, ub1;
                ldmatrix_x2(ub0, ub1, b_addr);

#pragma unroll
                for (int m_mm1 = 0; m_mm1 < M_TILES_MM1_PW; ++m_mm1) {
                    const int row_base    = s_idx * BT_PER_WARP + m_mm1 * MMA_M;
                    const int a_row       = row_base + matrix_half * 8 + lane_in_8;
                    const unsigned a_addr = smem_addr(W_view.ptr(a_row, W_k_local));
                    unsigned packed0, packed1;
                    ldmatrix_x2(packed0, packed1, a_addr);
                    unsigned ua0, ua1, ua2, ua3;
                    unpack_bf16x2_to_fp32_bits(packed0, ua0, ua2);
                    unpack_bf16x2_to_fp32_bits(packed1, ua1, ua3);

                    mma_tf32_bits(vnew_frag[m_mm1][0], vnew_frag[m_mm1][1], vnew_frag[m_mm1][2],
                                  vnew_frag[m_mm1][3], ua0, ua1, ua2, ua3, ub0, ub1);
                }
            }
        }

        // === Phase C: subtract converted FP32 U ===
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
        const float g_C = g_smem[BT - 1];
        float gamma_C   = 0.0f;
        if (lane == 0) { gamma_C = exp2_approx(g_C * kLog2E); }
        gamma_C = __shfl_sync(0xffffffffU, gamma_C, 0);

#pragma unroll
        for (int m_mm1 = 0; m_mm1 < M_TILES_MM1_PW; ++m_mm1) {
            const int row_g0 = s_idx * BT_PER_WARP + m_mm1 * MMA_M + lane_g;
            const int row_g1 = row_g0 + 8;
            const int col_d0 = warp_d_global + 2 * lane_t;

            float dec_top = 0.0f;
            float dec_bot = 0.0f;
            if (lane_t == 0) {
                dec_top = exp2_approx((g_C - g_smem[row_g0]) * kLog2E);
                dec_bot = exp2_approx((g_C - g_smem[row_g1]) * kLog2E);
            }
            const int decay_src_lane = lane & ~3;
            dec_top                  = __shfl_sync(0xffffffffU, dec_top, decay_src_lane);
            dec_bot                  = __shfl_sync(0xffffffffU, dec_bot, decay_src_lane);

            const float v0 = vnew_frag[m_mm1][0];
            const float v1 = vnew_frag[m_mm1][1];
            const float v2 = vnew_frag[m_mm1][2];
            const float v3 = vnew_frag[m_mm1][3];

            const __nv_bfloat162 out0 = __floats2bfloat162_rn(v0, v1);
            const __nv_bfloat162 out1 = __floats2bfloat162_rn(v2, v3);
            store_vec(&v_new[vn_base + (int64_t)row_g0 * vn_stride + col_d0], out0);
            store_vec(&v_new[vn_base + (int64_t)row_g1 * vn_stride + col_d0], out1);

            const int row_g0_loc = s_idx * BT_PER_WARP + m_mm1 * MMA_M + lane_g;
            const int row_g1_loc = row_g0_loc + 8;
            const int col_d0_loc = warp_d_local + 2 * lane_t;
            store_vec(&vd_view.at(row_g0_loc, col_d0_loc), make_float2(v0 * dec_top, v1 * dec_top));
            store_vec(&vd_view.at(row_g1_loc, col_d0_loc), make_float2(v2 * dec_bot, v3 * dec_bot));
        }

#pragma unroll
        for (int m = 0; m < M_TILES_H_PW; ++m) {
#pragma unroll
            for (int e = 0; e < 4; ++e) { h_frag[m][e] *= gamma_C; }
        }

        // Drain native-BF16 K, then make K and Phase-D v_decay visible to MM2.
        cp_wait<0>();
        __syncthreads();

        // === Phase E: matmul2 over the full chunk of k ===
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
                const int col_a_base = s_idx * S_PER_WARP + m * MMA_M;
                const int a_row      = k_off_local + (lane & 7);
                const int a_col      = col_a_base + ((lane >> 3) & 1) * 8;
                unsigned packed0, packed1;
                ldmatrix_x2_t(packed0, packed1, smem_addr(k_view.physical_ptr(a_row, a_col)));
                unsigned ua0, ua1, ua2, ua3;
                unpack_bf16x2_to_fp32_bits(packed0, ua0, ua2);
                unpack_bf16x2_to_fp32_bits(packed1, ua1, ua3);

                mma_tf32_bits(h_frag[m][0], h_frag[m][1], h_frag[m][2], h_frag[m][3], ua0, ua1, ua2,
                              ua3, __float_as_uint(b0), __float_as_uint(b1));
            }
        }

        __syncthreads(); // gates MM2 before the next chunk overwrites W/K/U

        // The next chunk repeats the W-then-K async group order used by the
        // prologue. Its Phase A drains W only; Phase E drains K.
        if (chunk + 1 < chunks) {
            issue_load_w_bf16<THREADS_K>(W_view, W_in + W_base_next, W_stride, tid);
            cp_commit();
            issue_load_k_bf16<THREADS_K>(k_view, k_in + k_base_next, k_stride, tid);
            cp_commit();
            issue_load_bf16_to_float_vec4<BT, N_STRIP_PER_BLOCK, THREADS_K>(
                U_view, U_in + W_base_next + d_off, W_stride, tid);
        }

        // Advance loop-carried bases for the next chunk.
        W_base = W_base_next;
        k_base = k_base_next;
        hc_base += hc_chunk_stride;
        vn_base += vn_chunk_stride;
        g_cs_offset += g_chunk_step;
    }

    // === Phase Z: store h_frag -> state_out (AR-transposed) ===
    const int64_t st_base = static_cast<int64_t>(h_v) * kStateDim * kStateDim;

#pragma unroll
    for (int m = 0; m < M_TILES_H_PW; ++m) {
        const int k_g0 = s_idx * S_PER_WARP + m * MMA_M + lane_g;
        const int k_g1 = k_g0 + 8;
        const int d0   = warp_d_global + 2 * lane_t;
        const int d1   = d0 + 1;
        state_out[st_base + (int64_t)d0 * kStateDim + k_g0] = h_frag[m][0];
        state_out[st_base + (int64_t)d1 * kStateDim + k_g0] = h_frag[m][1];
        state_out[st_base + (int64_t)d0 * kStateDim + k_g1] = h_frag[m][2];
        state_out[st_base + (int64_t)d1 * kStateDim + k_g1] = h_frag[m][3];
    }
}

} // namespace ninfer::ops::detail::gated_delta_net::chunked::state_passing
