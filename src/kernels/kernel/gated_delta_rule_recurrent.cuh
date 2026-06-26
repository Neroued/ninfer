#pragma once

#include "kernels/kernel/gdn_common.cuh"

#include <cstdint>

namespace qus::kernels {

inline constexpr int kGdnDvPerWarp = 4;
inline constexpr int kGdnNumWarps  = 4;
inline constexpr int kGdnBlockDv   = kGdnNumWarps * kGdnDvPerWarp;

template <int S, int DQK_PER_LANE, int ACTIVE_LANES>
__device__ __forceinline__ void gdn_load_qk_lane(float (&reg)[DQK_PER_LANE], const float* base,
                                                 int lane, std::uint32_t dqk_base) {
    if constexpr (S < WARP_SIZE) {
        reg[0] = (lane < ACTIVE_LANES) ? base[lane] : 0.0f;
    } else {
        cuda_memcpy_1<DQK_PER_LANE * sizeof(float)>(reg, base + dqk_base);
    }
}

template <int S, int DQK_PER_LANE, int ACTIVE_LANES>
__device__ __forceinline__ void gdn_store_qk_lane(const float (&reg)[DQK_PER_LANE], float* base,
                                                  int lane, std::uint32_t dqk_base) {
    if constexpr (S < WARP_SIZE) {
        if (lane < ACTIVE_LANES) { base[lane] = reg[0]; }
    } else {
        cuda_memcpy_1<DQK_PER_LANE * sizeof(float)>(base + dqk_base, reg);
    }
}

template <int S>
__global__ void __launch_bounds__(WARP_SIZE* kGdnNumWarps, 2)
    gated_delta_rule_recurrent_kernel(const float* __restrict__ q, const float* __restrict__ k,
                                      const float* __restrict__ v, const float* __restrict__ g,
                                      const float* __restrict__ beta, float* __restrict__ ssm_state,
                                      float* __restrict__ out, std::int64_t T, head_map heads,
                                      float scale) {
    constexpr int active_lanes  = (S < WARP_SIZE) ? S : WARP_SIZE;
    constexpr int d_qk_per_lane = S / active_lanes;
    static_assert(S % active_lanes == 0, "S must be a multiple of active_lanes");
    static_assert(S % kGdnBlockDv == 0, "S must be a multiple of kGdnBlockDv");

    const int lane           = threadIdx.x;
    const int warp_id        = threadIdx.y;
    const std::uint32_t h_v  = static_cast<std::uint32_t>(heads.cta_h_v(blockIdx.x));
    const std::uint32_t h_qk = static_cast<std::uint32_t>(heads.qk_head(static_cast<int>(h_v)));

    const std::uint32_t dv_base =
        static_cast<std::uint32_t>(blockIdx.z * kGdnBlockDv + warp_id * kGdnDvPerWarp);
    const std::uint32_t dqk_base = static_cast<std::uint32_t>(lane * d_qk_per_lane);

    float* state_h = ssm_state + static_cast<std::int64_t>(h_v) * S * S;

    __align__(16) float s_tile[kGdnDvPerWarp][d_qk_per_lane];
#pragma unroll
    for (int r = 0; r < kGdnDvPerWarp; ++r) {
        gdn_load_qk_lane<S, d_qk_per_lane, active_lanes>(
            s_tile[r], state_h + static_cast<std::int64_t>(dv_base + r) * S, lane, dqk_base);
    }

    __align__(16) float k_reg[d_qk_per_lane];
    gdn_load_qk_lane<S, d_qk_per_lane, active_lanes>(k_reg, k + static_cast<std::int64_t>(h_qk) * S,
                                                     lane, dqk_base);

    for (std::int64_t t = 0; t < T; ++t) {
        const float* v_t          = v + (t * heads.H_v + h_v) * S;
        const std::int64_t gb_off = t * heads.H_v + h_v;
        const float beta_val      = beta[gb_off];
        const float alpha         = expf(g[gb_off]);

        float v_local = 0.0f;
        if (lane < kGdnDvPerWarp) { v_local = v_t[dv_base + lane]; }

#pragma unroll
        for (int r = 0; r < kGdnDvPerWarp; ++r) {
            float partial = 0.0f;
#pragma unroll
            for (int c = 0; c < d_qk_per_lane; ++c) { partial += s_tile[r][c] * k_reg[c]; }
            partial = warp_reduce_sum<WARP_SIZE>(partial);

            const float v_r   = __shfl_sync(0xffffffff, v_local, r, WARP_SIZE);
            const float delta = beta_val * (v_r - alpha * partial);

#pragma unroll
            for (int c = 0; c < d_qk_per_lane; ++c) {
                s_tile[r][c] = alpha * s_tile[r][c] + delta * k_reg[c];
            }
        }

        if (t + 1 < T) {
            gdn_load_qk_lane<S, d_qk_per_lane, active_lanes>(
                k_reg, k + ((t + 1) * heads.H_qk + h_qk) * S, lane, dqk_base);
        }

        __align__(16) float q_reg[d_qk_per_lane];
        gdn_load_qk_lane<S, d_qk_per_lane, active_lanes>(q_reg, q + (t * heads.H_qk + h_qk) * S,
                                                         lane, dqk_base);

        float attn_val = 0.0f;
#pragma unroll
        for (int r = 0; r < kGdnDvPerWarp; ++r) {
            float partial = 0.0f;
#pragma unroll
            for (int c = 0; c < d_qk_per_lane; ++c) { partial += s_tile[r][c] * q_reg[c]; }
            partial = warp_reduce_sum<WARP_SIZE>(partial);
            if (lane == r) { attn_val = partial; }
        }

        if (lane < kGdnDvPerWarp) {
            out[(t * heads.H_v + h_v) * S + dv_base + lane] = attn_val * scale;
        }
    }

#pragma unroll
    for (int r = 0; r < kGdnDvPerWarp; ++r) {
        gdn_store_qk_lane<S, d_qk_per_lane, active_lanes>(
            s_tile[r], state_h + static_cast<std::int64_t>(dv_base + r) * S, lane, dqk_base);
    }
}

} // namespace qus::kernels
