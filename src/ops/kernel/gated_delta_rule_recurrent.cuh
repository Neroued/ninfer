#pragma once

#include "ops/kernel/gdn_common.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kGdnDvPerWarp = 4;
inline constexpr int kGdnNumWarps  = 4;
inline constexpr int kGdnBlockDv   = kGdnNumWarps * kGdnDvPerWarp;

template <int S, int DQK_PER_LANE, int ACTIVE_LANES>
__device__ __forceinline__ void gdn_load_qk_lane(float (&reg)[DQK_PER_LANE], const float* base,
                                                 int lane, std::uint32_t dqk_base) {
    if constexpr (S < kWarpSize) {
        reg[0] = (lane < ACTIVE_LANES) ? base[lane] : 0.0f;
    } else if constexpr (DQK_PER_LANE == 1) {
        reg[0] = base[dqk_base];
    } else if constexpr (DQK_PER_LANE == 2) {
        store_vec(reg, load_vec<float2>(base + dqk_base));
    } else if constexpr (DQK_PER_LANE == 4) {
        store_vec(reg, load_vec<float4>(base + dqk_base));
    } else {
        static_assert(DQK_PER_LANE <= 4, "unsupported GDN lane vector width");
    }
}

template <int S, int DQK_PER_LANE, int ACTIVE_LANES>
__device__ __forceinline__ void gdn_store_qk_lane(const float (&reg)[DQK_PER_LANE], float* base,
                                                  int lane, std::uint32_t dqk_base) {
    if constexpr (S < kWarpSize) {
        if (lane < ACTIVE_LANES) { base[lane] = reg[0]; }
    } else if constexpr (DQK_PER_LANE == 1) {
        base[dqk_base] = reg[0];
    } else if constexpr (DQK_PER_LANE == 2) {
        store_vec(base + dqk_base, load_vec<float2>(reg));
    } else if constexpr (DQK_PER_LANE == 4) {
        store_vec(base + dqk_base, load_vec<float4>(reg));
    } else {
        static_assert(DQK_PER_LANE <= 4, "unsupported GDN lane vector width");
    }
}

template <int S>
__global__ void __launch_bounds__(kWarpSize* kGdnNumWarps, 2)
    gated_delta_rule_recurrent_kernel(const float* __restrict__ q, const float* __restrict__ k,
                                      const float* __restrict__ v, const float* __restrict__ g,
                                      const float* __restrict__ beta, float* __restrict__ ssm_state,
                                      float* __restrict__ out, std::int64_t T, head_map heads,
                                      float scale) {
    constexpr int active_lanes  = (S < kWarpSize) ? S : kWarpSize;
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
            partial = warp_sum<kWarpSize>(partial);

            const float v_r   = __shfl_sync(0xffffffff, v_local, r, kWarpSize);
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
            partial = warp_sum<kWarpSize>(partial);
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

inline constexpr float kGdnQkL2NormEps = 1.0e-6f;

template <int S, int DQK_PER_LANE, int ACTIVE_LANES, bool NormalizeQK>
__device__ __forceinline__ void gdn_load_qk_lane_bf16(float (&reg)[DQK_PER_LANE],
                                                      const __nv_bfloat16* base, int lane,
                                                      std::uint32_t dqk_base) {
    if constexpr (S < kWarpSize) {
        reg[0] = (lane < ACTIVE_LANES) ? __bfloat162float(base[lane]) : 0.0f;
    } else {
#pragma unroll
        for (int i = 0; i < DQK_PER_LANE; ++i) { reg[i] = __bfloat162float(base[dqk_base + i]); }
    }

    if constexpr (NormalizeQK) {
        float sum = 0.0f;
#pragma unroll
        for (int i = 0; i < DQK_PER_LANE; ++i) { sum += reg[i] * reg[i]; }
        sum       = warp_reduce_sum(sum);
        float inv = lane == 0 ? rsqrtf(sum + kGdnQkL2NormEps) : 0.0f;
        inv       = __shfl_sync(kFullWarpMask, inv, 0);
#pragma unroll
        for (int i = 0; i < DQK_PER_LANE; ++i) { reg[i] *= inv; }
    }
}

// state_read / state_write are the read and write bases (fp32).
//   Spec (snapshot):    read from state_read[safe(*initial_slot)] slot, write per-token snapshots
//                       into state_write slots 0..T-1.
//   non-spec:           read from state_read (caller-resolved view), write the final running state
//                       into state_write (slot-0 view). Passing state_read == state_write is the
//                       in-place form; distinct views let prefix-append prefill read a committed
//                       snapshot slot and publish the running state to slot 0.
template <int HeadDim, bool Spec, bool NormalizeQK>
__global__ void __launch_bounds__(kWarpSize* kGdnNumWarps, 2)
    gated_delta_rule_recurrent_bf16_kernel(
        const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
        const __nv_bfloat16* __restrict__ v, const float* __restrict__ g,
        const float* __restrict__ beta, float* __restrict__ state_read,
        float* __restrict__ state_write, const std::int32_t* __restrict__ initial_slot,
        __nv_bfloat16* __restrict__ out, std::int64_t T, head_map heads, float scale,
        std::int64_t state_slot_stride, std::int32_t slots) {
    constexpr int active_lanes  = (HeadDim < kWarpSize) ? HeadDim : kWarpSize;
    constexpr int d_qk_per_lane = HeadDim / active_lanes;
    static_assert(HeadDim % active_lanes == 0, "HeadDim must be a multiple of active_lanes");
    static_assert(HeadDim % kGdnBlockDv == 0, "HeadDim must be a multiple of kGdnBlockDv");

    const int lane           = threadIdx.x;
    const int warp_id        = threadIdx.y;
    const std::uint32_t h_v  = static_cast<std::uint32_t>(heads.cta_h_v(blockIdx.x));
    const std::uint32_t h_qk = static_cast<std::uint32_t>(heads.qk_head(static_cast<int>(h_v)));

    const std::uint32_t dv_base =
        static_cast<std::uint32_t>(blockIdx.z * kGdnBlockDv + warp_id * kGdnDvPerWarp);
    const std::uint32_t dqk_base = static_cast<std::uint32_t>(lane * d_qk_per_lane);

    float* read_base = state_read;
    if constexpr (Spec) {
        const std::int32_t slot      = *initial_slot;
        const std::int32_t safe_slot = (slot >= 0 && slot < slots) ? slot : 0;
        read_base = state_read + static_cast<std::int64_t>(safe_slot) * state_slot_stride;
    }
    float* read_h = read_base + static_cast<std::int64_t>(h_v) * HeadDim * HeadDim;

    __align__(16) float s_tile[kGdnDvPerWarp][d_qk_per_lane];
#pragma unroll
    for (int r = 0; r < kGdnDvPerWarp; ++r) {
        gdn_load_qk_lane<HeadDim, d_qk_per_lane, active_lanes>(
            s_tile[r], read_h + static_cast<std::int64_t>(dv_base + r) * HeadDim, lane, dqk_base);
    }

    __align__(16) float k_reg[d_qk_per_lane];
    gdn_load_qk_lane_bf16<HeadDim, d_qk_per_lane, active_lanes, NormalizeQK>(
        k_reg, k + static_cast<std::int64_t>(h_qk) * HeadDim, lane, dqk_base);

    for (std::int64_t t = 0; t < T; ++t) {
        const __nv_bfloat16* v_t  = v + (t * heads.H_v + h_v) * HeadDim;
        const std::int64_t gb_off = t * heads.H_v + h_v;
        const float beta_val      = beta[gb_off];
        const float alpha         = expf(g[gb_off]);

        float v_local = 0.0f;
        if (lane < kGdnDvPerWarp) { v_local = __bfloat162float(v_t[dv_base + lane]); }

#pragma unroll
        for (int r = 0; r < kGdnDvPerWarp; ++r) {
            float partial = 0.0f;
#pragma unroll
            for (int c = 0; c < d_qk_per_lane; ++c) { partial += s_tile[r][c] * k_reg[c]; }
            partial = warp_sum<kWarpSize>(partial);

            const float v_r   = __shfl_sync(0xffffffff, v_local, r, kWarpSize);
            const float delta = beta_val * (v_r - alpha * partial);

#pragma unroll
            for (int c = 0; c < d_qk_per_lane; ++c) {
                s_tile[r][c] = alpha * s_tile[r][c] + delta * k_reg[c];
            }
        }

        if (t + 1 < T) {
            gdn_load_qk_lane_bf16<HeadDim, d_qk_per_lane, active_lanes, NormalizeQK>(
                k_reg, k + ((t + 1) * heads.H_qk + h_qk) * HeadDim, lane, dqk_base);
        }

        __align__(16) float q_reg[d_qk_per_lane];
        gdn_load_qk_lane_bf16<HeadDim, d_qk_per_lane, active_lanes, NormalizeQK>(
            q_reg, q + (t * heads.H_qk + h_qk) * HeadDim, lane, dqk_base);

        float attn_val = 0.0f;
#pragma unroll
        for (int r = 0; r < kGdnDvPerWarp; ++r) {
            float partial = 0.0f;
#pragma unroll
            for (int c = 0; c < d_qk_per_lane; ++c) { partial += s_tile[r][c] * q_reg[c]; }
            partial = warp_sum<kWarpSize>(partial);
            if (lane == r) { attn_val = partial; }
        }

        if (lane < kGdnDvPerWarp) {
            out[(t * heads.H_v + h_v) * HeadDim + dv_base + lane] =
                __float2bfloat16(attn_val * scale);
        }

        if constexpr (Spec) {
            float* snapshot_h = state_write + t * state_slot_stride +
                                static_cast<std::int64_t>(h_v) * HeadDim * HeadDim;
#pragma unroll
            for (int r = 0; r < kGdnDvPerWarp; ++r) {
                gdn_store_qk_lane<HeadDim, d_qk_per_lane, active_lanes>(
                    s_tile[r], snapshot_h + static_cast<std::int64_t>(dv_base + r) * HeadDim, lane,
                    dqk_base);
            }
        }
    }

    if constexpr (!Spec) {
        float* write_h = state_write + static_cast<std::int64_t>(h_v) * HeadDim * HeadDim;
#pragma unroll
        for (int r = 0; r < kGdnDvPerWarp; ++r) {
            gdn_store_qk_lane<HeadDim, d_qk_per_lane, active_lanes>(
                s_tile[r], write_h + static_cast<std::int64_t>(dv_base + r) * HeadDim, lane,
                dqk_base);
        }
    }
}

} // namespace ninfer::ops
