#pragma once

#include "ops/common/numeric/convert.cuh"
#include "ops/common/cuda/reduce.cuh"
#include "ops/common/cuda/memory.cuh"
#include "ops/linear_attention/gated_delta_net/operands.h"
#include "ops/common/numeric/math.cuh"
#include <cuda_bf16.h>
#include <cstdint>

namespace ninfer::ops::detail::gated_delta_net {
struct HeadMap {
    int H_qk, H_v;
    common::FastDivisorU31 group;

    static HeadMap of(int qk, int value) {
        return {qk, value, common::FastDivisorU31::make(value / qk)};
    }

    __host__ __device__ int group_size() const { return H_v / H_qk; }

    __host__ __device__ int qk_head(int value_head) const {
        return static_cast<int>(group.divide(static_cast<std::uint32_t>(value_head)));
    }
};

inline constexpr int kDvPerWarp = 4;
inline constexpr int kNumWarps  = 4;
inline constexpr int kBlockDv   = kNumWarps * kDvPerWarp;
inline constexpr int kQkPerLane = kStateDim / common::kWarpSize;

static_assert(kStateDim % common::kWarpSize == 0);
static_assert(kQkPerLane == 4);
static_assert(kStateDim % kBlockDv == 0);

template <class StateT>
__device__ __forceinline__ void load_qk_lane(float (&reg)[kQkPerLane], const StateT* base,
                                             std::uint32_t dqk_base) {
    common::load_state_vec4(reg, base + dqk_base);
}

template <class StateT>
__device__ __forceinline__ void store_qk_lane(const float (&reg)[kQkPerLane], StateT* base,
                                              std::uint32_t dqk_base) {
    common::store_state_vec4(base + dqk_base, reg);
}

inline constexpr float kQkL2NormEps = 1.0e-6f;

struct RawQkLane {
    common::Bf16x4Pack bits;
    float value[kQkPerLane];
};

struct RawGatePair {
    float g;
    float beta;
};

__device__ __forceinline__ RawQkLane load_raw_qk_lane(const __nv_bfloat16* base,
                                                      std::uint32_t dqk_base) {
    RawQkLane out;
    out.bits        = common::load_vec<common::Bf16x4Pack>(base + dqk_base);
    const float2 lo = common::bf16x2_to_float2(out.bits.pair[0]);
    const float2 hi = common::bf16x2_to_float2(out.bits.pair[1]);
    out.value[0]    = lo.x;
    out.value[1]    = lo.y;
    out.value[2]    = hi.x;
    out.value[3]    = hi.y;
    return out;
}

template <bool Normalize>
__device__ __forceinline__ void normalize_qk_lane(float (&value)[kQkPerLane], int lane) {
    if constexpr (Normalize) {
        float sum = 0.0f;
#pragma unroll
        for (int i = 0; i < kQkPerLane; ++i) { sum += value[i] * value[i]; }
        sum       = common::warp_reduce_sum(sum);
        float inv = lane == 0 ? rsqrtf(sum + kQkL2NormEps) : 0.0f;
        inv       = __shfl_sync(common::kFullWarpMask, inv, 0);
#pragma unroll
        for (int i = 0; i < kQkPerLane; ++i) { value[i] *= inv; }
    }
}

__device__ __forceinline__ float load_value_lane(const __nv_bfloat16* base, int lane,
                                                 std::uint32_t dv_base) {
    return lane < kDvPerWarp ? __bfloat162float(base[dv_base + lane]) : 0.0f;
}

__device__ __forceinline__ RawGatePair load_source_gate(const float* g, const float* beta,
                                                        std::int64_t offset) {
    return {g[offset], beta[offset]};
}

template <bool Normalize>
__device__ __forceinline__ void readout_and_store(float (&state)[kDvPerWarp][kQkPerLane],
                                                  const __nv_bfloat16* query, __nv_bfloat16* output,
                                                  std::uint32_t dqk_base, std::uint32_t dv_base,
                                                  int lane, float scale) {
    RawQkLane q = load_raw_qk_lane(query, dqk_base);
    normalize_qk_lane<Normalize>(q.value, lane);

    float attn_val = 0.0f;
#pragma unroll
    for (int r = 0; r < kDvPerWarp; ++r) {
        float partial = 0.0f;
#pragma unroll
        for (int c = 0; c < kQkPerLane; ++c) { partial += state[r][c] * q.value[c]; }
        partial = common::warp_sum<common::kWarpSize>(partial);
        if (lane == r) { attn_val = partial; }
    }
    if (lane < kDvPerWarp) { output[dv_base + lane] = __float2bfloat16(attn_val * scale); }
}

struct RecurrentCoordinates {
    int lane;
    int warp;
    std::int32_t batch;
    std::int32_t layer;
    std::int32_t state_tile;
    std::uint32_t value_head;
    std::uint32_t qk_head;
    std::uint32_t dv_base;
    std::uint32_t dqk_base;
};

__device__ __forceinline__ RecurrentCoordinates make_coordinates(std::int32_t batch,
                                                                 std::int32_t layer,
                                                                 std::int32_t state_tile,
                                                                 HeadMap heads) {
    const int lane                 = threadIdx.x;
    const int warp                 = threadIdx.y;
    const std::uint32_t value_head = static_cast<std::uint32_t>(blockIdx.x);
    const std::uint32_t qk_head =
        static_cast<std::uint32_t>(heads.qk_head(static_cast<int>(value_head)));
    const std::uint32_t dv_base =
        static_cast<std::uint32_t>(state_tile * kBlockDv + warp * kDvPerWarp);
    return {lane,    warp,       batch,
            layer,   state_tile, value_head,
            qk_head, dv_base,    static_cast<std::uint32_t>(lane * kQkPerLane)};
}

template <class Effects, class Access>
__device__ __forceinline__ void apply_gdn_transition(float (&state)[kDvPerWarp][kQkPerLane],
                                                     const float (&key)[kQkPerLane], float v_local,
                                                     float g, float beta, const Access& access,
                                                     const RecurrentCoordinates& coord, int token) {
    const float alpha = expf(g);
    float delta_lane  = 0.0f;
    if constexpr (Effects::kRecordTransition) access.store_alpha(coord, token, alpha);

#pragma unroll
    for (int r = 0; r < kDvPerWarp; ++r) {
        float partial = 0.0f;
#pragma unroll
        for (int c = 0; c < kQkPerLane; ++c) { partial += state[r][c] * key[c]; }
        partial = common::warp_sum<common::kWarpSize>(partial);

        const float v_r   = __shfl_sync(0xffffffff, v_local, r, common::kWarpSize);
        const float delta = beta * (v_r - alpha * partial);
        if constexpr (Effects::kRecordTransition) {
            if (coord.lane == r) delta_lane = delta;
        }

#pragma unroll
        for (int c = 0; c < kQkPerLane; ++c) { state[r][c] = alpha * state[r][c] + delta * key[c]; }
    }
    if constexpr (Effects::kRecordTransition) {
        if (coord.lane < kDvPerWarp)
            access.store_delta_element(coord, token, coord.dv_base + coord.lane, delta_lane);
    }
}

struct OutputEffects {
    static constexpr bool kRecordTransition = false;
    template <class Access>
    __device__ __forceinline__ static void observe_key(const Access&, const RecurrentCoordinates&,
                                                       std::int32_t, const RawQkLane&) {}

    template <bool NormalizeInputs, class Access>
    __device__ __forceinline__ static void
    publish_output(float (&state)[kDvPerWarp][kQkPerLane], const Access& access,
                   const RecurrentCoordinates& coord, std::int32_t token) {
        readout_and_store<NormalizeInputs>(state, access.query_ptr(coord, token),
                                           access.output_ptr(coord, token), coord.dqk_base,
                                           coord.dv_base, coord.lane, access.scale);
    }
};

template <class StateT>
__device__ __forceinline__ void load_state_tile(float (&state)[kDvPerWarp][kQkPerLane],
                                                const StateT* base,
                                                const RecurrentCoordinates& coord) {
#pragma unroll
    for (int r = 0; r < kDvPerWarp; ++r) {
        load_qk_lane(state[r], base + static_cast<std::int64_t>(coord.dv_base + r) * kStateDim,
                     coord.dqk_base);
    }
}

template <class StateT>
__device__ __forceinline__ void store_state_tile(const float (&state)[kDvPerWarp][kQkPerLane],
                                                 StateT* base, const RecurrentCoordinates& coord) {
#pragma unroll
    for (int r = 0; r < kDvPerWarp; ++r) {
        store_qk_lane(state[r], base + static_cast<std::int64_t>(coord.dv_base + r) * kStateDim,
                      coord.dqk_base);
    }
}

template <bool NormalizeInputs, class Effects, bool PrefetchKey = true, class Access>
__device__ __forceinline__ void
run_recurrent_sequence(float (&state)[kDvPerWarp][kQkPerLane], const Access& access,
                       const RecurrentCoordinates& coord, std::int32_t valid) {
    if constexpr (!PrefetchKey) {
        for (std::int32_t token = 0; token < valid; ++token) {
            RawQkLane key = load_raw_qk_lane(access.key_ptr(coord, token), coord.dqk_base);
            Effects::observe_key(access, coord, token, key);
            normalize_qk_lane<NormalizeInputs>(key.value, coord.lane);
            const auto gate = access.load_gate(coord, token);
            const float value =
                load_value_lane(access.value_ptr(coord, token), coord.lane, coord.dv_base);
            apply_gdn_transition<Effects>(state, key.value, value, gate.g, gate.beta, access, coord,
                                          token);
            Effects::template publish_output<NormalizeInputs>(state, access, coord, token);
        }
        return;
    }
    RawQkLane key = load_raw_qk_lane(access.key_ptr(coord, 0), coord.dqk_base);
    Effects::observe_key(access, coord, 0, key);
    normalize_qk_lane<NormalizeInputs>(key.value, coord.lane);

    for (std::int32_t token = 0; token < valid; ++token) {
        const RawGatePair gate = access.load_gate(coord, token);
        const float value =
            load_value_lane(access.value_ptr(coord, token), coord.lane, coord.dv_base);

        apply_gdn_transition<Effects>(state, key.value, value, gate.g, gate.beta, access, coord,
                                      token);

        if (token + 1 < valid) {
            key = load_raw_qk_lane(access.key_ptr(coord, token + 1), coord.dqk_base);
            Effects::observe_key(access, coord, token + 1, key);
            normalize_qk_lane<NormalizeInputs>(key.value, coord.lane);
        }

        Effects::template publish_output<NormalizeInputs>(state, access, coord, token);
    }
}

template <class Access>
__device__ __forceinline__ void zero_output_suffix(const Access& access,
                                                   const RecurrentCoordinates& coord,
                                                   std::int32_t valid, std::int32_t width) {
    if (coord.lane < kDvPerWarp) {
        for (std::int32_t token = valid; token < width; ++token) {
            access.output_ptr(coord, token)[coord.dv_base + coord.lane] = __float2bfloat16(0.0f);
        }
    }
}

} // namespace ninfer::ops::detail::gated_delta_net
