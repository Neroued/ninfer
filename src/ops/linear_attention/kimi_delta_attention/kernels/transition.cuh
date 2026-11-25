#pragma once

#include "ops/common/numeric/math.cuh"
#include "ops/common/numeric/convert.cuh"
#include "ops/common/cuda/reduce.cuh"
#include "ops/common/cuda/memory.cuh"
#include "ops/linear_attention/kimi_delta_attention/launch.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace ninfer::ops::detail::kimi_delta_attention {

inline constexpr int kDvPerWarp = 4;
inline constexpr int kNumWarps  = 4;
inline constexpr int kBlockDv   = kNumWarps * kDvPerWarp;
inline constexpr int kQkPerLane = kStateDim / common::kWarpSize;

static_assert(kQkPerLane == 4);
static_assert(kStateDim % kBlockDv == 0);

__device__ __forceinline__ void normalize_qk(float (&values)[kQkPerLane], int lane) {
    float sum = 0.0F;
#pragma unroll
    for (int index = 0; index < kQkPerLane; ++index) {
        sum = fmaf(values[index], values[index], sum);
    }
    sum       = common::warp_reduce_sum(sum);
    float inv = lane == 0 ? rsqrtf(sum + kQkL2NormEps) : 0.0F;
    inv       = __shfl_sync(common::kFullWarpMask, inv, 0);
#pragma unroll
    for (int index = 0; index < kQkPerLane; ++index) { values[index] *= inv; }
}

template <int Rows, class StateT>
__device__ __forceinline__ void load_state(float (&state)[Rows][kQkPerLane], const StateT* base,
                                           int dv_base, int dqk_base) {
#pragma unroll
    for (int row = 0; row < Rows; ++row) {
        common::load_state_vec4(
            state[row], base + static_cast<std::int64_t>(dv_base + row) * kStateDim + dqk_base);
    }
}

template <int Rows, class StateT>
__device__ __forceinline__ void store_state(const float (&state)[Rows][kQkPerLane], StateT* base,
                                            int dv_base, int dqk_base) {
#pragma unroll
    for (int row = 0; row < Rows; ++row) {
        common::store_state_vec4(
            base + static_cast<std::int64_t>(dv_base + row) * kStateDim + dqk_base, state[row]);
    }
}

template <bool Record, int Rows>
__device__ __forceinline__ float
apply_transition(float (&state)[Rows][kQkPerLane], const float (&key)[kQkPerLane],
                 const float (&alpha)[kQkPerLane], float value_local, float beta, int lane) {
    float delta_lane = 0.0F;
#pragma unroll
    for (int row = 0; row < Rows; ++row) {
        float prediction = 0.0F;
#pragma unroll
        for (int column = 0; column < kQkPerLane; ++column) {
            prediction = fmaf(state[row][column] * alpha[column], key[column], prediction);
        }
        prediction        = common::warp_sum<common::kWarpSize>(prediction);
        const float value = __shfl_sync(common::kFullWarpMask, value_local, row);
        const float delta = beta * (value - prediction);
        if constexpr (Record) {
            if (lane == row) delta_lane = delta;
        }
#pragma unroll
        for (int column = 0; column < kQkPerLane; ++column) {
            state[row][column] = fmaf(delta, key[column], alpha[column] * state[row][column]);
        }
    }
    return delta_lane;
}

template <int Rows>
__device__ __forceinline__ void
store_readout(const float (&state)[Rows][kQkPerLane], const float (&query)[kQkPerLane],
              __nv_bfloat16* output, int dv_base, int lane, float scale) {
    float result = 0.0F;
#pragma unroll
    for (int row = 0; row < Rows; ++row) {
        float partial = 0.0F;
#pragma unroll
        for (int column = 0; column < kQkPerLane; ++column) {
            partial = fmaf(state[row][column], query[column], partial);
        }
        partial = common::warp_sum<common::kWarpSize>(partial);
        if (lane == row) { result = partial; }
    }
    if (lane < Rows) { output[dv_base + lane] = __float2bfloat16(result * scale); }
}

template <bool Record, int Rows>
__device__ __forceinline__ void
run_recurrent_token(float (&state)[Rows][kQkPerLane], const __nv_bfloat16* query_source,
                    const __nv_bfloat16* key_source, const __nv_bfloat16* value_source,
                    const __nv_bfloat16* gate_source, const __nv_bfloat16* beta_source,
                    const float* dt_bias_source, __nv_bfloat16* output, int lane, int thread,
                    int dv_base, int dqk_base, float a_scale, float lower_bound, float scale,
                    float* delta_record, float* alpha_record) {
    float key[kQkPerLane];
    float query[kQkPerLane];
    common::load_bf16x4(key, key_source + dqk_base);
    common::load_bf16x4(query, query_source + dqk_base);
    normalize_qk(key, lane);
    normalize_qk(query, lane);

    float raw[kQkPerLane], bias[kQkPerLane], alpha[kQkPerLane];
    common::load_bf16x4(raw, gate_source + dqk_base);
#pragma unroll
    for (int c = 0; c < kQkPerLane; ++c) bias[c] = dt_bias_source[dqk_base + c];
#pragma unroll
    for (int c = 0; c < kQkPerLane; ++c) {
        const float log_alpha = lower_bound * common::sigmoid_approx(a_scale * (raw[c] + bias[c]));
        alpha[c] = common::exp_approx_ftz(log_alpha);
    }
    float beta = lane == 0 ? common::sigmoid_approx(__bfloat162float(*beta_source)) : 0.0F;
    beta       = __shfl_sync(common::kFullWarpMask, beta, 0);
    if constexpr (Record) {
        if (thread < common::kWarpSize && alpha_record)
            common::store_vec(alpha_record + dqk_base, common::load_vec<float4>(alpha));
    }

    float value_local = 0.0F;
    if (lane < Rows) { value_local = __bfloat162float(value_source[dv_base + lane]); }
    const float delta_lane = apply_transition<Record>(state, key, alpha, value_local, beta, lane);
    if constexpr (Record) {
        if (lane < Rows) delta_record[dv_base + lane] = delta_lane;
    }
    store_readout(state, query, output, dv_base, lane, scale);
}

} // namespace ninfer::ops::detail::kimi_delta_attention
