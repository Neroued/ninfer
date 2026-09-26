#pragma once

#include "ops/common/math.cuh"
#include "ops/linear_attention/kimi_delta_attention/launch.h"
#include "ops/linear_attention/kimi_delta_attention/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail::kimi_delta_attention {

inline constexpr int kDvPerWarp     = 4;
inline constexpr int kNumWarps      = 4;
inline constexpr int kBlockDv       = kNumWarps * kDvPerWarp;
inline constexpr int kQkPerLane     = kStateDim / kWarpSize;
inline constexpr float kQkL2NormEps = 1.0e-6F;

static_assert(kQkPerLane == 4);
static_assert(kStateDim % kBlockDv == 0);

struct alignas(16) GateStage {
    // Gate producers write a contiguous vector. State warps consume it with aligned float4 shared
    // loads, avoiding the producer-side conflicts caused by a warp-swizzled layout.
    float alpha[2][kStateDim];
    float beta[2];
    float a_log_exp;
};

__device__ __forceinline__ void load_bf16x4(float (&values)[kQkPerLane],
                                            const __nv_bfloat16* source) {
    // A built-in 64-bit vector keeps this as one coalesced LDG. Loading the equivalent struct lets
    // ptxas scalarize it into two stride-8 32-bit requests.
    const uint2 packed = load_vec<uint2>(source);
    const float2 low   = bf16x2_bits_to_float2(packed.x);
    const float2 high  = bf16x2_bits_to_float2(packed.y);
    values[0]          = low.x;
    values[1]          = low.y;
    values[2]          = high.x;
    values[3]          = high.y;
}

__device__ __forceinline__ void normalize_qk(float (&values)[kQkPerLane], int lane) {
    float sum = 0.0F;
#pragma unroll
    for (int index = 0; index < kQkPerLane; ++index) {
        sum = fmaf(values[index], values[index], sum);
    }
    sum       = warp_reduce_sum(sum);
    float inv = lane == 0 ? rsqrtf(sum + kQkL2NormEps) : 0.0F;
    inv       = __shfl_sync(kFullWarpMask, inv, 0);
#pragma unroll
    for (int index = 0; index < kQkPerLane; ++index) { values[index] *= inv; }
}

__device__ __forceinline__ void load_state(float (&state)[kDvPerWarp][kQkPerLane],
                                           const float* base, int dv_base, int dqk_base) {
#pragma unroll
    for (int row = 0; row < kDvPerWarp; ++row) {
        store_vec(state[row],
                  load_vec<float4>(base + static_cast<std::int64_t>(dv_base + row) * kStateDim +
                                   dqk_base));
    }
}

__device__ __forceinline__ void store_state(const float (&state)[kDvPerWarp][kQkPerLane],
                                            float* base, int dv_base, int dqk_base) {
#pragma unroll
    for (int row = 0; row < kDvPerWarp; ++row) {
        store_vec(base + static_cast<std::int64_t>(dv_base + row) * kStateDim + dqk_base,
                  load_vec<float4>(state[row]));
    }
}

__device__ __forceinline__ void apply_transition(float (&state)[kDvPerWarp][kQkPerLane],
                                                 const float (&key)[kQkPerLane],
                                                 const float (&alpha)[kQkPerLane],
                                                 float value_local, float beta) {
#pragma unroll
    for (int row = 0; row < kDvPerWarp; ++row) {
        float prediction = 0.0F;
#pragma unroll
        for (int column = 0; column < kQkPerLane; ++column) {
            prediction = fmaf(state[row][column] * alpha[column], key[column], prediction);
        }
        prediction        = warp_sum<kWarpSize>(prediction);
        const float value = __shfl_sync(kFullWarpMask, value_local, row);
        const float delta = beta * (value - prediction);
#pragma unroll
        for (int column = 0; column < kQkPerLane; ++column) {
            state[row][column] = fmaf(delta, key[column], alpha[column] * state[row][column]);
        }
    }
}

__device__ __forceinline__ void store_readout(const float (&state)[kDvPerWarp][kQkPerLane],
                                              const float (&query)[kQkPerLane],
                                              __nv_bfloat16* output, int dv_base, int lane,
                                              float scale) {
    float result = 0.0F;
#pragma unroll
    for (int row = 0; row < kDvPerWarp; ++row) {
        float partial = 0.0F;
#pragma unroll
        for (int column = 0; column < kQkPerLane; ++column) {
            partial = fmaf(state[row][column], query[column], partial);
        }
        partial = warp_sum<kWarpSize>(partial);
        if (lane == row) { result = partial; }
    }
    if (lane < kDvPerWarp) { output[dv_base + lane] = __float2bfloat16(result * scale); }
}

__device__ __forceinline__ void
run_recurrent_token(float (&state)[kDvPerWarp][kQkPerLane], const __nv_bfloat16* query_source,
                    const __nv_bfloat16* key_source, const __nv_bfloat16* value_source,
                    const __nv_bfloat16* gate_source, const __nv_bfloat16* beta_source,
                    const float* dt_bias_source, __nv_bfloat16* output, GateStage& gate_stage,
                    int stage, int lane, int thread, int dv_base, int dqk_base, float a_scale,
                    float lower_bound, float scale) {
    float key[kQkPerLane];
    float query[kQkPerLane];
    load_bf16x4(key, key_source + dqk_base);
    load_bf16x4(query, query_source + dqk_base);
    normalize_qk(key, lane);
    normalize_qk(query, lane);

    const float gate_raw = __bfloat162float(gate_source[thread]);
    const float log_alpha =
        lower_bound * sigmoid_approx(a_scale * (gate_raw + dt_bias_source[thread]));
    gate_stage.alpha[stage][thread] = exp_approx(log_alpha);
    if (thread == 0) { gate_stage.beta[stage] = sigmoid_approx(__bfloat162float(*beta_source)); }

    // Alpha and beta are produced once per CTA. Direct alternates buffers so the writes for a
    // later token cannot race state warps still consuming the preceding token.
    __syncthreads();

    float alpha[kQkPerLane];
    store_vec(alpha, load_vec<float4>(gate_stage.alpha[stage] + dqk_base));

    float value_local = 0.0F;
    if (lane < kDvPerWarp) { value_local = __bfloat162float(value_source[dv_base + lane]); }
    apply_transition(state, key, alpha, value_local, gate_stage.beta[stage]);
    store_readout(state, query, output, dv_base, lane, scale);
}

__global__ void __launch_bounds__(kWarpSize* kNumWarps, 2) recurrent_direct_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v, const __nv_bfloat16* __restrict__ g,
    const __nv_bfloat16* __restrict__ beta, const float* __restrict__ a_log,
    const float* __restrict__ dt_bias, const float* state_read, float* state_write,
    __nv_bfloat16* __restrict__ out, std::int32_t qk_heads, std::int32_t heads, std::int32_t width,
    float lower_bound, float scale) {
    __shared__ GateStage gate_stage;

    const int lane          = threadIdx.x;
    const int warp          = threadIdx.y;
    const int thread        = warp * kWarpSize + lane;
    const int head          = static_cast<int>(blockIdx.x);
    const int state_tile    = static_cast<int>(blockIdx.z);
    const int dv_base       = state_tile * kBlockDv + warp * kDvPerWarp;
    const int dqk_base      = lane * kQkPerLane;
    const auto state_offset = static_cast<std::int64_t>(head) * kStateDim * kStateDim;

    __align__(16) float state[kDvPerWarp][kQkPerLane];
    load_state(state, state_read + state_offset, dv_base, dqk_base);

    if (thread == 0) { gate_stage.a_log_exp = expf(a_log[head]); }
    __syncthreads();
    const float a_scale = gate_stage.a_log_exp;

    for (std::int32_t token = 0; token < width; ++token) {
        const std::int64_t vector_offset =
            (static_cast<std::int64_t>(token) * heads + head) * kStateDim;
        const std::int64_t query_offset =
            (static_cast<std::int64_t>(token) * qk_heads + head / (heads / qk_heads)) * kStateDim;
        run_recurrent_token(
            state, q + query_offset, k + query_offset, v + vector_offset, g + vector_offset,
            beta + static_cast<std::int64_t>(token) * heads + head,
            dt_bias + static_cast<std::int64_t>(head) * kStateDim, out + vector_offset, gate_stage,
            token & 1, lane, thread, dv_base, dqk_base, a_scale, lower_bound, scale);
    }

    store_state(state, state_write + state_offset, dv_base, dqk_base);
}

__global__ void __launch_bounds__(kWarpSize* kNumWarps, 2) recurrent_batch_update_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v, const __nv_bfloat16* __restrict__ g,
    const __nv_bfloat16* __restrict__ beta, const float* __restrict__ a_log,
    const float* __restrict__ dt_bias, float* __restrict__ states,
    const std::int32_t* __restrict__ state_slots, __nv_bfloat16* __restrict__ out,
    std::int32_t qk_heads, std::int32_t heads, std::int64_t state_slot_stride, float lower_bound,
    float scale) {
    __shared__ GateStage gate_stage;

    const int lane       = threadIdx.x;
    const int warp       = threadIdx.y;
    const int thread     = warp * kWarpSize + lane;
    const int head       = static_cast<int>(blockIdx.x);
    const int batch      = static_cast<int>(blockIdx.y);
    const int state_tile = static_cast<int>(blockIdx.z);
    const int dv_base    = state_tile * kBlockDv + warp * kDvPerWarp;
    const int dqk_base   = lane * kQkPerLane;
    const std::int64_t state_offset =
        static_cast<std::int64_t>(state_slots[batch]) * state_slot_stride +
        static_cast<std::int64_t>(head) * kStateDim * kStateDim;
    const std::int64_t vector_offset =
        (static_cast<std::int64_t>(batch) * heads + head) * kStateDim;

    __align__(16) float state[kDvPerWarp][kQkPerLane];
    load_state(state, states + state_offset, dv_base, dqk_base);

    if (thread == 0) { gate_stage.a_log_exp = expf(a_log[head]); }
    __syncthreads();

    const std::int64_t query_offset =
        (static_cast<std::int64_t>(batch) * qk_heads + head / (heads / qk_heads)) * kStateDim;
    run_recurrent_token(state, q + query_offset, k + query_offset, v + vector_offset,
                        g + vector_offset, beta + static_cast<std::int64_t>(batch) * heads + head,
                        dt_bias + static_cast<std::int64_t>(head) * kStateDim, out + vector_offset,
                        gate_stage, 0, lane, thread, dv_base, dqk_base, gate_stage.a_log_exp,
                        lower_bound, scale);

    store_state(state, states + state_offset, dv_base, dqk_base);
}

} // namespace ninfer::ops::detail::kimi_delta_attention
