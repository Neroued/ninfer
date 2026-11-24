#pragma once
#include "ops/linear_attention/kimi_delta_attention/kernels/transition.cuh"

namespace ninfer::ops::detail::kimi_delta_attention {
template <int Steps, int Rows>
__global__ __launch_bounds__(128, 2) void staged_record_kernel(Arguments a,
                                                               const std::int64_t* sources,
                                                               const std::int32_t* lengths,
                                                               RecurrenceRecords records) {
    const int lane = threadIdx.x, warp = threadIdx.y;
    const int head = blockIdx.x, batch = blockIdx.y, tile = blockIdx.z;
    const int qhead = head / (a.value_heads / a.qk_heads), dk = lane * 4;
    const int dv    = tile * (4 * Rows) + warp * Rows;
    const int valid = lengths ? lengths[batch] : a.tokens;
    const auto state_base =
        sources[batch] / sizeof(float) + static_cast<std::int64_t>(head) * 128 * 128;
    __align__(16) float state[Rows][4];
    load_state(state, a.state_in + state_base, dv, dk);

    __shared__ struct {
        float key[Steps][128], query[Steps][128], alpha[Steps][128], value[Steps][128], beta[Steps];
    } stage;

    float bias[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) bias[j] = a.dt_bias[head * 128 + dk + j];
    float a_scale = lane == 0 ? expf(a.a_log[head]) : 0.0F;
    a_scale       = __shfl_sync(common::kFullWarpMask, a_scale, 0);
    for (int begin = 0; begin < valid; begin += Steps) {
#pragma unroll
        for (int offset = warp; offset < Steps; offset += 4) {
            const int t = begin + offset;
            if (t < valid) {
                const auto col = static_cast<std::int64_t>(batch) * a.tokens + t;
                const auto qo  = (col * a.qk_heads + qhead) * 128,
                           vo  = (col * a.value_heads + head) * 128;
                float key[4], query[4], gate[4], value[4], decay[4], alpha[4];
                common::load_bf16x4(key, a.k + qo + dk);
                common::load_bf16x4(query, a.q + qo + dk);
                common::load_bf16x4(gate, a.g + vo + dk);
                common::load_bf16x4(value, a.v + vo + dk);
                if (tile == 0) {
                    if (head % (a.value_heads / a.qk_heads) == 0 && records.key != a.k)
                        common::store_vec(records.key + qo + dk,
                                          common::load_vec<uint2>(a.k + qo + dk));
                    if (records.value != a.v)
                        common::store_vec(records.value + vo + dk,
                                          common::load_vec<uint2>(a.v + vo + dk));
                }
                normalize_qk(key, lane);
                normalize_qk(query, lane);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    decay[j] =
                        a.lower_bound * common::sigmoid_approx(a_scale * (gate[j] + bias[j]));
                    alpha[j] = common::exp_approx_ftz(decay[j]);
                }
                common::store_vec(stage.key[offset] + dk, common::load_vec<float4>(key));
                common::store_vec(stage.query[offset] + dk, common::load_vec<float4>(query));
                common::store_vec(stage.alpha[offset] + dk, common::load_vec<float4>(alpha));
                common::store_vec(stage.value[offset] + dk, common::load_vec<float4>(value));
                if (tile == 0)
                    common::store_vec(records.log_decay + vo + dk, common::load_vec<float4>(decay));
                if (lane == 0) {
                    const float beta = common::sigmoid_approx(
                        __bfloat162float(a.beta[col * a.value_heads + head]));
                    stage.beta[offset] = beta;
                    if (tile == 0) records.beta[col * a.value_heads + head] = beta;
                }
            }
        }
        __syncthreads();
#pragma unroll
        for (int offset = 0; offset < Steps; ++offset)
            if (begin + offset < valid) {
                float key[4], query[4], alpha[4];
                common::store_vec(key, common::load_vec<float4>(stage.key[offset] + dk));
                common::store_vec(query, common::load_vec<float4>(stage.query[offset] + dk));
                common::store_vec(alpha, common::load_vec<float4>(stage.alpha[offset] + dk));
                const float value = lane < Rows ? stage.value[offset][dv + lane] : 0.0F;
                apply_transition(state, key, alpha, value, stage.beta[offset]);
                const auto col = static_cast<std::int64_t>(batch) * a.tokens + begin + offset;
                store_readout(state, query, a.out + (col * a.value_heads + head) * 128, dv, lane,
                              a.scale);
            }
        __syncthreads();
    }
    if (lane < Rows)
        for (int t = valid; t < a.tokens; ++t)
            a.out[((static_cast<std::int64_t>(batch) * a.tokens + t) * a.value_heads + head) * 128 +
                  dv + lane] = __float2bfloat16(0.0F);
}
} // namespace ninfer::ops::detail::kimi_delta_attention
