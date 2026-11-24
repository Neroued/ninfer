#pragma once

#include "ops/linear_attention/kimi_delta_attention/kernels/transition.cuh"

namespace ninfer::ops::detail::kimi_delta_attention {
template <bool Batched, bool Record, int Rows = kDvPerWarp>
__global__ void __launch_bounds__(common::kWarpSize* kNumWarps, 2)
    recurrent_kernel(Arguments args, const std::int64_t* source_offsets,
                     const std::int64_t* destination_offsets, const std::int32_t* valid_columns,
                     RecurrenceRecords records) {
    constexpr int kBlockDv = kNumWarps * Rows;
    const int lane = threadIdx.x, warp = threadIdx.y, thread = warp * common::kWarpSize + lane;
    const int head = blockIdx.x, batch = Batched ? blockIdx.y : 0, tile = blockIdx.z;
    const int dv_base = tile * kBlockDv + warp * Rows, dqk_base = lane * kQkPerLane;
    const int qhead       = head / (args.value_heads / args.qk_heads);
    const int valid       = valid_columns ? valid_columns[batch] : args.tokens;
    const auto state_head = static_cast<std::int64_t>(head) * kStateDim * kStateDim;
    const auto source     = state_head + (Batched ? source_offsets[batch] / sizeof(float) : 0);
    __align__(16) float state[Rows][kQkPerLane];
    load_state(state, args.state_in + source, dv_base, dqk_base);
    float a_scale = lane == 0 ? expf(args.a_log[head]) : 0.0F;
    a_scale       = __shfl_sync(common::kFullWarpMask, a_scale, 0);
    for (int t = 0; t < valid; ++t) {
        const auto column = static_cast<std::int64_t>(batch) * args.tokens + t;
        const auto vo     = (column * args.value_heads + head) * kStateDim;
        const auto qo     = (column * args.qk_heads + qhead) * kStateDim;
        const auto bo     = column * args.value_heads + head;
        if constexpr (Record) {
            if (tile == 0 && head % (args.value_heads / args.qk_heads) == 0 &&
                records.key != args.k)
                records.key[qo + thread] = args.k[qo + thread];
            if (lane < Rows && records.value != args.v)
                records.value[vo + dv_base + lane] = args.v[vo + dv_base + lane];
        }
        run_recurrent_token(state, args.q + qo, args.k + qo, args.v + vo, args.g + vo,
                            args.beta + bo, args.dt_bias + head * kStateDim, args.out + vo, lane,
                            thread, dv_base, dqk_base, a_scale, args.lower_bound, args.scale,
                            Record && tile == 0 ? records.log_decay + vo : nullptr,
                            Record && tile == 0 ? records.beta + bo : nullptr);
    }
    if (lane < Rows)
        for (int t = valid; t < args.tokens; ++t) {
            const auto offset =
                ((static_cast<std::int64_t>(batch) * args.tokens + t) * args.value_heads + head) *
                    kStateDim +
                dv_base + lane;
            args.out[offset] = __float2bfloat16(0.0F);
        }
    if constexpr (!Record) {
        const auto destination =
            state_head + (Batched ? destination_offsets[batch] / sizeof(float) : 0);
        store_state(state, args.state_out + destination, dv_base, dqk_base);
    }
}

} // namespace ninfer::ops::detail::kimi_delta_attention
