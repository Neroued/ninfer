#pragma once

#include "ops/common/math.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Device-side implementation detail shared by the exact packed projection kernels. The public
// fused Op owns the projection -> causal convolution -> SiLU -> split/snapshot semantics; this
// value only carries its already-validated physical views into a compile-time epilogue.
struct GdnConvSnapshotEpilogue {
    const __nv_bfloat16* conv_weight;
    __nv_bfloat16* conv_states;
    const std::int32_t* initial_slot;
    __nv_bfloat16* query;
    __nv_bfloat16* key;
    __nv_bfloat16* value;
    std::int32_t channels;
    std::int32_t query_rows;
    std::int32_t key_rows;
    std::int32_t value_rows;
    std::int32_t global_row_offset;

    template <int Tokens>
    __device__ __forceinline__ void store(std::int32_t local_row,
                                          const float (&projected)[Tokens]) const {
        static_assert(Tokens >= 1);
        const std::int32_t row          = global_row_offset + local_row;
        const std::int64_t slot_stride  = static_cast<std::int64_t>(channels) * 3;
        const std::int64_t initial_base = static_cast<std::int64_t>(*initial_slot) * slot_stride;

        float s0       = __bfloat162float(conv_states[initial_base + row]);
        float s1       = __bfloat162float(conv_states[initial_base + channels + row]);
        float s2       = __bfloat162float(conv_states[initial_base + 2LL * channels + row]);
        const float w0 = __bfloat162float(conv_weight[row]);
        const float w1 = __bfloat162float(conv_weight[channels + row]);
        const float w2 = __bfloat162float(conv_weight[2LL * channels + row]);
        const float w3 = __bfloat162float(conv_weight[3LL * channels + row]);

#pragma unroll
        for (int token = 0; token < Tokens; ++token) {
            const float p              = projected[token];
            float conv                 = fmaf(w0, s0, 0.0F);
            conv                       = fmaf(w1, s1, conv);
            conv                       = fmaf(w2, s2, conv);
            conv                       = fmaf(w3, p, conv);
            const __nv_bfloat16 output = __float2bfloat16_rn(silu(conv));
            if (row < query_rows) {
                query[static_cast<std::int64_t>(token) * query_rows + row] = output;
            } else if (row < query_rows + key_rows) {
                key[static_cast<std::int64_t>(token) * key_rows + row - query_rows] = output;
            } else {
                value[static_cast<std::int64_t>(token) * value_rows + row - query_rows - key_rows] =
                    output;
            }

            const std::int64_t snapshot_base = static_cast<std::int64_t>(token) * slot_stride;
            conv_states[snapshot_base + row] = __float2bfloat16_rn(s1);
            conv_states[snapshot_base + channels + row]       = __float2bfloat16_rn(s2);
            conv_states[snapshot_base + 2LL * channels + row] = __float2bfloat16_rn(p);
            s0                                                = s1;
            s1                                                = s2;
            s2                                                = p;
        }
    }
};

template <int Channels, int QueryRows, int KeyRows, int ValueRows, int Tokens>
__launch_bounds__(64) __global__
    void gdn_projected_conv_snapshot_kernel(const __nv_bfloat16* __restrict__ projected,
                                            const __nv_bfloat16* __restrict__ conv_weight,
                                            __nv_bfloat16* __restrict__ conv_states,
                                            const std::int32_t* __restrict__ initial_slot,
                                            __nv_bfloat16* __restrict__ query,
                                            __nv_bfloat16* __restrict__ key,
                                            __nv_bfloat16* __restrict__ value) {
    static_assert(Channels == QueryRows + KeyRows + ValueRows && Tokens >= 1);
    const int row =
        static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (row >= Channels) { return; }

    constexpr std::int64_t slot_stride = static_cast<std::int64_t>(Channels) * 3;
    const std::int64_t initial_base    = static_cast<std::int64_t>(*initial_slot) * slot_stride;
    float s0                           = __bfloat162float(conv_states[initial_base + row]);
    float s1       = __bfloat162float(conv_states[initial_base + Channels + row]);
    float s2       = __bfloat162float(conv_states[initial_base + 2LL * Channels + row]);
    const float w0 = __bfloat162float(conv_weight[row]);
    const float w1 = __bfloat162float(conv_weight[Channels + row]);
    const float w2 = __bfloat162float(conv_weight[2LL * Channels + row]);
    const float w3 = __bfloat162float(conv_weight[3LL * Channels + row]);

#pragma unroll
    for (int token = 0; token < Tokens; ++token) {
        const float p =
            __bfloat162float(projected[static_cast<std::int64_t>(token) * Channels + row]);
        float conv                 = fmaf(w0, s0, 0.0F);
        conv                       = fmaf(w1, s1, conv);
        conv                       = fmaf(w2, s2, conv);
        conv                       = fmaf(w3, p, conv);
        const __nv_bfloat16 output = __float2bfloat16_rn(silu(conv));
        if (row < QueryRows) {
            query[static_cast<std::int64_t>(token) * QueryRows + row] = output;
        } else if (row < QueryRows + KeyRows) {
            key[static_cast<std::int64_t>(token) * KeyRows + row - QueryRows] = output;
        } else {
            value[static_cast<std::int64_t>(token) * ValueRows + row - QueryRows - KeyRows] =
                output;
        }

        const std::int64_t snapshot_base = static_cast<std::int64_t>(token) * slot_stride;
        conv_states[snapshot_base + row] = __float2bfloat16_rn(s1);
        conv_states[snapshot_base + Channels + row]       = __float2bfloat16_rn(s2);
        conv_states[snapshot_base + 2LL * Channels + row] = __float2bfloat16_rn(p);
        s0                                                = s1;
        s1                                                = s2;
        s2                                                = p;
    }
}

} // namespace ninfer::ops::detail
