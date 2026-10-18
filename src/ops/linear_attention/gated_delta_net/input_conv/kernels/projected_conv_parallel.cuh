#pragma once

#include "ops/linear_attention/gated_delta_net/input_conv/projected_conv.h"
#include "core/device.h"
#include "ops/linear_attention/gated_delta_net/input_conv/kernels/conv.cuh"
#include "ops/linear_attention/gated_delta_net/input_conv/kernels/output.cuh"
#include <cuda_bf16.h>
#include <cstdint>

namespace ninfer::ops::detail {
using ConvGeometry = GdnConvGeometry<6144>;

namespace gdn_projected_parallel {

template <class ConvGeometry, int WarpsPerCta, int TokenTile, class Publish>
__global__ __launch_bounds__(WarpsPerCta * 32) void gdn_projected_conv_parallel_kernel(
    const __nv_bfloat16* __restrict__ projected, const __nv_bfloat16* __restrict__ conv_weight,
    const __nv_bfloat16* __restrict__ conv_states, const std::int32_t* __restrict__ initial_slot,
    const std::int32_t* __restrict__ valid_columns, __nv_bfloat16* __restrict__ query,
    __nv_bfloat16* __restrict__ key, __nv_bfloat16* __restrict__ value, std::int32_t tokens,
    std::int64_t slot_stride, Publish publish) {
    static_assert((ConvGeometry::channels % 32) == 0);

    struct Shared {
        float initial[3][32];
        float weight[4][32];
    };

    __shared__ Shared shared;

    const int warp     = static_cast<int>(threadIdx.x) >> 5;
    const int lane     = static_cast<int>(threadIdx.x) & 31;
    const int row      = static_cast<int>(blockIdx.x) * 32 + lane;
    std::int32_t valid = valid_columns == nullptr ? tokens : *valid_columns;
    valid              = valid < 0 ? 0 : (valid > tokens ? tokens : valid);
    if (warp == 0) {
        const std::int64_t initial_base = static_cast<std::int64_t>(*initial_slot) * slot_stride;
#pragma unroll
        for (int history = 0; history < 3; ++history) {
            shared.initial[history][lane] = __bfloat162float(
                conv_states[initial_base +
                            static_cast<std::int64_t>(history) * ConvGeometry::channels + row]);
        }
#pragma unroll
        for (int tap = 0; tap < 4; ++tap) {
            shared.weight[tap][lane] = __bfloat162float(
                conv_weight[static_cast<std::int64_t>(tap) * ConvGeometry::channels + row]);
        }
    }
    __syncthreads();

    const float w0 = shared.weight[0][lane];
    const float w1 = shared.weight[1][lane];
    const float w2 = shared.weight[2][lane];
    const float w3 = shared.weight[3][lane];

    __nv_bfloat16* output;
    std::int32_t output_rows;
    std::int32_t output_row;
    if (row < ConvGeometry::query_rows) {
        output      = query;
        output_rows = ConvGeometry::query_rows;
        output_row  = row;
    } else if (row < ConvGeometry::query_rows + ConvGeometry::key_rows) {
        output      = key;
        output_rows = ConvGeometry::key_rows;
        output_row  = row - ConvGeometry::query_rows;
    } else {
        output      = value;
        output_rows = ConvGeometry::value_rows;
        output_row  = row - ConvGeometry::query_rows - ConvGeometry::key_rows;
    }

    const int token_tiles = (tokens + TokenTile - 1) / TokenTile;
    for (int tile = warp; tile < token_tiles; tile += WarpsPerCta) {
        const int token0        = tile * TokenTile;
        const auto load_history = [&](int token) {
            if (token < 0) { return shared.initial[token + 3][lane]; }
            return __bfloat162float(
                projected[static_cast<std::int64_t>(token) * ConvGeometry::channels + row]);
        };
        float s0 = load_history(token0 - 3);
        float s1 = load_history(token0 - 2);
        float s2 = load_history(token0 - 1);

#pragma unroll
        for (int local_token = 0; local_token < TokenTile; ++local_token) {
            const int token = token0 + local_token;
            if (token < tokens) {
                const std::int64_t projected_base =
                    static_cast<std::int64_t>(token) * ConvGeometry::channels;
                if (token >= valid) {
                    output[static_cast<std::int64_t>(token) * output_rows + output_row] =
                        __float2bfloat16_rn(0.0F);
                    continue;
                }
                const float p = __bfloat162float(projected[projected_base + row]);
                output[static_cast<std::int64_t>(token) * output_rows + output_row] =
                    causal_conv4_silu(w0, w1, w2, w3, s0, s1, s2, p);
                publish.publish(token, 0, row, s1, s2, p);
                s0 = s1;
                s1 = s2;
                s2 = p;
            }
        }
    }
}

template <int WarpsPerCta, class ConvGeometry = GdnConvGeometry<6144>, class Publish>
void launch(const Tensor& projected, const Tensor& conv_weight, const Tensor& conv_states,
            const Tensor& valid_columns, const Tensor& initial_slot, Tensor& query, Tensor& key,
            Tensor& value, Publish publish, cudaStream_t stream) {
    constexpr int kWarpsPerCta = WarpsPerCta;
    constexpr int kBlockTokens = 8;
    constexpr int kBlocks      = ConvGeometry::channels / 32;
    gdn_projected_conv_parallel_kernel<ConvGeometry, kWarpsPerCta, kBlockTokens>
        <<<kBlocks, kWarpsPerCta * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(projected.data),
            static_cast<const __nv_bfloat16*>(conv_weight.data),
            static_cast<const __nv_bfloat16*>(conv_states.data),
            static_cast<const std::int32_t*>(initial_slot.data),
            valid_columns.data == nullptr ? nullptr
                                          : static_cast<const std::int32_t*>(valid_columns.data),
            static_cast<__nv_bfloat16*>(query.data), static_cast<__nv_bfloat16*>(key.data),
            static_cast<__nv_bfloat16*>(value.data), projected.ne[1], conv_states.nb[2] / 2,
            publish);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace gdn_projected_parallel

} // namespace ninfer::ops::detail
