#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.h"

#include "core/device.h"
#include "ops/gdn_input_proj/gdn_conv.cuh"
#include "ops/gdn_input_proj/gdn_conv_output.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

template <int WarpsPerCta, int TokenTile, class Publish>
__global__ __launch_bounds__(WarpsPerCta * 32) void nvfp4_gdn_conv_post_kernel(
    const __nv_bfloat16* __restrict__ projected, const __nv_bfloat16* __restrict__ conv_weight,
    const __nv_bfloat16* __restrict__ conv_states, const std::int32_t* __restrict__ initial_slot,
    const std::int32_t* __restrict__ valid_columns, __nv_bfloat16* __restrict__ query,
    __nv_bfloat16* __restrict__ key, __nv_bfloat16* __restrict__ value, std::int32_t tokens,
    Publish publish) {
    static_assert((kGdnChannels % 32) == 0);

    struct Shared {
        float initial[3][32];
        float weight[4][32];
    };

    __shared__ Shared shared;

    const int warp                     = static_cast<int>(threadIdx.x) >> 5;
    const int lane                     = static_cast<int>(threadIdx.x) & 31;
    const int row                      = static_cast<int>(blockIdx.x) * 32 + lane;
    constexpr std::int64_t kSlotStride = static_cast<std::int64_t>(kGdnChannels) * 3;
    std::int32_t valid                 = valid_columns == nullptr ? tokens : *valid_columns;
    valid                              = valid < 0 ? 0 : (valid > tokens ? tokens : valid);
    if (warp == 0) {
        const std::int64_t initial_base = static_cast<std::int64_t>(*initial_slot) * kSlotStride;
#pragma unroll
        for (int history = 0; history < 3; ++history) {
            shared.initial[history][lane] = __bfloat162float(
                conv_states[initial_base + static_cast<std::int64_t>(history) * kGdnChannels +
                            row]);
        }
#pragma unroll
        for (int tap = 0; tap < 4; ++tap) {
            shared.weight[tap][lane] =
                __bfloat162float(conv_weight[static_cast<std::int64_t>(tap) * kGdnChannels + row]);
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
    if (row < kGdnQueryRows) {
        output      = query;
        output_rows = kGdnQueryRows;
        output_row  = row;
    } else if (row < kGdnQueryRows + kGdnKeyRows) {
        output      = key;
        output_rows = kGdnKeyRows;
        output_row  = row - kGdnQueryRows;
    } else {
        output      = value;
        output_rows = kGdnValueRows;
        output_row  = row - kGdnQueryRows - kGdnKeyRows;
    }

    const int token_tiles = (tokens + TokenTile - 1) / TokenTile;
    for (int tile = warp; tile < token_tiles; tile += WarpsPerCta) {
        const int token0        = tile * TokenTile;
        const auto load_history = [&](int token) {
            if (token < 0) { return shared.initial[token + 3][lane]; }
            return __bfloat162float(
                projected[static_cast<std::int64_t>(token) * kGdnChannels + row]);
        };
        float s0 = load_history(token0 - 3);
        float s1 = load_history(token0 - 2);
        float s2 = load_history(token0 - 1);

#pragma unroll
        for (int local_token = 0; local_token < TokenTile; ++local_token) {
            const int token = token0 + local_token;
            if (token < tokens) {
                const std::int64_t projected_base = static_cast<std::int64_t>(token) * kGdnChannels;
                if (token >= valid) {
                    output[static_cast<std::int64_t>(token) * output_rows + output_row] =
                        __float2bfloat16_rn(0.0F);
                    continue;
                }
                const float p = __bfloat162float(projected[projected_base + row]);
                float conv    = fmaf(w0, s0, 0.0F);
                conv          = fmaf(w1, s1, conv);
                conv          = fmaf(w2, s2, conv);
                conv          = fmaf(w3, p, conv);
                output[static_cast<std::int64_t>(token) * output_rows + output_row] =
                    __float2bfloat16_rn(silu(conv));
                publish.publish(token, 0, row, s1, s2, p);
                s0 = s1;
                s1 = s2;
                s2 = p;
            }
        }
    }
}

template <class Publish>
void launch(const Tensor& projected, const Tensor& conv_weight, const Tensor& conv_states,
            const Tensor& valid_columns, const Tensor& initial_slot, Tensor& query, Tensor& key,
            Tensor& value, Publish publish, cudaStream_t stream) {
    constexpr int kWarpsPerCta = 32;
    constexpr int kTokenTile   = 8;
    constexpr int kBlocks      = kGdnChannels / 32;
    nvfp4_gdn_conv_post_kernel<kWarpsPerCta, kTokenTile><<<kBlocks, kWarpsPerCta * 32, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(projected.data),
        static_cast<const __nv_bfloat16*>(conv_weight.data),
        static_cast<const __nv_bfloat16*>(conv_states.data),
        static_cast<const std::int32_t*>(initial_slot.data),
        valid_columns.data == nullptr ? nullptr
                                      : static_cast<const std::int32_t*>(valid_columns.data),
        static_cast<__nv_bfloat16*>(query.data), static_cast<__nv_bfloat16*>(key.data),
        static_cast<__nv_bfloat16*>(value.data), projected.ne[1], publish);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void nvfp4_gdn_snapshot_post_launch(const Tensor& projected, const Tensor& conv_weight,
                                    Tensor& conv_states, const Tensor& valid_columns,
                                    const Tensor& initial_slot, const Tensor& snapshot_base_slot,
                                    Tensor& query, Tensor& key, Tensor& value,
                                    cudaStream_t stream) {
    launch(projected, conv_weight, conv_states, valid_columns, initial_slot, query, key, value,
           SnapshotHistoryPublish{static_cast<__nv_bfloat16*>(conv_states.data),
                                  static_cast<const std::int32_t*>(snapshot_base_slot.data),
                                  kGdnChannels},
           stream);
}

void nvfp4_gdn_record_post_launch(const Tensor& conv_record, const Tensor& conv_weight,
                                  const Tensor& conv_states, const Tensor& valid_columns,
                                  const Tensor& initial_slot, Tensor& query, Tensor& key,
                                  Tensor& value, cudaStream_t stream) {
    launch(conv_record, conv_weight, conv_states, valid_columns, initial_slot, query, key, value,
           NoHistoryPublish{}, stream);
}

} // namespace ninfer::ops::detail
