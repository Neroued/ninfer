#include "ops/linear_attention/gated_delta_net/input_conv/projected_conv.h"
#include "core/device.h"
#include "core/platform.h"
#include "ops/linear_attention/gated_delta_net/input_conv/kernels/conv.cuh"
#include <cuda_bf16.h>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

template <int Channels, int QueryRows, int KeyRows, int ValueRows, int StaticWidth, class Publish>
__global__ void gdn_projected_conv_kernel(
    const __nv_bfloat16* __restrict__ projected, const __nv_bfloat16* __restrict__ conv_weight,
    const __nv_bfloat16* state_read, const std::int32_t* __restrict__ valid_columns,
    const std::int64_t* __restrict__ source_offsets, __nv_bfloat16* __restrict__ query,
    __nv_bfloat16* __restrict__ key, __nv_bfloat16* __restrict__ value, std::int32_t width,
    Publish publish) {
    static_assert(Channels == QueryRows + KeyRows + ValueRows);
    const std::int32_t row = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= Channels) { return; }
    const std::int32_t batch = static_cast<std::int32_t>(blockIdx.y);
    if constexpr (StaticWidth != 0) { width = StaticWidth; }

    std::int32_t valid              = valid_columns == nullptr ? width : valid_columns[batch];
    valid                           = valid < 0 ? 0 : (valid > width ? width : valid);
    const std::int64_t initial_base = source_offsets[batch] / 2;
    float s0                        = __bfloat162float(state_read[initial_base + row]);
    float s1                        = __bfloat162float(state_read[initial_base + Channels + row]);
    float s2       = __bfloat162float(state_read[initial_base + 2LL * Channels + row]);
    const float w0 = __bfloat162float(conv_weight[row]);
    const float w1 = __bfloat162float(conv_weight[Channels + row]);
    const float w2 = __bfloat162float(conv_weight[2LL * Channels + row]);
    const float w3 = __bfloat162float(conv_weight[3LL * Channels + row]);

    for (std::int32_t token = 0; token < width; ++token) {
        const std::int64_t column = static_cast<std::int64_t>(batch) * width + token;
        if (token >= valid) {
            if (row < QueryRows) {
                query[column * QueryRows + row] = __float2bfloat16_rn(0.0F);
            } else if (row < QueryRows + KeyRows) {
                key[column * KeyRows + row - QueryRows] = __float2bfloat16_rn(0.0F);
            } else {
                value[column * ValueRows + row - QueryRows - KeyRows] = __float2bfloat16_rn(0.0F);
            }
            continue;
        }

        const float p              = __bfloat162float(projected[column * Channels + row]);
        const __nv_bfloat16 output = causal_conv4_silu(w0, w1, w2, w3, s0, s1, s2, p);
        if (row < QueryRows) {
            query[column * QueryRows + row] = output;
        } else if (row < QueryRows + KeyRows) {
            key[column * KeyRows + row - QueryRows] = output;
        } else {
            value[column * ValueRows + row - QueryRows - KeyRows] = output;
        }
        publish.publish(token, batch, row, s1, s2, p);
        s0 = s1;
        s1 = s2;
        s2 = p;
    }
}

template <int Channels, int QueryRows, int KeyRows, int ValueRows, class Publish>
void launch(const Tensor& projected, const Tensor& conv_weight,
            const ConstStateFieldView& state_read, const Tensor& valid_columns, Tensor& query,
            Tensor& key, Tensor& value, Publish publish, cudaStream_t stream) {
    constexpr int kDefaultThreads = 256;
    const std::int32_t width      = projected.ne[1];
    const std::int32_t batch      = projected.ne[2];
    if constexpr (Channels == 10240) {
        if (width == 4 && batch == 1) {
            constexpr int kT4Threads = 64;
            gdn_projected_conv_kernel<Channels, QueryRows, KeyRows, ValueRows, 4>
                <<<(Channels + kT4Threads - 1) / kT4Threads, kT4Threads, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(projected.data),
                    static_cast<const __nv_bfloat16*>(conv_weight.data),
                    reinterpret_cast<const __nv_bfloat16*>(
                        (static_cast<const std::byte*>(state_read.records.storage.data) +
                         state_read.field.tensor.region.offset)),
                    valid_columns.data == nullptr
                        ? nullptr
                        : static_cast<const std::int32_t*>(valid_columns.data),
                    static_cast<const std::int64_t*>(state_read.records.byte_offsets.data),
                    static_cast<__nv_bfloat16*>(query.data), static_cast<__nv_bfloat16*>(key.data),
                    static_cast<__nv_bfloat16*>(value.data), width, publish);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
    }
    const dim3 grid((Channels + kDefaultThreads - 1) / kDefaultThreads,
                    static_cast<unsigned>(batch));
    gdn_projected_conv_kernel<Channels, QueryRows, KeyRows, ValueRows, 0>
        <<<grid, kDefaultThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(projected.data),
            static_cast<const __nv_bfloat16*>(conv_weight.data),
            reinterpret_cast<const __nv_bfloat16*>(
                (static_cast<const std::byte*>(state_read.records.storage.data) +
                 state_read.field.tensor.region.offset)),
            valid_columns.data == nullptr ? nullptr
                                          : static_cast<const std::int32_t*>(valid_columns.data),
            static_cast<const std::int64_t*>(state_read.records.byte_offsets.data),
            static_cast<__nv_bfloat16*>(query.data), static_cast<__nv_bfloat16*>(key.data),
            static_cast<__nv_bfloat16*>(value.data), width, publish);
    CUDA_CHECK(cudaGetLastError());
}

template <class Publish>
void dispatch(const Tensor& projected, const Tensor& conv_weight,
              const ConstStateFieldView& state_read, const Tensor& valid_columns, Tensor& query,
              Tensor& key, Tensor& value, Publish publish, cudaStream_t stream) {
    if (projected.ne[0] == 10240 && query.ne[0] == 2048 && key.ne[0] == 2048 &&
        value.ne[0] == 6144) {
        launch<10240, 2048, 2048, 6144>(projected, conv_weight, state_read, valid_columns, query,
                                        key, value, publish, stream);
        return;
    }
    if (projected.ne[0] == 8192 && query.ne[0] == 2048 && key.ne[0] == 2048 &&
        value.ne[0] == 4096) {
        launch<8192, 2048, 2048, 4096>(projected, conv_weight, state_read, valid_columns, query,
                                       key, value, publish, stream);
        return;
    }
#if NINFER_TARGET_SPARK
    if (projected.ne[0] == 5120 && query.ne[0] == 1024 && key.ne[0] == 1024 &&
        value.ne[0] == 3072) {
        launch<5120, 1024, 1024, 3072>(projected, conv_weight, state_read, valid_columns, query,
                                       key, value, publish, stream);
        return;
    }
#endif
    throw std::invalid_argument("GDN projected-conv received an unregistered geometry");
}

} // namespace

void launch_gdn_projected_conv_snapshot(const Tensor& projected, const Tensor& conv_weight,
                                        const ConstStateFieldView& conv_states,
                                        const Tensor& valid_columns,
                                        const gdn_input::HistoryDestination& destination,
                                        Tensor& query, Tensor& key, Tensor& value,
                                        cudaStream_t stream) {
    dispatch(projected, conv_weight, conv_states, valid_columns, query, key, value,
             make_snapshot_history_publish(destination, projected.ne[0]), stream);
}

void launch_gdn_projected_conv_record(const Tensor& conv_record, const Tensor& conv_weight,
                                      const ConstStateFieldView& conv_states,
                                      const Tensor& valid_columns, Tensor& query, Tensor& key,
                                      Tensor& value, cudaStream_t stream) {
    dispatch(conv_record, conv_weight, conv_states, valid_columns, query, key, value,
             NoHistoryPublish{}, stream);
}

} // namespace ninfer::ops::detail
