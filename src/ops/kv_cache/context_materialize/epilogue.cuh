#pragma once

#include "ninfer/ops/context_kv_materialize.h"
#include "ops/common/cuda/reduce.cuh"
#include "ops/kv_cache/kernels/layout.cuh"
#include "ops/rope/kernels/phase.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace ninfer::ops::detail::context_kv {

constexpr int kLayers  = static_cast<int>(kContextKVMaterializeLayers);
constexpr int kRows    = 1024;
constexpr int kHeadDim = 128;

__device__ __forceinline__ int context_column(int column, int width, int prefix) {
    return width == prefix ? column : column / prefix * width + column % prefix;
}

struct DeviceLayerView {
    const std::uint8_t* key_codes;
    const std::uint8_t* key_scales;
    const std::uint8_t* value_codes;
    const std::uint8_t* value_scales;
    const __nv_bfloat16* key_norm;
    __nv_bfloat16* cache_k;
    __half* cache_v;
    std::int32_t padded_capacity;
};

struct DeviceLayers {
    DeviceLayerView layer[kContextKVMaterializeLayers];
};

__device__ __forceinline__ void store_key_head(const float* input, DeviceLayerView layer,
                                               const int* positions, const std::int64_t* offsets,
                                               int column, int width, int head) {
    const int lane = threadIdx.x & 31;
    const int j    = lane * 2;
    float x0 = input[j], x1 = input[j + 1], y0 = input[j + 64], y1 = input[j + 65];
    float sum           = common::warp_reduce_sum(x0 * x0 + x1 * x1 + y0 * y0 + y1 * y1);
    const float inverse = rsqrtf(__shfl_sync(0xffffffffU, sum, 0) / 128.0f + 1.e-6f);
    x0 *= inverse * __bfloat162float(layer.key_norm[j]);
    x1 *= inverse * __bfloat162float(layer.key_norm[j + 1]);
    y0 *= inverse * __bfloat162float(layer.key_norm[j + 64]);
    y1 *= inverse * __bfloat162float(layer.key_norm[j + 65]);
    float sin0, cos0, sin1, cos1;
    rope_d128_theta1e7_sincos(positions, column, j, &sin0, &cos0);
    rope_d128_theta1e7_sincos(positions, column, j + 1, &sin1, &cos1);
    const auto dst = offsets[column / width] / sizeof(__nv_bfloat16) +
                     kv::cyclic_offset<128, 2048>(positions[column], head, layer.padded_capacity);
    auto* out      = reinterpret_cast<__nv_bfloat162*>(layer.cache_k + dst);
    out[lane]      = __floats2bfloat162_rn(x0 * cos0 - y0 * sin0, x1 * cos1 - y1 * sin1);
    out[lane + 32] = __floats2bfloat162_rn(y0 * cos0 + x0 * sin0, y1 * cos1 + x1 * sin1);
}

__device__ __forceinline__ void store_value(int row, int column, std::int64_t offset,
                                            const int* positions, DeviceLayerView layer,
                                            float result) {
    const auto dst =
        offset / sizeof(__half) + kv::cyclic_offset<128, 2048>(positions[column], row / 128,
                                                               layer.padded_capacity, row % 128);
    layer.cache_v[dst] = __float2half_rn(__bfloat162float(__float2bfloat16_rn(result)));
}

struct MaterializeProjectionEpilogue {
    DeviceLayerView layer;
    const int* positions;
    const int* counts;
    const std::int64_t* offsets;
    float* scratch;
    int layer_index, width, batch, min_count, max_count;
    bool value;

    __device__ void store(int row, int packed_column, float result) const {
        if (packed_column >= max_count * batch) return;
        const int column  = context_column(packed_column, width, max_count);
        const int request = column / width, count = counts[request];
        if (count < min_count || count > max_count || column % width >= count) return;
        if (!value)
            scratch[row + 1024LL * (packed_column + max_count * batch * layer_index)] = result;
        else { store_value(row, column, offsets[request], positions, layer, result); }
    }

    template <class Output>
    __device__ void store_fragment(const Output&, int row, int col, float4 sum, int,
                                   int columns) const {
        if (col < columns) {
            store(row, col, sum.x);
            store(row + 8, col, sum.z);
        }
        if (col + 1 < columns) {
            store(row, col + 1, sum.y);
            store(row + 8, col + 1, sum.w);
        }
    }
};

struct ContextPrefixColumns {
    int width, prefix;

    __device__ __forceinline__ int operator()(int column) const {
        return context_column(column, width, prefix);
    }
};

__global__ __launch_bounds__(256) void context_kv_key_post_kernel(
    const float* __restrict__ key_scratch, const std::int32_t* __restrict__ positions,
    const std::int32_t* __restrict__ counts, const std::int64_t* __restrict__ state_offsets,
    DeviceLayers layers, std::int32_t batch_size, std::int32_t width, std::int32_t min_count,
    std::int32_t max_count) {
    const int packed_column   = static_cast<int>(blockIdx.x);
    const int physical_column = context_column(packed_column, width, max_count);
    const int layer_index     = static_cast<int>(blockIdx.y);
    const int batch           = physical_column / width;
    const int local           = physical_column % width;
    const int count           = counts[batch];
    if (count < min_count || count > max_count || local >= count) return;
    const auto layer   = layers.layer[layer_index];
    const int head     = threadIdx.x >> 5;
    const float* input = key_scratch +
                         kRows * (packed_column + max_count * batch_size * layer_index) +
                         head * kHeadDim;
    store_key_head(input, layer, positions, state_offsets, physical_column, width, head);
}

} // namespace ninfer::ops::detail::context_kv
