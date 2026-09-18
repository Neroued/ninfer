#pragma once

#include "core/weight.h"
#include "ninfer_bench_common.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::bench {

struct DirectBf16Weight {
    DeviceBuffer storage;
    Weight weight{};

    [[nodiscard]] std::uint64_t model_weight_bytes() const noexcept {
        return static_cast<std::uint64_t>(storage.bytes);
    }
};

namespace detail {

static __global__ void fill_direct_bf16_weight_kernel(__nv_bfloat16* values, std::uint64_t count,
                                                      std::uint32_t seed) {
    const std::uint64_t begin  = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t stride = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t index = begin; index < count; index += stride) {
        std::uint32_t hash = static_cast<std::uint32_t>(index) * 0x9e3779b9U ^ seed * 0x85ebca6bU;
        hash ^= hash >> 16;
        const int centered = static_cast<int>((hash >> 8) & 0xffU) - 128;
        values[index]      = __float2bfloat16_rn(static_cast<float>(centered) * (1.0F / 1024.0F));
    }
}

inline std::size_t direct_bf16_bytes(std::int32_t n, std::int32_t k) {
    if (n <= 0 || k <= 0) {
        throw std::invalid_argument("invalid direct BF16 benchmark weight shape");
    }
    const auto elements = static_cast<std::uint64_t>(n) * static_cast<std::uint64_t>(k);
    if (elements > std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t)) {
        throw std::overflow_error("direct BF16 benchmark weight size overflow");
    }
    return static_cast<std::size_t>(elements * sizeof(std::uint16_t));
}

} // namespace detail

inline DirectBf16Weight make_direct_bf16_weight(std::int32_t n, std::int32_t k,
                                                std::uint32_t seed = 0x51U) {
    const std::size_t bytes = detail::direct_bf16_bytes(n, k);
    DirectBf16Weight result{DeviceBuffer(bytes), {}};
    const std::uint64_t elements = static_cast<std::uint64_t>(n) * k;
    const int blocks             = static_cast<int>(
        std::min<std::uint64_t>(65535, std::max<std::uint64_t>(1, (elements + 255) / 256)));
    detail::fill_direct_bf16_weight_kernel<<<blocks, 256>>>(
        static_cast<__nv_bfloat16*>(result.storage.p), elements, seed);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    Weight& weight         = result.weight;
    weight.payload         = result.storage.p;
    weight.payload_bytes   = bytes;
    weight.qtype           = QType::BF16;
    weight.shape[0]        = n;
    weight.shape[1]        = k;
    weight.padded_shape[0] = n;
    weight.padded_shape[1] = k;
    weight.ndim            = 2;
    weight.qdata           = result.storage.p;
    weight.n               = n;
    weight.k               = k;
    weight.layout          = QuantLayout::Contiguous;
    return result;
}

} // namespace ninfer::bench
