#pragma once

#include "core/buffer.h"
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
        return static_cast<std::uint64_t>(storage.size());
    }
};

namespace detail {

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
                                                std::uint32_t seed = 0x51U, float extent = 0.05F) {
    const std::size_t bytes = detail::direct_bf16_bytes(n, k);
    DirectBf16Weight result{DeviceBuffer(ninfer::bench::device_context().device(), bytes), {}};
    const std::uint64_t elements = static_cast<std::uint64_t>(n) * k;
    CUDA_CHECK(fixture::fill_values(static_cast<__nv_bfloat16*>(result.storage.data()), elements,
                                    seed, -extent, extent));
    CUDA_CHECK(cudaDeviceSynchronize());

    Weight& weight         = result.weight;
    weight.device          = result.storage.device();
    weight.payload         = result.storage.data();
    weight.payload_bytes   = bytes;
    weight.qtype           = QType::BF16;
    weight.shape[0]        = n;
    weight.shape[1]        = k;
    weight.padded_shape[0] = n;
    weight.padded_shape[1] = k;
    weight.ndim            = 2;
    weight.qdata           = result.storage.data();
    weight.n               = n;
    weight.k               = k;
    weight.layout          = QuantLayout::Contiguous;
    return result;
}

} // namespace ninfer::bench
