#pragma once

#include "core/weight.h"
#include "ops/op_tester.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::test::direct_bf16_weight {

struct HostWeight {
    std::int32_t n = 0;
    std::int32_t k = 0;
    std::vector<std::uint16_t> bits;

    [[nodiscard]] Weight device_weight(const void* data) const {
        Weight weight{};
        weight.payload         = data;
        weight.payload_bytes   = bits.size() * sizeof(std::uint16_t);
        weight.qtype           = QType::BF16;
        weight.shape[0]        = n;
        weight.shape[1]        = k;
        weight.padded_shape[0] = n;
        weight.padded_shape[1] = k;
        weight.ndim            = 2;
        weight.qdata           = data;
        weight.n               = n;
        weight.k               = k;
        weight.layout          = QuantLayout::Contiguous;
        return weight;
    }
};

inline HostWeight make_patterned(std::int32_t n, std::int32_t k, std::uint32_t seed) {
    if (n <= 0 || k <= 0) {
        throw std::invalid_argument("direct BF16 test weight requires positive shape");
    }
    HostWeight result;
    result.n = n;
    result.k = k;
    result.bits.resize(static_cast<std::size_t>(n) * k);
    for (std::int32_t row = 0; row < n; ++row) {
        for (std::int32_t column = 0; column < k; ++column) {
            std::uint32_t hash = static_cast<std::uint32_t>(row) * 0x9e3779b9U ^
                                 static_cast<std::uint32_t>(column) * 0x85ebca6bU ^
                                 seed * 0xc2b2ae35U;
            hash ^= hash >> 16;
            hash *= 0x7feb352dU;
            hash ^= hash >> 15;
            const int centered = static_cast<int>((hash >> 8) & 0xffU) - 128;
            const float value  = static_cast<float>(centered) * (1.0F / 1024.0F);
            result.bits[static_cast<std::size_t>(row) * k + column] = f32_to_bf16(value);
        }
    }
    return result;
}

inline double dot_fp64(const HostWeight& weight, std::int32_t row,
                       std::span<const float> activation) {
    if (row < 0 || row >= weight.n || activation.size() != static_cast<std::size_t>(weight.k)) {
        throw std::invalid_argument("invalid direct BF16 oracle argument");
    }
    const std::uint16_t* weight_row = weight.bits.data() + static_cast<std::size_t>(row) * weight.k;
    double result                   = 0.0;
    for (std::int32_t column = 0; column < weight.k; ++column) {
        result += static_cast<double>(bf16_to_f32(weight_row[column])) *
                  static_cast<double>(activation[static_cast<std::size_t>(column)]);
    }
    return result;
}

class DeviceWeight {
public:
    explicit DeviceWeight(HostWeight host_weight)
        : host(std::move(host_weight)), device(host.bits.size() * sizeof(std::uint16_t)) {
        device.copy_from_host(host.bits.data(), device.bytes);
    }

    [[nodiscard]] Weight view() const { return host.device_weight(device.p); }

    int verify_preserved(std::string_view label) const {
        std::vector<std::uint16_t> after(host.bits.size());
        device.copy_to_host(after.data(), device.bytes);
        if (after == host.bits) { return 0; }
        std::cerr << label << ": direct BF16 weight was modified\n";
        return 1;
    }

    HostWeight host;
    DeviceBuffer device;
};

} // namespace ninfer::test::direct_bf16_weight
