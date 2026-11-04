#pragma once

#include "core/buffer.h"
#include "core/weight_view.h"

#include <cuda_fp8.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <span>
#include <vector>

namespace ninfer::tool::kpool {

// Native encoded operands shared by the Op test and benchmark. Reference mathematics
// decodes these bytes independently; it does not use the source values or this encoder.
struct EncodedWeight {
    WeightGeometry geometry;
    std::vector<std::byte> host;
    DeviceBuffer storage;
    Weight weight;

    EncodedWeight(Device device, QType format, int rows, int columns, std::span<const float> values)
        : geometry(weight_geometry(
              format,
              format == QType::MXFP8 ? QuantLayout::RowGroupScale : QuantLayout::BlockScaleM128K128,
              std::array<std::uint64_t, 2>{std::uint64_t(rows), std::uint64_t(columns)})),
          host(geometry.bytes), storage(device, geometry.bytes) {
        if (values.size() != std::size_t(rows) * columns)
            throw std::invalid_argument("Kpool encoded weight fixture size");
        const bool mx        = format == QType::MXFP8;
        const int group_rows = mx ? 1 : 128, group_columns = mx ? 32 : 128;
        for (int r = 0; r < rows; r += group_rows)
            for (int k = 0; k < columns; k += group_columns) {
                float maximum = 0;
                for (int dr = 0; dr < group_rows; ++dr)
                    for (int dk = 0; dk < group_columns; ++dk)
                        maximum = std::max(maximum, std::abs(values[(r + dr) * columns + k + dk]));
                float multiplier  = maximum ? maximum / 448.F : 1.F;
                const auto offset = weight_scale_offset(geometry, r, k / group_columns);
                if (mx) {
                    int exponent = 0;
                    if (maximum) {
                        int e;
                        const double fraction = std::frexp(double(maximum) / 448., &e);
                        exponent              = std::max(-127, e - (fraction == .5));
                    }
                    host[offset] = std::byte(exponent + 127);
                    multiplier   = std::ldexp(1.F, exponent);
                } else
                    std::memcpy(host.data() + offset, &multiplier, sizeof(multiplier));
                for (int dr = 0; dr < group_rows; ++dr)
                    for (int dk = 0; dk < group_columns; ++dk) {
                        const auto index = std::size_t(r + dr) * columns + k + dk;
                        host[index] = std::byte(__nv_cvt_float_to_fp8(values[index] / multiplier,
                                                                      __NV_SATFINITE, __NV_E4M3));
                    }
            }
        storage.copy_from_host(host.data(), host.size());
        const WeightParent parent{
            geometry, device, static_cast<const std::byte*>(storage.data()), {}};
        weight = native_weight({geometry.shape, {{&parent, 0, geometry.elements}}});
    }

    std::size_t bytes() const { return geometry.code_bytes + geometry.scale_bytes; }

    bool unchanged() const {
        std::vector<std::byte> actual(host.size());
        storage.copy_to_host(actual.data(), actual.size());
        return actual == host;
    }
};
} // namespace ninfer::tool::kpool
