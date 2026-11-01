#pragma once

#include "ninfer/ops/qsa.h"
#include "ops/common/host/operands.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail::qsa {

struct ViewValidation {
    const char* op;
    Device device;

    void require(bool condition, const char* label) const {
        if (!condition) throw std::invalid_argument(std::string(op) + ": invalid " + label);
    }

    void dense(const Tensor& t, DType dtype, std::array<int, 4> shape, int alignment,
               const char* label) const {
        require(device.type() == DeviceType::CUDA && t.device == device, "operand device");
        require(t.dtype == dtype && t.is_contiguous() &&
                    common::operand::aligned(t.data, alignment),
                label);
        for (int d = 0; d < 4; ++d) require(t.ne[d] == shape[d] && shape[d] > 0, label);
    }

    bool overlap(const Tensor& a, const Tensor& b) const {
        const auto span = [&](const Tensor& t) {
            std::uint64_t bytes = dtype_size(t.dtype);
            for (int d = 0; d < 4; ++d) {
                require(t.ne[d] > 0, "view extent");
                if (t.ne[d] == 1) continue;
                require(t.nb[d] > 0, "view stride");
                const std::uint64_t stride = t.nb[d], count = t.ne[d] - 1;
                require(stride <= (std::numeric_limits<std::uintptr_t>::max() - bytes) / count,
                        "view byte span");
                bytes += count * stride;
            }
            return bytes;
        };
        const auto x  = reinterpret_cast<std::uintptr_t>(a.data),
                   y  = reinterpret_cast<std::uintptr_t>(b.data);
        const auto nx = span(a), ny = span(b);
        require(nx <= std::numeric_limits<std::uintptr_t>::max() - x &&
                    ny <= std::numeric_limits<std::uintptr_t>::max() - y,
                "address range");
        return common::operand::overlaps({x, x + nx}, {y, y + ny});
    }

    void metadata(const Tensor& t, int batch, const char* label) const {
        require(device.type() == DeviceType::CUDA && t.device == device, "operand device");
        require(t.dtype == DType::I32 && t.ne[0] == batch && t.ne[1] == 1 && t.ne[2] == 1 &&
                    t.ne[3] == 1 && t.is_contiguous() && common::operand::aligned(t.data, 4),
                label);
    }

    // Features are contiguous. Outer axes may have layer gaps or be interleaved; the smaller
    // axis is at most eight for QSA state/records, keeping the overlap check bounded.
    void columns(const Tensor& t, DType dtype, int features, int count, int outer,
                 const char* label) const {
        require(device.type() == DeviceType::CUDA && t.device == device, "operand device");
        const std::uint64_t element = dtype_size(dtype), width = features * element;
        require(count > 0 && outer > 0 && t.dtype == dtype && t.ne[0] == features &&
                    t.ne[1] == count && t.ne[2] == outer && t.ne[3] == 1 &&
                    t.nb[0] == static_cast<std::int64_t>(element) &&
                    common::operand::aligned(t.data, element),
                label);
        std::uint64_t span = width;
        for (int d = 1; d <= 2; ++d) {
            if (t.ne[d] == 1) continue;
            require(t.nb[d] >= static_cast<std::int64_t>(width) && t.nb[d] % element == 0, label);
            const auto stride = static_cast<std::uint64_t>(t.nb[d]);
            const auto extent = static_cast<std::uint64_t>(t.ne[d] - 1);
            require(stride <= (std::numeric_limits<std::int64_t>::max() - span) / extent, label);
            span += stride * extent;
        }
        require(span <= std::numeric_limits<std::uintptr_t>::max() -
                            reinterpret_cast<std::uintptr_t>(t.data),
                label);
        const int small = count <= outer ? 1 : 2, large = 3 - small;
        const auto small_stride = t.ne[small] == 1 ? width : std::uint64_t(t.nb[small]);
        const auto large_stride = t.ne[large] == 1 ? width : std::uint64_t(t.nb[large]);
        // Only the two nearest columns of the other progression can intersect this column.
        for (int delta = 1; delta < t.ne[small]; ++delta) {
            const auto distance = delta * small_stride, q = distance / large_stride;
            if (q < std::uint64_t(t.ne[large]))
                require(distance - q * large_stride >= width, label);
            if (q + 1 < std::uint64_t(t.ne[large]))
                require((q + 1) * large_stride - distance >= width, label);
        }
    }

    void state(const ConstQsaTailStateView& v, int batch, int layers = 1) const {
        common::operand::state_field(v.raw_keys, DType::BF16, {128, 3}, 2, op, "tail raw keys",
                                     device);
        common::operand::state_field(v.first_rope, DType::I32, {3}, 4, op, "tail coordinates",
                                     device);
        require(v.raw_keys.field.layers == layers && v.first_rope.field.layers == layers,
                "state layer");
        const auto& raw  = v.raw_keys.records;
        const auto& rope = v.first_rope.records;
        require(raw.byte_offsets.ne[0] == batch && rope.byte_offsets.ne[0] == batch, "state rows");
    }
};

} // namespace ninfer::ops::detail::qsa
