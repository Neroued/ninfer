#pragma once

#include "core/tensor.h"
#include "core/record_view.h"
#include "core/paged_kv_cache.h"
#include "core/weight.h"

#include <array>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <initializer_list>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ninfer::ops::common::operand {

struct Range {
    std::uintptr_t begin, end;
};

inline void device(Device actual, Device expected, const char* op, const char* name) {
    if (expected.type() == DeviceType::None || actual != expected) {
        throw std::invalid_argument(std::string(op) + ": " + name +
                                    " must belong to the execution device");
    }
}

inline void dtype(const Tensor& tensor, DType expected, const char* op, const char* name) {
    if (tensor.dtype != expected) throw std::invalid_argument(std::string(op) + ": " + name);
}

inline void shape(const Tensor& tensor, std::array<std::int32_t, 4> expected, const char* op,
                  const char* name) {
    for (int d = 0; d < 4; ++d)
        if (tensor.ne[d] != expected[d])
            throw std::invalid_argument(std::string(op) + ": invalid shape for " + name);
}

inline void contiguous_nonnull(const Tensor& tensor, const char* op, const char* name,
                               Device expected) {
    device(tensor.device, expected, op, name);
    if (!tensor.is_contiguous())
        throw std::invalid_argument(std::string(op) + ": " + name + " must be contiguous");
    if (!tensor.data)
        throw std::invalid_argument(std::string(op) + ": " + name + " data must be non-null");
}

inline std::int64_t numel(const Tensor& tensor, const char* op) {
    bool empty = false;
    for (const auto extent : tensor.ne) {
        if (extent < 0)
            throw std::invalid_argument(std::string(op) + ": negative tensor dimension");
        empty |= extent == 0;
    }
    if (empty) return 0;
    std::int64_t count = 1;
    for (const auto extent : tensor.ne) {
        if (count > std::numeric_limits<std::int64_t>::max() / extent)
            throw std::overflow_error(std::string(op) + ": tensor size overflows int64");
        count *= extent;
    }
    return count;
}

inline bool aligned(const void* pointer, std::size_t alignment) noexcept {
    return pointer && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

inline void tensor(const Tensor& value, DType dtype, std::initializer_list<std::int32_t> dimensions,
                   std::size_t alignment, const char* op, const char* name,
                   Device expected_device) {
    device(value.device, expected_device, op, name);
    const auto expected = Tensor::metadata(dtype, dimensions);
    if (value.dtype != dtype || !value.is_contiguous() || !aligned(value.data, alignment))
        throw std::invalid_argument(std::string(op) + ": invalid " + name);
    for (int axis = 0; axis < 4; ++axis)
        if (value.ne[axis] != expected.ne[axis])
            throw std::invalid_argument(std::string(op) + ": invalid " + name);
    if (value.bytes() != expected.bytes())
        throw std::invalid_argument(std::string(op) + ": invalid " + name);
}

template <class FieldView>
inline void state_field(const FieldView& value, DType dtype,
                        std::initializer_list<std::int32_t> dimensions, std::size_t alignment,
                        const char* op, const char* name, Device expected_device,
                        bool require_offsets = true) {
    device(value.records.storage.device, expected_device, op, name);
    const auto expected            = Tensor::metadata(dtype, dimensions);
    const auto& layout             = value.field.tensor;
    const auto& region             = layout.region;
    const std::size_t layer_stride = value.field.layer_stride;
    if (!value.records.storage.data || layout.dtype != dtype ||
        !std::equal(layout.shape.begin(), layout.shape.end(), expected.ne) ||
        !value.field.metadata().is_contiguous() || value.field.layers == 0 ||
        region.bytes != expected.bytes() || region.offset > value.records.storage.bytes ||
        region.bytes > value.records.storage.bytes - region.offset ||
        !aligned(static_cast<const std::byte*>(value.records.storage.data) + region.offset,
                 alignment) ||
        (value.field.layers > 1 &&
         (layer_stride < region.bytes || layer_stride % alignment != 0 ||
          (value.field.layers - 1) >
              (value.records.storage.bytes - region.offset - region.bytes) / layer_stride))) {
        throw std::invalid_argument(std::string(op) + ": invalid " + name);
    }
    if (require_offsets || value.records.byte_offsets.data) {
        const auto& offsets = value.records.byte_offsets;
        if (offsets.ne[0] <= 0)
            throw std::invalid_argument(std::string(op) + ": empty state offset table");
        tensor(offsets, DType::I64, {offsets.ne[0]}, 8, op, "state byte offsets", expected_device);
    }
}

template <class T, class FieldView>
[[nodiscard]] T* state_field_base(const FieldView& value) {
    using Byte = std::conditional_t<
        std::is_const_v<std::remove_pointer_t<decltype(value.records.storage.data)>>,
        const std::byte, std::byte>;
    return reinterpret_cast<T*>(static_cast<Byte*>(value.records.storage.data) +
                                value.field.tensor.region.offset);
}

inline std::size_t bytes(std::uint64_t count, std::size_t width, const char* op) {
    if (count > std::numeric_limits<std::size_t>::max() / width)
        throw std::overflow_error(std::string(op) + ": byte count overflows");
    return static_cast<std::size_t>(count) * width;
}

inline Range range(const void* pointer, std::size_t count, const char* op) {
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (count > std::numeric_limits<std::uintptr_t>::max() - begin)
        throw std::overflow_error(std::string(op) + ": address range overflows");
    return {begin, begin + count};
}

inline Range range(const Tensor& tensor, const char* op) {
    return range(tensor.data, tensor.bytes(), op);
}

inline bool overlaps(Range a, Range b) noexcept { return a.begin < b.end && b.begin < a.end; }

inline Range strided_range(const Tensor& value, const char* op) {
    std::size_t extent = dtype_size(value.dtype);
    for (int axis = 0; axis < 4; ++axis) {
        if (value.ne[axis] <= 0 || value.nb[axis] <= 0)
            throw std::invalid_argument(std::string(op) + ": invalid strided extent");
        const auto tail = bytes(value.ne[axis] - 1, value.nb[axis], op);
        if (tail > std::numeric_limits<std::size_t>::max() - extent)
            throw std::overflow_error(std::string(op) + ": strided range overflows");
        extent += tail;
    }
    return range(value.data, extent, op);
}

inline void strided_tensor(const Tensor& value, DType dtype,
                           std::initializer_list<std::int32_t> dimensions, std::size_t alignment,
                           const char* op, const char* name, Device expected_device) {
    device(value.device, expected_device, op, name);
    const auto expected = Tensor::metadata(dtype, dimensions);
    if (value.dtype != dtype || !aligned(value.data, alignment))
        throw std::invalid_argument(std::string(op) + ": invalid " + name);
    std::array<int, 4> axes{};
    int count       = 0;
    const auto word = dtype_size(dtype);
    for (int axis = 0; axis < 4; ++axis) {
        if (value.ne[axis] != expected.ne[axis] || value.ne[axis] <= 0 || value.nb[axis] <= 0 ||
            value.nb[axis] % word)
            throw std::invalid_argument(std::string(op) + ": invalid strides for " + name);
        if (value.ne[axis] > 1) axes[count++] = axis;
    }
    std::sort(axes.begin(), axes.begin() + count,
              [&](int a, int b) { return value.nb[a] < value.nb[b]; });
    std::size_t extent = word;
    for (int i = 0; i < count; ++i) {
        const int axis    = axes[i];
        const auto stride = static_cast<std::size_t>(value.nb[axis]);
        if (stride < extent)
            throw std::invalid_argument(std::string(op) + ": overlapping elements in " + name);
        const auto tail = bytes(value.ne[axis] - 1, stride, op);
        if (tail > std::numeric_limits<std::size_t>::max() - extent)
            throw std::overflow_error(std::string(op) + ": tensor extent overflows");
        extent += tail;
    }
    (void)range(value.data, extent, op);
}

inline void disjoint(Range a, Range b, const char* op) {
    if (overlaps(a, b))
        throw std::invalid_argument(std::string(op) + ": operands must not overlap");
}

inline void disjoint(std::span<const Range> ranges, const char* op) {
    for (std::size_t i = 0; i < ranges.size(); ++i)
        for (std::size_t j = 0; j < i; ++j) disjoint(ranges[i], ranges[j], op);
}

// Dense blocks separated by one explicit outer stride. Unlike a bounding interval, this
// representation does not claim ownership of the gaps occupied by other StateImage fields.
struct Blocks {
    Range first{};
    std::size_t stride = 0;
    std::size_t count  = 0;

    Blocks() = default;

    Blocks(Range first, std::size_t stride, std::size_t count)
        : first(first), stride(stride), count(count) {}
};

inline Blocks blocks(const Tensor& value, const char* op) {
    std::size_t inner = dtype_size(value.dtype);
    int axis          = 0;
    for (; axis < 4; ++axis) {
        if (value.ne[axis] <= 0) throw std::invalid_argument(std::string(op) + ": empty blocks");
        if (value.ne[axis] != 1 && value.nb[axis] != static_cast<std::int64_t>(inner)) break;
        inner = bytes(value.ne[axis], inner, op);
    }
    if (axis == 4) return {range(value.data, inner, op), inner, 1};
    if (value.nb[axis] < static_cast<std::int64_t>(inner))
        throw std::invalid_argument(std::string(op) + ": blocks overlap");
    for (int outer = axis + 1; outer < 4; ++outer)
        if (value.ne[outer] != 1)
            throw std::invalid_argument(std::string(op) +
                                        ": only one strided outer axis is allowed");
    const auto stride = static_cast<std::size_t>(value.nb[axis]);
    const auto extent = bytes(value.ne[axis] - 1, stride, op);
    const auto first  = range(value.data, inner, op);
    (void)range(reinterpret_cast<const void*>(first.end), extent, op);
    return {first, stride, static_cast<std::size_t>(value.ne[axis])};
}

inline bool overlaps(Blocks a, Blocks b) noexcept {
    if (a.count == 0 || b.count == 0 || a.first.begin == a.first.end ||
        b.first.begin == b.first.end)
        return false;
    const auto a_end = a.first.end + (a.count - 1) * a.stride;
    const auto b_end = b.first.end + (b.count - 1) * b.stride;
    if (a_end <= b.first.begin || b_end <= a.first.begin) return false;
    std::size_t i = 0, j = 0;
    while (i < a.count && j < b.count) {
        if (overlaps(a.first, b.first)) return true;
        if (a.first.end <= b.first.begin) {
            const auto step = (b.first.begin - a.first.end) / a.stride + 1;
            if (step >= a.count - i) return false;
            i += step;
            a.first.begin += step * a.stride;
            a.first.end += step * a.stride;
        } else {
            const auto step = (a.first.begin - b.first.end) / b.stride + 1;
            if (step >= b.count - j) return false;
            j += step;
            b.first.begin += step * b.stride;
            b.first.end += step * b.stride;
        }
    }
    return false;
}

inline bool overlaps(Blocks a, Range b) noexcept {
    return overlaps(a, Blocks{b, b.end - b.begin, 1});
}

inline void paged_plane(const ConstPagedPlaneView& value, DType dtype,
                        std::initializer_list<std::int32_t> dimensions, std::size_t alignment,
                        const char* op, const char* name, Device expected_device) {
    device(value.device(), expected_device, op, name);
    const auto expected  = Tensor::metadata(dtype, dimensions);
    const auto& layout   = value.plane.tensor;
    const auto& region   = layout.region;
    const auto& geometry = value.plane.geometry;
    const auto& backing  = value.records.storage;
    if (!backing.data || layout.dtype != dtype ||
        !std::equal(layout.shape.begin(), layout.shape.end(), expected.ne) ||
        region.bytes != expected.bytes() || region.offset > backing.bytes ||
        region.bytes > backing.bytes - region.offset ||
        !aligned(static_cast<const std::byte*>(backing.data) + region.offset, alignment) ||
        geometry.dtype != dtype || geometry.leading_extent != expected.ne[0] ||
        geometry.head_extent != expected.ne[2] || geometry.tokens_per_row == 0 ||
        value.records.page_tokens == 0 ||
        value.records.page_tokens % geometry.tokens_per_row != 0 ||
        value.records.page_tokens / geometry.tokens_per_row !=
            static_cast<std::uint32_t>(expected.ne[1]) ||
        value.plane.head_stride_bytes != static_cast<std::size_t>(expected.nb[2])) {
        throw std::invalid_argument(std::string(op) + ": invalid " + name);
    }
    const auto& offsets = value.records.page_offsets;
    if (offsets.ne[0] <= 0 || offsets.ne[1] <= 0) {
        throw std::invalid_argument(std::string(op) + ": empty page offset table");
    }
    tensor(offsets, DType::I64, {offsets.ne[0], offsets.ne[1]}, 8, op, "page byte offsets",
           expected_device);
}

inline void same_paged_records(const ConstPagedPlaneView& a, const ConstPagedPlaneView& b,
                               const char* op, const char* name) {
    const auto& x = a.records;
    const auto& y = b.records;
    if (x.storage.data != y.storage.data || x.storage.bytes != y.storage.bytes ||
        x.storage.device != y.storage.device || x.page_tokens != y.page_tokens ||
        x.page_offsets.data != y.page_offsets.data ||
        x.page_offsets.dtype != y.page_offsets.dtype ||
        !std::equal(x.page_offsets.ne, x.page_offsets.ne + 4, y.page_offsets.ne) ||
        !std::equal(x.page_offsets.nb, x.page_offsets.nb + 4, y.page_offsets.nb)) {
        throw std::invalid_argument(std::string(op) + ": " + name +
                                    " must share the same records and page offset table");
    }
}

inline void slotted_tensor(const Tensor& value, DType dtype,
                           std::initializer_list<std::int32_t> dimensions, int slot_axis,
                           std::size_t alignment, const char* op, const char* name,
                           Device expected_device) {
    device(value.device, expected_device, op, name);
    const auto expected = Tensor::metadata(dtype, dimensions);
    if (value.dtype != dtype || !aligned(value.data, alignment))
        throw std::invalid_argument(std::string(op) + ": invalid " + name);
    for (int axis = 0; axis < 4; ++axis) {
        if (value.ne[axis] != expected.ne[axis] || value.ne[axis] <= 0 ||
            (axis < slot_axis && value.nb[axis] != expected.nb[axis]) ||
            (axis > slot_axis && value.ne[axis] != 1))
            throw std::invalid_argument(std::string(op) + ": invalid " + name);
    }
    if (value.nb[slot_axis] < expected.nb[slot_axis] ||
        value.nb[slot_axis] % static_cast<std::int64_t>(alignment) != 0)
        throw std::invalid_argument(std::string(op) + ": invalid slot stride for " + name);
    (void)blocks(value, op);
}

} // namespace ninfer::ops::common::operand

namespace ninfer::ops::common {

struct Fp8WeightGeometry {
    std::uint64_t code_plane_bytes;
    std::uint64_t scale_plane_offset;
    std::uint64_t scale_plane_bytes;
    std::uint64_t required_payload_bytes;
};

Fp8WeightGeometry validate_fp8_weight(const Weight& weight, const char* operation);

struct Nvfp4WeightGeometry {
    std::uint64_t code_plane_bytes;
    std::uint64_t scale_plane_offset;
    std::uint64_t scale_plane_bytes;
    std::uint64_t required_payload_bytes;
};

Nvfp4WeightGeometry validate_nvfp4_weight(const Weight& weight, const char* operation);

} // namespace ninfer::ops::common
