#pragma once

#include "core/layout.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <limits>
#include <stdexcept>

namespace ninfer {

inline constexpr std::int64_t kInvalidRecordOffset = -1;

// Rows select complete records by arena-relative byte offset. Device selectors are I64 [B].
// Every live offset selects a borrowed complete record. The caller guarantees that its field
// addresses fit the record and satisfy the declared alignments; selectors are not read back.
struct RecordBatchView {
    MemorySpan storage;
    Tensor byte_offsets;
};

struct ConstRecordBatchView {
    ConstMemorySpan storage;
    Tensor byte_offsets;

    ConstRecordBatchView() = default;

    ConstRecordBatchView(ConstMemorySpan storage, Tensor byte_offsets)
        : storage(storage), byte_offsets(byte_offsets) {}

    ConstRecordBatchView(const RecordBatchView& view)
        : storage(view.storage), byte_offsets(view.byte_offsets) {}
};

// A field or field slice within one record; layers repeat at a fixed record-relative stride.
struct StateFieldLayout {
    TensorRegion tensor;
    std::uint32_t layers     = 1;
    std::size_t layer_stride = 0;
    std::array<std::int64_t, 4> byte_strides{};

    StateFieldLayout() = default;

    StateFieldLayout(TensorRegion tensor, std::uint32_t layers, std::size_t layer_stride)
        : tensor(tensor), layers(layers), layer_stride(layer_stride) {
        const auto& shape = tensor.shape;
        const auto value = Tensor::metadata(tensor.dtype, {shape[0], shape[1], shape[2], shape[3]});
        std::copy(std::begin(value.nb), std::end(value.nb), byte_strides.begin());
    }

    [[nodiscard]] Tensor metadata() const {
        const auto& shape = tensor.shape;
        auto value = Tensor::metadata(tensor.dtype, {shape[0], shape[1], shape[2], shape[3]});
        std::copy(byte_strides.begin(), byte_strides.end(), std::begin(value.nb));
        return value;
    }

    [[nodiscard]] Tensor bind(MemorySpan record) const {
        auto value        = metadata();
        const auto memory = tensor.region.bind(record);
        value.data        = memory.data;
        value.device      = memory.device;
        return value;
    }

    [[nodiscard]] StateFieldLayout layer(std::uint32_t index) const {
        if (index >= layers) { throw std::out_of_range("State field layer is out of range"); }
        auto result = *this;
        result.tensor.region.offset += static_cast<std::size_t>(index) * layer_stride;
        result.layers = 1;
        return result;
    }

    [[nodiscard]] StateFieldLayout slice(int axis, std::int32_t start, std::int32_t count) const {
        if (axis < 0 || axis >= 4 || start < 0 || count <= 0 || start > tensor.shape[axis] ||
            count > tensor.shape[axis] - start) {
            throw std::out_of_range("State field slice is out of range");
        }
        auto result               = *this;
        const auto advance        = static_cast<std::size_t>(start) * byte_strides[axis];
        result.tensor.shape[axis] = count;
        result.tensor.region.offset += advance;
        result.tensor.region.alignment = std::gcd(tensor.region.alignment, advance);
        std::size_t bytes              = dtype_size(tensor.dtype);
        for (int i = 0; i < 4; ++i) {
            bytes += static_cast<std::size_t>(result.tensor.shape[i] - 1) * byte_strides[i];
        }
        result.tensor.region.bytes = bytes;
        return result;
    }
};

struct StateFieldView {
    RecordBatchView records;
    StateFieldLayout field;

    [[nodiscard]] StateFieldView layer(std::uint32_t index) const {
        return {records, field.layer(index)};
    }

    [[nodiscard]] StateFieldView slice(int axis, std::int32_t start, std::int32_t count) const {
        return {records, field.slice(axis, start, count)};
    }

    [[nodiscard]] Tensor tensor(std::int64_t byte_offset, std::uint32_t layer = 0) const {
        if (byte_offset < 0) { throw std::out_of_range("State record offset is negative"); }
        return field.layer(layer).bind(
            records.storage.subspan(static_cast<std::size_t>(byte_offset)));
    }
};

struct ConstStateFieldView {
    ConstRecordBatchView records;
    StateFieldLayout field;

    ConstStateFieldView() = default;

    ConstStateFieldView(ConstRecordBatchView records, StateFieldLayout field)
        : records(records), field(field) {}

    ConstStateFieldView(const StateFieldView& view) : records(view.records), field(view.field) {}

    [[nodiscard]] ConstStateFieldView layer(std::uint32_t index) const {
        auto result  = *this;
        result.field = field.layer(index);
        return result;
    }

    [[nodiscard]] ConstStateFieldView slice(int axis, std::int32_t start,
                                            std::int32_t count) const {
        return {records, field.slice(axis, start, count)};
    }

    [[nodiscard]] ConstMemorySpan memory(std::int64_t byte_offset, std::uint32_t layer = 0) const {
        if (byte_offset < 0) { throw std::out_of_range("State record offset is negative"); }
        return field.layer(layer).tensor.region.bind(
            records.storage.subspan(static_cast<std::size_t>(byte_offset)));
    }
};

// Page tables contain arena-relative I64 offsets, never physical page numbers. Every live
// entry selects a borrowed complete record whose planes fit and satisfy their field alignments.
struct PagedRecordView {
    MemorySpan storage;
    Tensor page_offsets;
    std::uint32_t page_tokens = 0;
};

struct ConstPagedRecordView {
    ConstMemorySpan storage;
    Tensor page_offsets;
    std::uint32_t page_tokens = 0;

    ConstPagedRecordView() = default;

    ConstPagedRecordView(ConstMemorySpan storage, Tensor page_offsets, std::uint32_t page_tokens)
        : storage(storage), page_offsets(page_offsets), page_tokens(page_tokens) {}

    ConstPagedRecordView(const PagedRecordView& view)
        : storage(view.storage), page_offsets(view.page_offsets), page_tokens(view.page_tokens) {}
};

} // namespace ninfer
