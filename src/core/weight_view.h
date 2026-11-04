#pragma once

#include "core/tensor.h"
#include "core/weight.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace ninfer {

// Existing dense Op ABI uses Weight axes directly; callers establish BF16/FP32 and layout.
inline Tensor as_dense(const Weight& weight) {
    const auto dtype = weight.qtype == QType::FP32 ? DType::FP32 : DType::BF16;
    auto* data       = const_cast<void*>(weight.qdata);
    switch (weight.ndim) {
    case 1:
        return Tensor(data, weight.device, dtype, {weight.shape[0]});
    case 2:
        return Tensor(data, weight.device, dtype, {weight.shape[0], weight.shape[1]});
    case 3:
        return Tensor(data, weight.device, dtype,
                      {weight.shape[0], weight.shape[1], weight.shape[2]});
    default:
        return Tensor(data, weight.device, dtype,
                      {weight.shape[0], weight.shape[1], weight.shape[2], weight.shape[3]});
    }
}

struct WeightLayoutParameters {
    std::optional<std::uint64_t> exception_count;
    friend bool operator==(const WeightLayoutParameters&, const WeightLayoutParameters&) = default;
};

struct Csf32Geometry {
    std::uint64_t bases_offset = 0, nibbles_offset = 0, offsets_offset = 0;
    std::uint64_t page_bases_offset = 0, exceptions_offset = 0, exception_count = 0;
    std::uint64_t blocks_per_matrix = 0, pages_per_matrix = 0;
    std::uint64_t offsets_per_matrix = 0, nibble_bytes_per_matrix = 0;
    friend bool operator==(const Csf32Geometry&, const Csf32Geometry&) = default;
};

// Fixed storage facts can be queried without retaining or allocating a logical shape.
struct WeightStorageGeometry {
    QType format                      = QType::BF16;
    QuantLayout layout                = QuantLayout::Contiguous;
    std::uint64_t elements            = 0;
    std::uint64_t bytes               = 0;
    std::uint64_t alignment           = 256;
    std::uint64_t padded_columns      = 0;
    std::uint64_t group_size          = 0;
    std::uint64_t code_bytes_per_row  = 0;
    std::uint64_t high_bytes_per_row  = 0;
    std::uint64_t scale_block_rows    = 1;
    std::uint64_t scale_row_bytes     = 0;
    std::uint64_t code_bytes          = 0;
    std::uint64_t high_offset         = 0;
    std::uint64_t high_bytes          = 0;
    std::uint64_t scale_offset        = 0;
    std::uint64_t scale_bytes         = 0;
    std::uint64_t matrix_scale_offset = 0;
    std::uint64_t matrix_scale_count  = 0;
    std::uint64_t row_bytes           = 0;
    std::uint64_t rows_per_page       = 0;
    std::uint64_t page_count          = 0;
    std::uint64_t scale_word_bytes    = 0;
    Csf32Geometry csf;
};

// Geometry describes one complete encoded parent. Regions address its logical elements;
// padding and scale words do not belong to that element domain.
struct WeightGeometry : WeightStorageGeometry {
    std::vector<std::uint64_t> shape;
};

[[nodiscard]] WeightStorageGeometry weight_storage_geometry(QType format, QuantLayout layout,
                                                            std::span<const std::uint64_t> shape,
                                                            WeightLayoutParameters parameters = {});
[[nodiscard]] WeightGeometry weight_geometry(QType format, QuantLayout layout,
                                             std::span<const std::uint64_t> shape,
                                             WeightLayoutParameters parameters = {});
[[nodiscard]] std::uint64_t weight_element_count(std::span<const std::uint64_t> shape);

struct Csf32Block {
    std::uint64_t row, column, columns, ordinal, page, offset_index, nibble_offset;
};

struct Csf32Page {
    std::uint64_t block_begin, block_end, offset_begin;
};

// Block/page indices and nibble offsets are relative to one matrix's CSF planes.
[[nodiscard]] Csf32Block csf32_block(const WeightGeometry&, std::uint64_t ordinal);
[[nodiscard]] std::uint64_t csf32_block_index(const WeightGeometry&, std::uint64_t row,
                                              std::uint64_t column);
[[nodiscard]] Csf32Page csf32_page(const WeightGeometry&, std::uint64_t page);

struct WeightParent {
    WeightGeometry geometry;
    Device device;
    const std::byte* data = nullptr;
    // Host matrix scalars borrowed from the backing owner; their arithmetic belongs to format.
    std::span<const float> matrix_scales;
};

[[nodiscard]] Csf32ScaleView csf32_scale_view(const WeightParent&, std::uint64_t matrix);

struct WeightRegion {
    const WeightParent* parent = nullptr;
    std::uint64_t begin        = 0;
    std::uint64_t end          = 0;
};

struct WeightView {
    std::vector<std::uint64_t> shape;
    std::vector<WeightRegion> parts;
};

// Coalesce adjacent logical regions without moving their bytes. Fails for a gather or
// multiple parents; those remain explicit inputs for a consumer supporting that form.
[[nodiscard]] WeightRegion contiguous_weight_region(const WeightView& view);
[[nodiscard]] bool is_complete_weight(const WeightView& view);
[[nodiscard]] std::uint64_t weight_scale_offset(const WeightGeometry& geometry, std::uint64_t row,
                                                std::uint64_t group);
// Byte offset of a complete NF4 encoded row within its paged parent.
[[nodiscard]] std::uint64_t weight_paged_row_offset(const WeightGeometry& geometry,
                                                    std::uint64_t row);

struct WeightRowPlanes {
    Device device;
    const std::byte* codes         = nullptr;
    const std::byte* high          = nullptr;
    const std::byte* scales        = nullptr;
    std::uint64_t row_begin        = 0;
    std::uint64_t row_count        = 0;
    std::uint64_t code_row_bytes   = 0;
    std::uint64_t high_row_bytes   = 0;
    std::uint64_t scale_row_bytes  = 0;
    std::uint64_t scale_block_rows = 1;
    // NVFP4 keeps the parent scale base; use its row origin and block geometry.
    bool swizzled_scales = false;
};

[[nodiscard]] WeightRowPlanes weight_row_planes(const WeightRegion& region);
[[nodiscard]] Tensor weight_tensor(const WeightView& view,
                                   std::initializer_list<std::int32_t> internal_shape);
// A matrix view preserves K and scale-block boundaries. NVFP4 views stay within one
// matrix-scale domain; block scales are rebased to the view's local coordinates.
[[nodiscard]] Weight native_weight(const WeightView& view, float input_scale = 0.0F);

} // namespace ninfer
