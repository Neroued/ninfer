#pragma once

#include "core/tensor.h"
#include "core/weight.h"

#include <cstddef>
#include <cstdint>
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

// Geometry describes one complete encoded parent. Regions address its logical elements;
// padding and scale words do not belong to that element domain.
struct WeightGeometry {
    QType format       = QType::BF16;
    QuantLayout layout = QuantLayout::Contiguous;
    std::vector<std::uint64_t> shape;
    std::uint64_t elements           = 0;
    std::uint64_t bytes              = 0;
    std::uint64_t alignment          = 256;
    std::uint64_t padded_columns     = 0;
    std::uint64_t group_size         = 0;
    std::uint64_t code_bytes_per_row = 0;
    std::uint64_t high_bytes_per_row = 0;
    std::uint64_t scale_block_rows   = 1;
    std::uint64_t scale_row_bytes    = 0;
    std::uint64_t code_bytes         = 0;
    std::uint64_t high_offset        = 0;
    std::uint64_t high_bytes         = 0;
    std::uint64_t scale_offset       = 0;
    std::uint64_t scale_bytes        = 0;
    std::uint64_t divisor_offset     = 0;
    std::uint64_t divisor_count      = 0;
    std::uint64_t row_bytes          = 0;
    std::uint64_t rows_per_page      = 0;
    std::uint64_t page_count         = 0;
};

[[nodiscard]] WeightGeometry weight_geometry(QType format, QuantLayout layout,
                                             std::span<const std::uint64_t> shape);
[[nodiscard]] std::uint64_t weight_element_count(std::span<const std::uint64_t> shape);

struct WeightParent {
    WeightGeometry geometry;
    Device device;
    const std::byte* data = nullptr;
    // Host metadata borrowed from the backing owner, including for CUDA parents.
    std::span<const float> weight_divisors;
};

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
// matrix divisor domain; block scales are rebased to the view's local coordinates.
[[nodiscard]] Weight native_weight(const WeightView& view, float input_divisor = 0.0F);

} // namespace ninfer
