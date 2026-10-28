#include "core/weight_view.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ninfer {
namespace {

std::uint64_t add(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) {
        throw std::overflow_error("weight geometry addition overflows u64");
    }
    return a + b;
}

std::uint64_t mul(std::uint64_t a, std::uint64_t b) {
    if (a && b > std::numeric_limits<std::uint64_t>::max() / a) {
        throw std::overflow_error("weight geometry multiplication overflows u64");
    }
    return a * b;
}

std::uint64_t aligned(std::uint64_t n, std::uint64_t alignment) {
    return add(n, alignment - 1) / alignment * alignment;
}

std::int32_t dimension(std::uint64_t n) {
    if (!n || n > std::numeric_limits<std::int32_t>::max()) {
        throw std::invalid_argument("weight dimension is outside the native i32 domain");
    }
    return static_cast<std::int32_t>(n);
}

std::uint64_t word_bytes(QType format) {
    switch (format) {
    case QType::BF16:
        return 2;
    case QType::FP32:
    case QType::INT32:
        return 4;
    default:
        throw std::invalid_argument("direct weight requires BF16, FP32 or INT32");
    }
}

DType direct_dtype(QType format) {
    switch (format) {
    case QType::BF16:
        return DType::BF16;
    case QType::FP32:
        return DType::FP32;
    case QType::INT32:
        return DType::I32;
    default:
        throw std::invalid_argument("quantized weight cannot be bound as a direct tensor");
    }
}

void validate_region(const WeightRegion& region) {
    if (!region.parent || region.begin >= region.end ||
        region.end > region.parent->geometry.elements) {
        throw std::invalid_argument("weight region exceeds its parent element domain");
    }
}

} // namespace

std::uint64_t weight_element_count(std::span<const std::uint64_t> shape) {
    if (shape.size() > 16) { throw std::invalid_argument("weight rank exceeds 16"); }
    std::uint64_t count = 1;
    for (const auto dim : shape) {
        if (!dim) { throw std::invalid_argument("weight dimensions must be positive"); }
        count = mul(count, dim);
    }
    return count;
}

WeightGeometry weight_geometry(QType format, QuantLayout layout,
                               std::span<const std::uint64_t> shape) {
    WeightGeometry out;
    out.format = format;
    out.layout = layout;
    out.shape.assign(shape.begin(), shape.end());
    out.elements = weight_element_count(shape);
    if (layout == QuantLayout::Contiguous) {
        out.bytes = out.code_bytes = mul(out.elements, word_bytes(format));
        if (shape.size() >= 2) {
            out.padded_columns     = shape.back();
            out.code_bytes_per_row = mul(shape.back(), word_bytes(format));
        }
        return out;
    }
    const bool nvfp4_bank = layout == QuantLayout::BatchedBlockScaleK16M128x4;
    const bool fp8_block =
        layout == QuantLayout::BlockScaleM128K128 || layout == QuantLayout::BlockScaleM64K64;
    if ((nvfp4_bank && shape.size() != 3) ||
        (fp8_block && shape.size() != 2 && shape.size() != 3) ||
        (!nvfp4_bank && !fp8_block && shape.size() != 2)) {
        throw std::invalid_argument("quantized weight rank differs from its layout");
    }
    const auto n        = shape[shape.size() - 2];
    const auto k        = shape.back();
    const auto matrices = shape.size() == 3 ? shape[0] : 1;
    out.padded_columns  = k;
    if (layout == QuantLayout::RowPagedG16_4K) {
        if (format != QType::NF4_G16_E4M3FN || k % 16) {
            throw std::invalid_argument("paged NF4 requires K%16=0");
        }
        out.alignment          = 4096;
        out.group_size         = 16;
        out.code_bytes_per_row = k / 2;
        out.scale_row_bytes    = k / 16;
        out.row_bytes          = add(k / 2, k / 16);
        out.rows_per_page      = 4096 / out.row_bytes;
        if (!out.rows_per_page) { throw std::invalid_argument("NF4 row exceeds page"); }
        out.page_count     = add(n, out.rows_per_page - 1) / out.rows_per_page;
        out.code_bytes     = mul(n, out.code_bytes_per_row);
        out.scale_bytes    = mul(n, out.scale_row_bytes);
        out.divisor_offset = mul(out.page_count, 4096);
        out.divisor_count  = 1;
        out.bytes          = add(out.divisor_offset, 4);
        return out;
    }
    if (layout == QuantLayout::RowSplit) {
        std::uint64_t high_per_group = 0;
        switch (format) {
        case QType::Q4_G64_FP16:
            out.group_size = 64;
            break;
        case QType::Q5_G64_FP16:
            out.group_size = 64;
            high_per_group = 8;
            break;
        case QType::Q6_G64_FP16:
            out.group_size = 64;
            high_per_group = 16;
            break;
        case QType::Q8_G32_FP16:
            out.group_size = 32;
            break;
        default:
            throw std::invalid_argument("RowSplit requires a grouped integer format");
        }
        out.padded_columns     = aligned(k, 128);
        const auto groups      = out.padded_columns / out.group_size;
        out.code_bytes_per_row = mul(groups, 32);
        out.high_bytes_per_row = mul(groups, high_per_group);
        out.scale_row_bytes    = mul(groups, 2);
        out.code_bytes         = mul(n, out.code_bytes_per_row);
        out.high_offset        = aligned(out.code_bytes, 256);
        out.high_bytes         = mul(n, out.high_bytes_per_row);
        out.scale_offset       = add(out.high_offset, aligned(out.high_bytes, 256));
    } else if (layout == QuantLayout::RowScale) {
        if (format != QType::FP8_E4M3FN_ROW_BF16) {
            throw std::invalid_argument("RowScale requires row-scaled FP8");
        }
        out.group_size         = k;
        out.code_bytes_per_row = k;
        out.scale_row_bytes    = 2;
        out.code_bytes         = out.elements;
        out.scale_offset       = aligned(out.code_bytes, 256);
    } else if (fp8_block) {
        const auto block = layout == QuantLayout::BlockScaleM64K64 ? 64U : 128U;
        const auto expected =
            block == 64 ? QType::FP8_E4M3FN_BLOCK64X64_BF16 : QType::FP8_E4M3FN_BLOCK128X128_BF16;
        if (format != expected || n % block || k % block) {
            throw std::invalid_argument(
                "block-scaled FP8 format or matrix block alignment differs");
        }
        out.group_size         = block;
        out.scale_block_rows   = block;
        out.code_bytes_per_row = k;
        out.scale_row_bytes    = mul(k / block, 2);
        out.code_bytes         = out.elements;
        out.scale_offset       = aligned(out.code_bytes, 256);
    } else if (layout == QuantLayout::BlockScaleK16M128x4 || nvfp4_bank) {
        if (format != QType::NVFP4 || n % 128 || k % 64) {
            throw std::invalid_argument("NVFP4 BlockScale requires N%128=0 and K%64=0");
        }
        out.group_size         = 16;
        out.code_bytes_per_row = k / 2;
        out.scale_row_bytes    = k / 16;
        out.code_bytes         = out.elements / 2;
        out.scale_offset       = aligned(out.code_bytes, 256);
    } else {
        throw std::invalid_argument("unknown quantized weight layout");
    }
    out.scale_bytes = mul(mul(matrices, n / out.scale_block_rows), out.scale_row_bytes);
    out.bytes       = add(out.scale_offset, out.scale_bytes);
    if (format == QType::NVFP4) {
        out.divisor_offset = out.bytes;
        out.divisor_count  = matrices;
        out.bytes          = add(out.bytes, mul(matrices, 4));
    }
    return out;
}

WeightRegion contiguous_weight_region(const WeightView& view) {
    if (view.parts.empty()) { throw std::invalid_argument("weight view has no regions"); }
    WeightRegion merged = view.parts.front();
    validate_region(merged);
    for (std::size_t i = 1; i < view.parts.size(); ++i) {
        const auto& next = view.parts[i];
        validate_region(next);
        if (next.parent != merged.parent || next.begin != merged.end) {
            throw std::invalid_argument("native input requires one contiguous parent region");
        }
        merged.end = next.end;
    }
    if (merged.end - merged.begin != weight_element_count(view.shape)) {
        throw std::invalid_argument("weight view shape differs from its regions");
    }
    return merged;
}

bool is_complete_weight(const WeightView& view) {
    if (view.parts.empty()) { return false; }
    const auto* parent = view.parts.front().parent;
    if (!parent || view.shape != parent->geometry.shape) { return false; }
    std::uint64_t end = 0;
    for (const auto& part : view.parts) {
        if (part.parent != parent || part.begin != end || part.end <= part.begin) { return false; }
        end = part.end;
    }
    return end == parent->geometry.elements;
}

std::uint64_t weight_paged_row_offset(const WeightGeometry& geometry, std::uint64_t row) {
    if (geometry.layout != QuantLayout::RowPagedG16_4K || geometry.shape.size() != 2 ||
        !geometry.rows_per_page || row >= geometry.shape[0]) {
        throw std::invalid_argument("invalid paged weight row");
    }
    return row / geometry.rows_per_page * 4096 + row % geometry.rows_per_page * geometry.row_bytes;
}

std::uint64_t weight_scale_offset(const WeightGeometry& geometry, std::uint64_t row,
                                  std::uint64_t group) {
    if (geometry.shape.size() < 2 || !geometry.group_size ||
        row >= geometry.elements / geometry.shape.back() ||
        group >= geometry.padded_columns / geometry.group_size) {
        throw std::invalid_argument("weight scale coordinate exceeds its parent");
    }
    if (geometry.layout == QuantLayout::RowPagedG16_4K) {
        return weight_paged_row_offset(geometry, row) + geometry.code_bytes_per_row + group;
    }
    if (geometry.layout == QuantLayout::BlockScaleK16M128x4 ||
        geometry.layout == QuantLayout::BatchedBlockScaleK16M128x4) {
        const auto inner = row % 128;
        return geometry.scale_offset +
               (row / 128 * (geometry.shape.back() / 64) + group / 4) * 512 + inner % 32 * 16 +
               inner / 32 * 4 + group % 4;
    }
    return geometry.scale_offset + row / geometry.scale_block_rows * geometry.scale_row_bytes +
           group * 2;
}

WeightRowPlanes weight_row_planes(const WeightRegion& region) {
    validate_region(region);
    const auto& parent = *region.parent;
    const auto& g      = parent.geometry;
    if (g.layout == QuantLayout::RowPagedG16_4K) {
        throw std::invalid_argument("paged weights require per-row encoded addressing");
    }
    if (!parent.data || g.shape.size() < 2 || region.begin % g.shape.back() ||
        region.end % g.shape.back()) {
        throw std::invalid_argument("row view requires resident complete logical rows");
    }
    WeightRowPlanes out;
    out.device           = parent.device;
    out.row_begin        = region.begin / g.shape.back();
    out.row_count        = (region.end - region.begin) / g.shape.back();
    out.code_row_bytes   = g.code_bytes_per_row;
    out.high_row_bytes   = g.high_bytes_per_row;
    out.scale_row_bytes  = g.scale_row_bytes;
    out.scale_block_rows = g.scale_block_rows;
    out.codes            = parent.data + out.row_begin * out.code_row_bytes;
    if (g.high_bytes) {
        out.high = parent.data + g.high_offset + out.row_begin * out.high_row_bytes;
    }
    out.swizzled_scales = g.layout == QuantLayout::BlockScaleK16M128x4 ||
                          g.layout == QuantLayout::BatchedBlockScaleK16M128x4;
    if (g.scale_bytes) {
        out.scales =
            parent.data + g.scale_offset +
            (out.swizzled_scales ? 0 : out.row_begin / out.scale_block_rows * out.scale_row_bytes);
    }
    return out;
}

Tensor weight_tensor(const WeightView& view, std::initializer_list<std::int32_t> internal_shape) {
    const auto region = contiguous_weight_region(view);
    const auto& g     = region.parent->geometry;
    if (g.layout != QuantLayout::Contiguous || !region.parent->data) {
        throw std::invalid_argument("direct tensor requires a resident contiguous weight");
    }
    const auto offset = mul(region.begin, word_bytes(g.format));
    Tensor out(const_cast<std::byte*>(region.parent->data + offset), region.parent->device,
               direct_dtype(g.format), internal_shape);
    if (static_cast<std::uint64_t>(out.numel()) != region.end - region.begin) {
        throw std::invalid_argument("native tensor shape differs from its logical weight");
    }
    return out;
}

Weight native_weight(const WeightView& view, float input_divisor) {
    const auto region = contiguous_weight_region(view);
    const auto& g     = region.parent->geometry;
    if (view.shape.size() != 2 || !region.parent->data) {
        throw std::invalid_argument("native Weight requires a resident logical matrix");
    }
    Weight out;
    out.device           = region.parent->device;
    out.payload          = region.parent->data;
    out.payload_bytes    = g.bytes;
    out.high_plane_bytes = g.high_bytes;
    out.qtype            = g.format;
    out.layout           = g.layout;
    out.ndim             = 2;
    out.n = out.shape[0] = out.padded_shape[0] = dimension(view.shape[0]);
    out.k = out.shape[1] = out.padded_shape[1] = dimension(view.shape[1]);
    if (g.layout == QuantLayout::Contiguous) {
        out.qdata = region.parent->data + mul(region.begin, word_bytes(g.format));
        return out;
    }
    if (g.shape.size() < 2 || view.shape[1] != g.shape.back()) {
        throw std::invalid_argument("quantized native Weight requires unchanged parent K");
    }
    const auto planes = weight_row_planes(region);
    if (g.layout == QuantLayout::RowScale && !is_complete_weight(view)) {
        throw std::invalid_argument("row-scaled FP8 native input requires a complete parent");
    }
    const auto row_alignment = g.format == QType::NVFP4 ? 128 : g.scale_block_rows;
    if (planes.row_begin % row_alignment || planes.row_count % row_alignment) {
        throw std::invalid_argument("native weight view splits a scale tile");
    }
    out.padded_shape[1]     = dimension(g.padded_columns);
    out.qdata               = planes.codes;
    out.qhigh               = planes.high;
    out.scales              = planes.scales;
    out.group_size          = static_cast<std::uint32_t>(g.group_size);
    out.group               = g.group_size ? dimension(g.group_size) : 0;
    out.input_scale_divisor = input_divisor;
    if (g.divisor_count) {
        const auto rows_per_matrix = g.shape[g.shape.size() - 2];
        const auto matrix          = planes.row_begin / rows_per_matrix;
        if (region.parent->weight_divisors.size() != g.divisor_count) {
            throw std::invalid_argument("native weight has incomplete divisor metadata");
        }
        if ((planes.row_begin + planes.row_count - 1) / rows_per_matrix != matrix) {
            throw std::invalid_argument("native weight view crosses matrix divisor domains");
        }
        out.weight_scale_divisor = region.parent->weight_divisors[matrix];
        if (!std::isfinite(out.weight_scale_divisor) || out.weight_scale_divisor <= 0) {
            throw std::invalid_argument("native weight requires a positive finite divisor");
        }
    }
    if (g.layout == QuantLayout::RowSplit) {
        out.scale_dtype = DType::FP16;
        out.scale_ne[0] = dimension(g.padded_columns / g.group_size);
        out.scale_ne[1] = out.n;
        out.scale_nb[0] = 2;
        out.scale_nb[1] = static_cast<std::int64_t>(g.scale_row_bytes);
        out.scale_nb[2] = out.scale_nb[3] = out.scale_nb[1] * out.n;
    } else if (g.layout == QuantLayout::RowScale) {
        out.scale_dtype = DType::BF16;
        out.scale_ne[0] = out.n;
        out.scale_nb[0] = 2;
        out.scale_nb[1] = out.scale_nb[2] = out.scale_nb[3] = static_cast<std::int64_t>(out.n) * 2;
    } else if (g.layout == QuantLayout::BlockScaleM128K128 ||
               g.layout == QuantLayout::BlockScaleM64K64) {
        out.scale_dtype = DType::BF16;
        out.scale_ne[0] = dimension(g.padded_columns / g.group_size);
        out.scale_ne[1] = out.n / dimension(g.scale_block_rows);
        out.scale_nb[0] = 2;
        out.scale_nb[1] = static_cast<std::int64_t>(g.scale_row_bytes);
        out.scale_nb[2] = out.scale_nb[3] = out.scale_nb[1] * out.scale_ne[1];
    } else if (planes.swizzled_scales) {
        out.layout      = QuantLayout::BlockScaleK16M128x4;
        out.scale_dtype = DType::FP8_E4M3FN;
        out.scales = region.parent->data + g.scale_offset + planes.row_begin * g.scale_row_bytes;
    }
    return out;
}

} // namespace ninfer
