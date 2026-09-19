#include "core/weight_view.h"

#include <algorithm>
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
                               std::span<const std::uint64_t> shape, std::uint64_t divisors) {
    if (divisors == 0 || (divisors != 1 && format != QType::NVFP4)) {
        throw std::invalid_argument("only an NVFP4 plane carries more than one divisor");
    }
    WeightGeometry out;
    out.format        = format;
    out.divisor_count = divisors;
    out.layout        = layout;
    out.shape.assign(shape.begin(), shape.end());
    out.elements = weight_element_count(shape);
    if (layout == QuantLayout::Contiguous) {
        out.bytes = out.code_bytes = mul(out.elements, word_bytes(format));
        if (shape.size() == 2) {
            out.padded_columns     = shape[1];
            out.code_bytes_per_row = mul(shape[1], word_bytes(format));
        }
        return out;
    }
    if (shape.size() != 2) { throw std::invalid_argument("quantized weight must be a matrix"); }
    const auto n       = shape[0];
    const auto k       = shape[1];
    out.padded_columns = k;
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
        out.padded_columns      = aligned(k, 128);
        const auto groups       = out.padded_columns / out.group_size;
        out.code_bytes_per_row  = mul(groups, 32);
        out.high_bytes_per_row  = mul(groups, high_per_group);
        out.scale_bytes_per_row = mul(groups, 2);
        out.code_bytes          = mul(n, out.code_bytes_per_row);
        out.high_offset         = aligned(out.code_bytes, 256);
        out.high_bytes          = mul(n, out.high_bytes_per_row);
        out.scale_offset        = add(out.high_offset, aligned(out.high_bytes, 256));
    } else if (layout == QuantLayout::RowScale) {
        if (format != QType::FP8_E4M3FN_ROW_BF16) {
            throw std::invalid_argument("RowScale requires row-scaled FP8");
        }
        out.group_size          = k;
        out.code_bytes_per_row  = k;
        out.scale_bytes_per_row = 2;
        out.code_bytes          = out.elements;
        out.scale_offset        = aligned(out.code_bytes, 256);
    } else if (layout == QuantLayout::BlockScaleK16M128x4) {
        if (format != QType::NVFP4 || n % 128 || k % 64) {
            throw std::invalid_argument("NVFP4 BlockScale requires N%128=0 and K%64=0");
        }
        out.group_size          = 16;
        out.code_bytes_per_row  = k / 2;
        out.scale_bytes_per_row = k / 16;
        out.code_bytes          = out.elements / 2;
        out.scale_offset        = aligned(out.code_bytes, 256);
    } else {
        throw std::invalid_argument("unknown quantized weight layout");
    }
    out.scale_bytes = mul(n, out.scale_bytes_per_row);
    out.bytes       = add(out.scale_offset, out.scale_bytes);
    if (format == QType::NVFP4) {
        if (n % divisors) {
            throw std::invalid_argument("divisor count must divide the plane's rows");
        }
        // Every other structural NVFP4 rule is checked here rather than left to the operator, and
        // this one belongs with them: a divisor that covered part of a 128-row scale tile would
        // make two rows of one tile disagree about which word they were quantised against.
        if (divisors > 1 && (n / divisors) % 128) {
            throw std::invalid_argument("each NVFP4 divisor must cover whole 128-row scale tiles");
        }
        out.divisor_offset = out.bytes;
        out.bytes          = add(out.bytes, mul(divisors, 4));
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

std::uint64_t weight_scale_offset(const WeightGeometry& geometry, std::uint64_t row,
                                  std::uint64_t group) {
    if (geometry.shape.size() != 2 || !geometry.group_size || row >= geometry.shape[0] ||
        group >= geometry.padded_columns / geometry.group_size) {
        throw std::invalid_argument("weight scale coordinate exceeds its parent");
    }
    if (geometry.layout == QuantLayout::BlockScaleK16M128x4) {
        const auto inner = row % 128;
        return geometry.scale_offset + (row / 128 * (geometry.shape[1] / 64) + group / 4) * 512 +
               inner % 32 * 16 + inner / 32 * 4 + group % 4;
    }
    return geometry.scale_offset + row * geometry.scale_bytes_per_row + group * 2;
}

WeightRowPlanes weight_row_planes(const WeightRegion& region) {
    validate_region(region);
    const auto& parent = *region.parent;
    const auto& g      = parent.geometry;
    if (!parent.data || g.shape.size() != 2 || region.begin % g.shape[1] ||
        region.end % g.shape[1]) {
        throw std::invalid_argument("row view requires resident complete logical rows");
    }
    WeightRowPlanes out;
    out.row_begin       = region.begin / g.shape[1];
    out.row_count       = (region.end - region.begin) / g.shape[1];
    out.code_row_bytes  = g.code_bytes_per_row;
    out.high_row_bytes  = g.high_bytes_per_row;
    out.scale_row_bytes = g.scale_bytes_per_row;
    out.codes           = parent.data + out.row_begin * out.code_row_bytes;
    if (g.high_bytes) {
        out.high = parent.data + g.high_offset + out.row_begin * out.high_row_bytes;
    }
    out.swizzled_scales = g.layout == QuantLayout::BlockScaleK16M128x4;
    if (g.scale_bytes) {
        out.scales = parent.data + g.scale_offset +
                     (out.swizzled_scales ? 0 : out.row_begin * out.scale_row_bytes);
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
    Tensor out(const_cast<std::byte*>(region.parent->data + offset), direct_dtype(g.format),
               internal_shape);
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
    if (g.shape.size() != 2 || view.shape[1] != g.shape[1]) {
        throw std::invalid_argument("quantized native Weight requires unchanged parent K");
    }
    const auto planes = weight_row_planes(region);
    if ((g.layout == QuantLayout::RowScale || g.layout == QuantLayout::BlockScaleK16M128x4) &&
        !is_complete_weight(view)) {
        throw std::invalid_argument(
            "this native Weight input requires a complete FP8/NVFP4 parent");
    }
    out.padded_shape[1]      = dimension(g.padded_columns);
    out.qdata                = planes.codes;
    out.qhigh                = planes.high;
    out.scales               = planes.scales;
    out.group_size           = static_cast<std::uint32_t>(g.group_size);
    out.group                = g.group_size ? dimension(g.group_size) : 0;
    out.weight_scale_divisor = region.parent->weight_scale_divisor;
    out.input_scale_divisor  = input_divisor;
    if (g.layout == QuantLayout::RowSplit) {
        out.scale_dtype = DType::FP16;
        out.scale_ne[0] = dimension(g.padded_columns / g.group_size);
        out.scale_ne[1] = out.n;
        out.scale_nb[0] = 2;
        out.scale_nb[1] = static_cast<std::int64_t>(g.scale_bytes_per_row);
        out.scale_nb[2] = out.scale_nb[3] = out.scale_nb[1] * out.n;
    } else if (g.layout == QuantLayout::RowScale) {
        out.scale_dtype = DType::BF16;
        out.scale_ne[0] = out.n;
        out.scale_nb[0] = 2;
        out.scale_nb[1] = out.scale_nb[2] = out.scale_nb[3] = static_cast<std::int64_t>(out.n) * 2;
    } else if (g.layout == QuantLayout::BlockScaleK16M128x4) {
        out.scale_dtype         = DType::FP8_E4M3FN;
        out.weight_divisors     = region.parent->data + g.divisor_offset;
        out.weight_divisor_rows = dimension(g.shape[0] / g.divisor_count);
    }
    return out;
}

} // namespace ninfer
