#include "core/weight_bank.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace ninfer {
namespace {

void check_expert(const WeightBankView& bank, std::uint32_t expert) {
    if (!bank.parent || expert >= bank.layout.region.count) {
        throw std::out_of_range("weight bank expert is outside its domain");
    }
}

} // namespace

WeightBankLayout describe_weight_bank(const WeightGeometry& g, const WeightBankRegion& r) {
    if (g.shape.size() < 2 || g.layout == QuantLayout::RowPagedG16_4K ||
        g.layout == QuantLayout::RowScale || !r.count || !r.rows || r.row_stride < r.rows) {
        throw std::invalid_argument("weight bank requires independently addressable matrices");
    }
    const auto total_rows = g.elements / g.shape.back();
    if (r.row_begin >= total_rows || r.rows > total_rows - r.row_begin ||
        (r.count - 1) > (total_rows - r.row_begin - r.rows) / r.row_stride ||
        r.rows > std::numeric_limits<std::int32_t>::max() ||
        g.shape.back() > std::numeric_limits<std::int32_t>::max()) {
        throw std::invalid_argument("weight bank exceeds its parent or native matrix domain");
    }
    const auto alignment = g.format == QType::NVFP4 ? 128 : g.scale_block_rows;
    if (r.row_begin % alignment || r.rows % alignment || r.row_stride % alignment) {
        throw std::invalid_argument("weight bank splits a scale tile");
    }
    WeightBankLayout out;
    out.region       = r;
    out.format       = g.format;
    out.layout       = g.layout;
    out.columns      = g.shape.back();
    const auto plane = [&](std::uint64_t offset, std::uint64_t bytes, std::uint64_t block) {
        return WeightBankPlane{offset + r.row_begin / block * bytes, r.rows / block * bytes,
                               r.row_stride / block * bytes};
    };
    out.codes = plane(0, g.code_bytes_per_row, 1);
    if (g.high_bytes) { out.high = plane(g.high_offset, g.high_bytes_per_row, 1); }
    if (g.scale_bytes) {
        out.scales = plane(g.scale_offset, g.scale_row_bytes, g.scale_block_rows);
    }
    if (g.divisor_count) {
        const auto domain      = g.shape[g.shape.size() - 2];
        const auto last        = r.row_begin + (r.count - 1) * r.row_stride + r.rows - 1;
        const bool same_domain = r.row_begin / domain == last / domain;
        if (r.row_begin % domain + r.rows > domain || (!same_domain && r.row_stride % domain)) {
            throw std::invalid_argument("weight bank crosses matrix divisor domains");
        }
        out.has_divisors   = true;
        out.divisor_begin  = r.row_begin / domain;
        out.divisor_stride = same_domain ? 0 : r.row_stride / domain;
    }
    return out;
}

WeightBankView bind_weight_bank(const WeightParent& parent, const WeightBankLayout& layout) {
    if (!parent.data || layout != describe_weight_bank(parent.geometry, layout.region) ||
        parent.weight_divisors.size() != parent.geometry.divisor_count) {
        throw std::invalid_argument("weight bank backing differs from its layout");
    }
    for (const auto value : parent.weight_divisors) {
        if (!std::isfinite(value) || value <= 0) {
            throw std::invalid_argument("weight bank divisor must be finite and positive");
        }
    }
    return {&parent, layout};
}

ConstMemorySpan WeightBankView::plane(const WeightBankPlane& plane, std::uint32_t expert) const {
    check_expert(*this, expert);
    if (!plane.expert_bytes) { return {nullptr, 0, parent->device}; }
    const auto offset = plane.offset + expert * plane.expert_stride;
    if (offset > parent->geometry.bytes || plane.expert_bytes > parent->geometry.bytes - offset) {
        throw std::out_of_range("weight bank plane exceeds its backing");
    }
    return {parent->data + offset, static_cast<std::size_t>(plane.expert_bytes), parent->device};
}

float WeightBankView::divisor(std::uint32_t expert) const {
    check_expert(*this, expert);
    if (!layout.has_divisors) { return 1.0F; }
    return parent->weight_divisors[layout.divisor_begin + expert * layout.divisor_stride];
}

Weight matrix_weight(const WeightBankView& bank, std::uint32_t expert, float input_divisor) {
    check_expert(bank, expert);
    const auto& r    = bank.layout.region;
    const auto k     = bank.layout.columns;
    const auto begin = (r.row_begin + expert * r.row_stride) * k;
    return native_weight({{r.rows, k}, {{bank.parent, begin, begin + r.rows * k}}}, input_divisor);
}

ExpertMatrixDeviceView device_weight_bank(const WeightBankView& bank) {
    if (!bank.parent || bank.parent->device.type() != DeviceType::CUDA) {
        throw std::invalid_argument("device weight bank requires CUDA backing");
    }
    const auto& layout = bank.layout;
    ExpertMatrixDeviceView out;
    out.codes        = static_cast<const std::byte*>(bank.plane(layout.codes, 0).data);
    out.high         = static_cast<const std::byte*>(bank.plane(layout.high, 0).data);
    out.scales       = static_cast<const std::byte*>(bank.plane(layout.scales, 0).data);
    out.code_stride  = layout.codes.expert_stride;
    out.high_stride  = layout.high.expert_stride;
    out.scale_stride = layout.scales.expert_stride;
    if (layout.has_divisors) {
        out.divisors = reinterpret_cast<const float*>(bank.parent->data +
                                                      bank.parent->geometry.divisor_offset) +
                       layout.divisor_begin;
        out.divisor_stride = layout.divisor_stride;
    }
    return out;
}

ExpertBankDeviceView device_expert_bank(const WeightBankView& gate_up, const WeightBankView& down) {
    if (!gate_up.parent || !down.parent || gate_up.parent->device != down.parent->device ||
        gate_up.layout.region.count != down.layout.region.count) {
        throw std::invalid_argument("expert bank matrices differ in device or count");
    }
    return {device_weight_bank(gate_up), device_weight_bank(down), gate_up.layout.region.count,
            gate_up.parent->device};
}

ExpertMatrixEntryLayout plan_expert_matrix(LayoutBuilder& builder, const WeightBankLayout& matrix) {
    const auto add = [&](const WeightBankPlane& plane) {
        return plane.expert_bytes ? builder.add(plane.expert_bytes, 16, "expert plane")
                                  : LayoutRegion{};
    };
    return {add(matrix.codes), add(matrix.high), add(matrix.scales)};
}

ExpertMatrixDeviceView indexed_weight_matrix(const WeightParent& parent,
                                             const std::uint64_t* offsets, const float* divisors,
                                             ConstMemorySpan storage) {
    const auto& g = parent.geometry;
    if (!parent.data || parent.device.type() != DeviceType::CUDA || !offsets ||
        parent.device != storage.device || g.shape.size() != 2)
        throw std::invalid_argument("indexed expert matrix requires a resident encoded matrix");
    return {parent.data,
            g.high_bytes ? parent.data + g.high_offset : nullptr,
            g.scale_bytes ? parent.data + g.scale_offset : nullptr,
            0,
            0,
            0,
            offsets,
            divisors,
            divisors ? 1U : 0U,
            storage};
}

} // namespace ninfer
