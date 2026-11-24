#include "core/gdn_replay_records.h"

#include <array>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer {
namespace {

constexpr std::size_t kRecordAlignment        = 256;
constexpr std::int32_t kMaximumRecordCapacity = 8;

void require_positive(std::int32_t value, const char* label) {
    if (value <= 0) {
        throw std::invalid_argument(std::string("GDN replay ") + label + " must be positive");
    }
}

std::int32_t checked_outer_extent(const GdnReplayRecordSpec& spec) {
    const auto layers   = static_cast<std::int64_t>(spec.layers);
    const auto capacity = static_cast<std::int64_t>(spec.record_capacity);
    if (layers > std::numeric_limits<std::int32_t>::max() / capacity) {
        throw std::overflow_error("GDN replay layer/row extent exceeds int32");
    }
    return static_cast<std::int32_t>(layers * capacity);
}

void validate_spec(const GdnReplayRecordSpec& spec) {
    require_positive(spec.layers, "layer count");
    require_positive(spec.record_capacity, "record capacity");
    require_positive(spec.width, "width");
    require_positive(spec.conv_channels, "convolution channel count");
    require_positive(spec.qk_heads, "Q/K head count");
    require_positive(spec.value_heads, "value head count");
    require_positive(spec.key_dim, "key dimension");
    require_positive(spec.value_dim, "value dimension");
    if (spec.record_capacity > kMaximumRecordCapacity) {
        throw std::invalid_argument("GDN replay record capacity exceeds eight rows");
    }
    if (spec.value_heads % spec.qk_heads != 0) {
        throw std::invalid_argument("GDN replay value heads must be grouped by Q/K heads");
    }
    (void)checked_outer_extent(spec);
}

void require_region(const TensorRegion& region, DType dtype,
                    const std::array<std::int32_t, 4>& shape, const char* label) {
    if (region.dtype != dtype || region.shape != shape ||
        region.region.alignment != kRecordAlignment) {
        throw std::logic_error(std::string("GDN replay ") + label +
                               " layout does not match its spec");
    }
    const auto expected = Tensor::metadata(dtype, {shape[0], shape[1], shape[2], shape[3]});
    if (region.region.bytes != expected.bytes()) {
        throw std::logic_error(std::string("GDN replay ") + label +
                               " layout has an inconsistent byte size");
    }
}

void require_disjoint(const TensorRegion& a, const TensorRegion& b) {
    const std::size_t a_begin = a.region.offset;
    const std::size_t b_begin = b.region.offset;
    if (a.region.bytes > std::numeric_limits<std::size_t>::max() - a_begin ||
        b.region.bytes > std::numeric_limits<std::size_t>::max() - b_begin) {
        throw std::overflow_error("GDN replay region end overflows size_t");
    }
    const std::size_t a_end = a_begin + a.region.bytes;
    const std::size_t b_end = b_begin + b.region.bytes;
    if (a_begin < b_end && b_begin < a_end) {
        throw std::logic_error("GDN replay record planes overlap");
    }
}

void validate_layout(const GdnReplayRecordLayout& layout) {
    validate_spec(layout.spec);
    const std::int32_t outer = checked_outer_extent(layout.spec);
    require_region(layout.conv, DType::BF16,
                   {layout.spec.conv_channels, layout.spec.width, outer, 1}, "conv");
    require_region(layout.key, DType::BF16,
                   {layout.spec.key_dim, layout.spec.qk_heads, layout.spec.width, outer}, "key");
    require_region(layout.value, DType::BF16,
                   {layout.spec.value_dim, layout.spec.value_heads, layout.spec.width, outer},
                   "value");
    require_region(layout.gate, DType::FP32, {2, layout.spec.value_heads, layout.spec.width, outer},
                   "gate");

    const TensorRegion* regions[] = {&layout.conv, &layout.key, &layout.value, &layout.gate};
    for (std::size_t i = 0; i < std::size(regions); ++i) {
        for (std::size_t j = 0; j < i; ++j) { require_disjoint(*regions[i], *regions[j]); }
    }
}

} // namespace

GdnReplayRecordLayout plan_gdn_replay_records(LayoutBuilder& builder,
                                              const GdnReplayRecordSpec& spec) {
    validate_spec(spec);
    const std::int32_t outer = checked_outer_extent(spec);

    GdnReplayRecordLayout layout;
    layout.spec = spec;
    layout.conv = builder.add_tensor(DType::BF16, {spec.conv_channels, spec.width, outer},
                                     kRecordAlignment, "GDN replay conv records");
    layout.key  = builder.add_tensor(DType::BF16, {spec.key_dim, spec.qk_heads, spec.width, outer},
                                     kRecordAlignment, "GDN replay key records");
    layout.value =
        builder.add_tensor(DType::BF16, {spec.value_dim, spec.value_heads, spec.width, outer},
                           kRecordAlignment, "GDN replay value records");
    layout.gate = builder.add_tensor(DType::FP32, {2, spec.value_heads, spec.width, outer},
                                     kRecordAlignment, "GDN replay gate records");
    return layout;
}

std::size_t GdnReplayRecordLayout::payload_bytes() const noexcept {
    return conv.region.bytes + key.region.bytes + value.region.bytes + gate.region.bytes;
}

GdnReplayRecords::GdnReplayRecords(MemorySpan backing, const GdnReplayRecordLayout& layout)
    : conv(layout.conv.bind(backing)), key(layout.key.bind(backing)),
      value(layout.value.bind(backing)), gate(layout.gate.bind(backing)), spec(layout.spec) {
    validate_layout(layout);
}

GdnReplayRecordLayer GdnReplayRecords::layer(std::int32_t layer_index, std::int32_t rows) const {
    validate_spec(spec);
    if (layer_index < 0 || layer_index >= spec.layers) {
        throw std::out_of_range("GDN replay layer index out of range");
    }
    if (rows <= 0 || rows > spec.record_capacity) {
        throw std::out_of_range("GDN replay active row count out of range");
    }
    const std::int32_t outer_begin = layer_index * spec.record_capacity;
    return GdnReplayRecordLayer{
        .conv  = conv.slice(2, outer_begin, rows).view({spec.conv_channels, spec.width, rows}),
        .key   = key.slice(3, outer_begin, rows),
        .value = value.slice(3, outer_begin, rows),
        .gate  = gate.slice(3, outer_begin, rows),
    };
}

} // namespace ninfer
