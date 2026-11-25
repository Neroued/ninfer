#include "core/kda_replay_records.h"

#include <array>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer {
namespace {

constexpr std::size_t kRecordAlignment = 256;

void require_positive(std::int32_t value, const char* label) {
    if (value <= 0) {
        throw std::invalid_argument(std::string("KDA replay ") + label + " must be positive");
    }
}

std::int32_t checked_outer_extent(const KdaReplayRecordSpec& spec) {
    const auto layers   = static_cast<std::int64_t>(spec.layers);
    const auto capacity = static_cast<std::int64_t>(spec.record_capacity);
    if (layers > std::numeric_limits<std::int32_t>::max() / capacity) {
        throw std::overflow_error("KDA replay layer/row extent exceeds int32");
    }
    return static_cast<std::int32_t>(layers * capacity);
}

void validate_spec(const KdaReplayRecordSpec& spec) {
    require_positive(spec.layers, "layer count");
    require_positive(spec.record_capacity, "record capacity");
    require_positive(spec.width, "width");
    require_positive(spec.conv_channels, "convolution channel count");
    require_positive(spec.qk_heads, "Q/K head count");
    require_positive(spec.value_heads, "value head count");
    require_positive(spec.key_dim, "key dimension");
    require_positive(spec.value_dim, "value dimension");
    if (spec.value_heads % spec.qk_heads != 0) {
        throw std::invalid_argument("KDA replay value heads must be grouped by Q/K heads");
    }
    (void)checked_outer_extent(spec);
}

void require_region(const TensorRegion& region, DType dtype,
                    const std::array<std::int32_t, 4>& shape, const char* label) {
    if (region.dtype != dtype || region.shape != shape ||
        region.region.alignment != kRecordAlignment ||
        region.region.offset % kRecordAlignment != 0) {
        throw std::logic_error(std::string("KDA replay ") + label +
                               " layout does not match its spec");
    }
    const auto expected = Tensor::metadata(dtype, {shape[0], shape[1], shape[2], shape[3]});
    if (region.region.bytes != expected.bytes()) {
        throw std::logic_error(std::string("KDA replay ") + label +
                               " layout has an inconsistent byte size");
    }
}

void require_disjoint(const TensorRegion& a, const TensorRegion& b) {
    const std::size_t a_begin = a.region.offset;
    const std::size_t b_begin = b.region.offset;
    if (a.region.bytes > std::numeric_limits<std::size_t>::max() - a_begin ||
        b.region.bytes > std::numeric_limits<std::size_t>::max() - b_begin) {
        throw std::overflow_error("KDA replay region end overflows size_t");
    }
    const std::size_t a_end = a_begin + a.region.bytes;
    const std::size_t b_end = b_begin + b.region.bytes;
    if (a_begin < b_end && b_begin < a_end) {
        throw std::logic_error("KDA replay record planes overlap");
    }
}

void validate_layout(const KdaReplayRecordLayout& layout) {
    validate_spec(layout.spec);
    const auto [key, delta, alpha] = detail::KdaReplayRecordAccess::planes(layout);
    const std::int32_t outer       = checked_outer_extent(layout.spec);
    require_region(layout.conv, DType::BF16,
                   {layout.spec.conv_channels, layout.spec.width, outer, 1}, "conv");
    require_region(key, DType::BF16,
                   {layout.spec.key_dim, layout.spec.qk_heads, layout.spec.width, outer}, "key");
    require_region(delta, DType::FP32,
                   {layout.spec.value_dim, layout.spec.value_heads, layout.spec.width, outer},
                   "delta");
    require_region(alpha, DType::FP32,
                   {layout.spec.key_dim, layout.spec.value_heads, layout.spec.width, outer},
                   "alpha");

    const TensorRegion* regions[] = {&layout.conv, &key, &delta, &alpha};
    for (std::size_t i = 0; i < std::size(regions); ++i) {
        for (std::size_t j = 0; j < i; ++j) { require_disjoint(*regions[i], *regions[j]); }
    }
}

} // namespace

KdaReplayRecordLayout plan_kda_replay_records(LayoutBuilder& builder,
                                              const KdaReplayRecordSpec& spec) {
    validate_spec(spec);
    const std::int32_t outer = checked_outer_extent(spec);

    KdaReplayRecordLayout layout;
    layout.spec = spec;
    layout.conv = builder.add_tensor(DType::BF16, {spec.conv_channels, spec.width, outer},
                                     kRecordAlignment, "KDA replay conv records");
    layout.key_ = builder.add_tensor(DType::BF16, {spec.key_dim, spec.qk_heads, spec.width, outer},
                                     kRecordAlignment, "KDA replay key records");
    layout.delta_ =
        builder.add_tensor(DType::FP32, {spec.value_dim, spec.value_heads, spec.width, outer},
                           kRecordAlignment, "KDA replay delta records");
    layout.alpha_ =
        builder.add_tensor(DType::FP32, {spec.key_dim, spec.value_heads, spec.width, outer},
                           kRecordAlignment, "KDA replay alpha records");
    return layout;
}

std::size_t KdaReplayRecordLayout::payload_bytes() const noexcept {
    return conv.region.bytes + key_.region.bytes + delta_.region.bytes + alpha_.region.bytes;
}

KdaReplayRecordLayout KdaReplayRecordLayout::with_width(std::int32_t width) const {
    validate_layout(*this);
    if (width < 1 || width > spec.width)
        throw std::invalid_argument("KDA replay: requested width exceeds record layout");
    auto narrowed  = spec;
    narrowed.width = width;
    LayoutBuilder builder;
    auto result                 = plan_kda_replay_records(builder, narrowed);
    result.conv.region.offset   = conv.region.offset;
    result.key_.region.offset   = key_.region.offset;
    result.delta_.region.offset = delta_.region.offset;
    result.alpha_.region.offset = alpha_.region.offset;
    return result;
}

KdaReplayRecords::KdaReplayRecords(MemorySpan backing, const KdaReplayRecordLayout& layout)
    : spec(layout.spec) {
    validate_layout(layout);
    const auto [key, delta, alpha] = detail::KdaReplayRecordAccess::planes(layout);
    conv                           = layout.conv.bind(backing);
    recurrence_                    = {key.bind(backing), delta.bind(backing), alpha.bind(backing)};
}

KdaReplayRecordLayer KdaReplayRecords::layer(std::int32_t layer_index, std::int32_t rows) const {
    validate_spec(spec);
    if (layer_index < 0 || layer_index >= spec.layers) {
        throw std::out_of_range("KDA replay layer index out of range");
    }
    if (rows <= 0 || rows > spec.record_capacity) {
        throw std::out_of_range("KDA replay active row count out of range");
    }
    const std::int32_t outer_begin = layer_index * spec.record_capacity;
    return KdaReplayRecordLayer{
        .conv       = conv.slice(2, outer_begin, rows).view({spec.conv_channels, spec.width, rows}),
        .recurrence = {recurrence_.key_.slice(3, outer_begin, rows),
                       recurrence_.delta_.slice(3, outer_begin, rows),
                       recurrence_.alpha_.slice(3, outer_begin, rows)},
    };
}

} // namespace ninfer
