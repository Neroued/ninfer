#include "core/kda_replay_records.h"

#include <array>
#include <limits>
#include <stdexcept>

namespace ninfer {
namespace {
int outer_extent(const KdaReplayRecordSpec& s) {
    if (s.layers <= 0 || s.record_capacity <= 0 || s.record_capacity > 8 || s.width <= 0 ||
        s.conv_channels <= 0 || s.qk_heads <= 0 || s.value_heads < s.qk_heads ||
        s.value_heads % s.qk_heads || s.key_dim <= 0 || s.value_dim <= 0)
        throw std::invalid_argument("KDA replay: invalid record geometry");
    if (s.layers > std::numeric_limits<int>::max() / s.record_capacity)
        throw std::overflow_error("KDA replay: layer/row extent exceeds int32");
    return s.layers * s.record_capacity;
}
} // namespace

KdaReplayRecordLayout plan_kda_replay_records(LayoutBuilder& b, const KdaReplayRecordSpec& s) {
    const int outer = outer_extent(s);
    return {
        s,
        b.add_tensor(DType::BF16, {s.conv_channels, s.width, outer}, 256, "KDA conv records"),
        b.add_tensor(DType::BF16, {s.key_dim, s.qk_heads, s.width, outer}, 256, "KDA keys"),
        b.add_tensor(DType::BF16, {s.value_dim, s.value_heads, s.width, outer}, 256, "KDA values"),
        b.add_tensor(DType::FP32, {s.key_dim, s.value_heads, s.width, outer}, 256, "KDA decay"),
        b.add_tensor(DType::FP32, {s.value_heads, s.width, outer}, 256, "KDA beta")};
}

std::size_t KdaReplayRecordLayout::payload_bytes() const noexcept {
    return conv.region.bytes + key.region.bytes + value.region.bytes + log_decay.region.bytes +
           beta.region.bytes;
}

KdaReplayRecords::KdaReplayRecords(MemorySpan backing, const KdaReplayRecordLayout& layout)
    : spec(layout.spec) {
    LayoutBuilder builder;
    const auto expected = plan_kda_replay_records(builder, spec);
    const std::array actual{&layout.conv, &layout.key, &layout.value, &layout.log_decay,
                            &layout.beta};
    const std::array schema{&expected.conv, &expected.key, &expected.value, &expected.log_decay,
                            &expected.beta};
    std::size_t end = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const auto& a = *actual[i];
        const auto& e = *schema[i];
        if (a.dtype != e.dtype || a.shape != e.shape || a.region.bytes != e.region.bytes ||
            a.region.alignment != 256 || a.region.offset < end)
            throw std::invalid_argument("KDA replay: inconsistent or overlapping record layout");
        (void)a.bind(backing);
        end = a.region.offset + a.region.bytes;
    }
    conv      = layout.conv.bind(backing);
    key       = layout.key.bind(backing);
    value     = layout.value.bind(backing);
    log_decay = layout.log_decay.bind(backing);
    beta      = layout.beta.bind(backing);
}

KdaReplayRecordLayer KdaReplayRecords::layer(std::int32_t index, std::int32_t rows) const {
    (void)outer_extent(spec);
    if (index < 0 || index >= spec.layers || rows <= 0 || rows > spec.record_capacity)
        throw std::out_of_range("KDA replay: layer or row count out of range");
    const int begin = index * spec.record_capacity;
    return {conv.slice(2, begin, rows), key.slice(3, begin, rows), value.slice(3, begin, rows),
            log_decay.slice(3, begin, rows), beta.slice(2, begin, rows)};
}
} // namespace ninfer
