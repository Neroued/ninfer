#pragma once

#include "core/layout.h"
#include "core/tensor.h"

#include <cstddef>
#include <cstdint>

namespace ninfer {

struct GdnReplayRecordSpec {
    std::int32_t layers          = 0;
    std::int32_t record_capacity = 0;
    std::int32_t width           = 0;
    std::int32_t conv_channels   = 0;
    std::int32_t qk_heads        = 0;
    std::int32_t value_heads     = 0;
    std::int32_t key_dim         = 0;
    std::int32_t value_dim       = 0;
};

namespace detail {
struct GdnReplayRecordAccess;
}

class GdnReplayRecordLayout {
public:
    GdnReplayRecordSpec spec;
    TensorRegion conv;

    [[nodiscard]] std::size_t payload_bytes() const noexcept;
    // Compact rows within each existing plane. Plane bases and record capacity are preserved.
    [[nodiscard]] GdnReplayRecordLayout with_width(std::int32_t width) const;

private:
    TensorRegion key_, delta_, alpha_;

    friend struct detail::GdnReplayRecordAccess;
    friend GdnReplayRecordLayout plan_gdn_replay_records(LayoutBuilder&,
                                                         const GdnReplayRecordSpec&);
};

[[nodiscard]] GdnReplayRecordLayout plan_gdn_replay_records(LayoutBuilder& builder,
                                                            const GdnReplayRecordSpec& spec);

class GdnRecurrenceRecordView {
    // Raw represented BF16 keys, FP32 solved corrections, and FP32 decay scalars.
    Tensor key_, delta_, alpha_;

    GdnRecurrenceRecordView(Tensor key, Tensor delta, Tensor alpha)
        : key_(key), delta_(delta), alpha_(alpha) {}

    friend class GdnReplayRecords;
    friend struct detail::GdnReplayRecordAccess;
};

struct GdnReplayRecordLayer {
    Tensor conv; // BF16 [conv_channels, width, rows]
    GdnRecurrenceRecordView recurrence;
};

/**
 * Bound non-owning all-layer ReplaySSM transition records.
 *
 * Layer and physical record row share the outer index `layer * record_capacity + row` in every
 * plane. The object owns no allocation and stores no active-row or valid-prefix metadata.
 */
class GdnReplayRecords {
public:
    Tensor conv;
    GdnReplayRecordSpec spec;

    GdnReplayRecords() = default;
    GdnReplayRecords(MemorySpan backing, const GdnReplayRecordLayout& layout);

    [[nodiscard]] GdnReplayRecordLayer layer(std::int32_t layer, std::int32_t rows) const;

private:
    GdnRecurrenceRecordView recurrence_{{}, {}, {}};

    friend struct detail::GdnReplayRecordAccess;
};

namespace detail {

// The record producer, fold consumer and layout checks share the private payload.
struct GdnReplayRecordAccess {
    template <class Plane>
    struct Planes {
        const Plane& key;
        const Plane& delta;
        const Plane& alpha;
    };

    static Planes<TensorRegion> planes(const GdnReplayRecordLayout& layout) {
        return {layout.key_, layout.delta_, layout.alpha_};
    }

    static Planes<Tensor> planes(const GdnRecurrenceRecordView& view) {
        return {view.key_, view.delta_, view.alpha_};
    }

    static Planes<Tensor> planes(const GdnReplayRecords& records) {
        return planes(records.recurrence_);
    }
};

} // namespace detail

} // namespace ninfer
