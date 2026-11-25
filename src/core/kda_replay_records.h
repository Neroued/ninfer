#pragma once

#include "core/layout.h"
#include "core/tensor.h"

#include <cstddef>
#include <cstdint>

namespace ninfer {

struct KdaReplayRecordSpec {
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
struct KdaReplayRecordAccess;
}

class KdaReplayRecordLayout {
public:
    KdaReplayRecordSpec spec;
    TensorRegion conv;

    [[nodiscard]] std::size_t payload_bytes() const noexcept;
    // Compact rows within each existing plane. Plane bases and record capacity are preserved.
    [[nodiscard]] KdaReplayRecordLayout with_width(std::int32_t width) const;

private:
    TensorRegion key_, delta_, alpha_;

    friend struct detail::KdaReplayRecordAccess;
    friend KdaReplayRecordLayout plan_kda_replay_records(LayoutBuilder&,
                                                         const KdaReplayRecordSpec&);
};

[[nodiscard]] KdaReplayRecordLayout plan_kda_replay_records(LayoutBuilder& builder,
                                                            const KdaReplayRecordSpec& spec);

class KdaRecurrenceRecordView {
    // Raw represented BF16 keys, FP32 solved corrections, and FP32 per-key-channel decays.
    Tensor key_, delta_, alpha_;

    KdaRecurrenceRecordView(Tensor key, Tensor delta, Tensor alpha)
        : key_(key), delta_(delta), alpha_(alpha) {}

    friend class KdaReplayRecords;
    friend struct detail::KdaReplayRecordAccess;
};

struct KdaReplayRecordLayer {
    Tensor conv; // BF16 [conv_channels, width, rows]
    KdaRecurrenceRecordView recurrence;
};

/**
 * Bound non-owning all-layer ReplaySSM transition records.
 *
 * Layer and physical record row share the outer index `layer * record_capacity + row` in every
 * plane. The object owns no allocation and stores no active-row or valid-prefix metadata.
 */
class KdaReplayRecords {
public:
    Tensor conv;
    KdaReplayRecordSpec spec;

    KdaReplayRecords() = default;
    KdaReplayRecords(MemorySpan backing, const KdaReplayRecordLayout& layout);

    [[nodiscard]] KdaReplayRecordLayer layer(std::int32_t layer, std::int32_t rows) const;

private:
    KdaRecurrenceRecordView recurrence_{{}, {}, {}};

    friend struct detail::KdaReplayRecordAccess;
};

namespace detail {

// The record producer, fold consumer and layout checks share the private payload.
struct KdaReplayRecordAccess {
    template <class Plane>
    struct Planes {
        const Plane& key;
        const Plane& delta;
        const Plane& alpha;
    };

    static Planes<TensorRegion> planes(const KdaReplayRecordLayout& layout) {
        return {layout.key_, layout.delta_, layout.alpha_};
    }

    static Planes<Tensor> planes(const KdaRecurrenceRecordView& view) {
        return {view.key_, view.delta_, view.alpha_};
    }

    static Planes<Tensor> planes(const KdaReplayRecords& records) {
        return planes(records.recurrence_);
    }
};

} // namespace detail

} // namespace ninfer
