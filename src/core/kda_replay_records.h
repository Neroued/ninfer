#pragma once

#include "core/layout.h"

namespace ninfer {

struct KdaReplayRecordSpec {
    std::int32_t layers = 0, record_capacity = 0, width = 0;
    std::int32_t conv_channels = 0, qk_heads = 0, value_heads = 0;
    std::int32_t key_dim = 0, value_dim = 0;
};

struct KdaReplayRecordLayout {
    KdaReplayRecordSpec spec;
    TensorRegion conv, key, value, log_decay, beta;

    [[nodiscard]] std::size_t payload_bytes() const noexcept;
};

[[nodiscard]] KdaReplayRecordLayout plan_kda_replay_records(LayoutBuilder& builder,
                                                            const KdaReplayRecordSpec& spec);

struct KdaReplayRecordLayer {
    Tensor conv, key, value, log_decay, beta;
};

// Non-owning frame planes. Their outer row is layer * record_capacity + batch_row.
struct KdaReplayRecords {
    Tensor conv, key, value, log_decay, beta;
    KdaReplayRecordSpec spec;

    KdaReplayRecords() = default;
    KdaReplayRecords(MemorySpan backing, const KdaReplayRecordLayout& layout);
    [[nodiscard]] KdaReplayRecordLayer layer(std::int32_t layer, std::int32_t rows) const;
};

} // namespace ninfer
