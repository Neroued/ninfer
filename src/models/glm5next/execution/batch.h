#pragma once

#include "core/kda_replay_records.h"
#include "models/common/parallel.h"
#include "ninfer/ops/ragged.h"

#include <array>
#include <optional>
#include <variant>

namespace ninfer::models::glm5next::execution {
struct AttentionBatchView {
    std::int32_t rows;
    Tensor positions, valid_columns, source_offsets, destination_offsets;
    Tensor table_rows, record_offsets, selection_rows, to_global, from_gather, live_rows;
    std::uint32_t max_visible_tokens;
};

struct BatchView {
    std::int32_t tokens, rows;
    Tensor ids, positions, valid_columns, source_offsets, destination_offsets, live_rows;
    AttentionBatchView attention;
    // Sequence execution has one owner and broadcasts its attention result. Decode uses
    // the fixed local capacity and an equal-size gather, independent of live owner counts.
    std::optional<std::uint32_t> sequence_owner;
};

struct SequenceUpdate {
    std::int64_t source, destination;
};

struct PackedPrefillUpdate {
    ops::RaggedSequenceView ragged;
    std::array<IndexRange, 2> owner_tokens;
    ops::RaggedSequenceView local_mla_ragged;
};

struct StepUpdate {};

struct RecordWindow {
    const KdaReplayRecords& kda;
};

using StateUpdate = std::variant<SequenceUpdate, StepUpdate, RecordWindow, PackedPrefillUpdate>;

} // namespace ninfer::models::glm5next::execution
