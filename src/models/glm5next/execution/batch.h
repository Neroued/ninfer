#pragma once

#include "core/kda_replay_records.h"
#include "models/common/parallel.h"
#include <variant>

namespace ninfer::models::glm5next::execution {
struct BatchView {
    std::int32_t tokens, rows;
    Tensor ids, positions, valid_columns, source_offsets, destination_offsets;
    Tensor table_rows, record_offsets, selection_rows;
    std::span<const IndexRange> owners;
    std::uint32_t max_visible_tokens;
};

struct SequenceUpdate {
    std::int64_t source, destination;
};

struct StepUpdate {};

struct RecordWindow {
    const KdaReplayRecords& kda;
};

using StateUpdate = std::variant<SequenceUpdate, StepUpdate, RecordWindow>;

} // namespace ninfer::models::glm5next::execution
