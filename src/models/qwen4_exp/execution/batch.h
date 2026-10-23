#pragma once

#include "core/tensor.h"
#include "core/gdn_replay_records.h"

#include <variant>

namespace ninfer::models::qwen4_exp::execution {

struct BatchView {
    std::int32_t tokens, batch;
    Tensor ids, positions, rope_positions, valid_columns;
    Tensor source_offsets, destination_offsets, table_rows;
    Tensor base_positions, record_offsets;
    std::uint32_t max_visible_tokens;
};

struct SequenceState {
    std::int64_t source, destination;
};

struct StepUpdate {};

struct VerifyRecords {
    const GdnReplayRecords& gdn;
};

using StateUpdate = std::variant<SequenceState, StepUpdate, VerifyRecords>;

struct TextOutputs {
    Tensor sample;
    Tensor conditioning;
};

struct VisualEmbeddingView {
    Tensor features;
    Tensor columns;
};

} // namespace ninfer::models::qwen4_exp::execution
