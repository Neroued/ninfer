#pragma once

#include "core/tensor.h"
#include "ninfer/ops/ragged.h"
#include "ninfer/ops/relative_attention.h"
#include <vector>

#include <cstdint>
#include <variant>

namespace ninfer::models::inkling2::execution {

struct BatchView {
    std::int32_t tokens = 0;
    std::int32_t rows   = 0;
    Tensor ids;
    Tensor positions;
    Tensor valid_columns;
    Tensor source_offsets;
    Tensor destination_offsets;
    Tensor live_rows;
    Tensor main_tables;
    Tensor dspark_tables;
    std::uint32_t max_visible_tokens = 0;
};

struct SequenceUpdate {
    std::int64_t source      = 0;
    std::int64_t destination = 0;
};

struct PackedPrefillUpdate {
    ops::RaggedSequenceView ragged;
};

struct StepUpdate {};

struct VerifyLayerRecords {
    Tensor k_conv, v_conv, attn_conv, mlp_conv;
    ops::RelativeAttentionChunkKV swa_chunk;
};

struct RecordWindow {
    Tensor base_slots;
    std::vector<VerifyLayerRecords> layers;
};

using StateUpdate = std::variant<SequenceUpdate, StepUpdate, PackedPrefillUpdate, RecordWindow>;

} // namespace ninfer::models::inkling2::execution
