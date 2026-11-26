#pragma once

#include "core/tensor.h"
#include "ninfer/ops/ragged.h"

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

using StateUpdate = std::variant<SequenceUpdate, StepUpdate, PackedPrefillUpdate>;

} // namespace ninfer::models::inkling2::execution
