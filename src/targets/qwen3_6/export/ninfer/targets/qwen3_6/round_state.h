#pragma once

#include "core/layout.h"
#include "core/tensor.h"

#include <cstdint>
#include <optional>

namespace ninfer::targets::qwen3_6 {

struct RoundStateSpec {
    std::int32_t hidden        = 0;
    std::int32_t output_rows   = 0;
    std::uint32_t draft_window = 0;
    bool enable_mtp            = false;
};

struct SpeculativeRoundStateLayout {
    TensorRegion target_argmax;
    TensorRegion draft_tokens;
    TensorRegion round_tokens;
    TensorRegion produced_count;
    TensorRegion target_input_ids;
    TensorRegion target_positions;
    TensorRegion accepted_drafts;
    TensorRegion stats;
};

struct MtpRoundStateLayout {
    TensorRegion alignment_ids;
    TensorRegion position;
    TensorRegion ar_hidden;
};

struct RoundStateLayout {
    RoundStateSpec spec;
    TensorRegion token;
    TensorRegion pos;
    TensorRegion rope_pos;
    TensorRegion rope_delta;
    TensorRegion logits;
    TensorRegion verify_hidden;
    TensorRegion gdn_initial_slot;
    SpeculativeRoundStateLayout speculative;
    std::optional<MtpRoundStateLayout> mtp;
    bool complete = false;
};

// The two planning calls expose one deliberate exact-target extension seam after verify_hidden.
// This lets a target retain its schedule-sized prefill activation at the established physical
// address without making that activation part of the family round contract.
[[nodiscard]] RoundStateLayout begin_round_state_layout(LayoutBuilder& builder,
                                                        const RoundStateSpec& spec);
void complete_round_state_layout(LayoutBuilder& builder, RoundStateLayout& layout);

struct SpeculativeRoundState {
    Tensor target_argmax;
    Tensor draft_tokens;
    Tensor round_tokens;
    Tensor produced_count;
    Tensor target_input_ids;
    Tensor target_positions;
    Tensor accepted_drafts;
    Tensor stats;

    SpeculativeRoundState() = default;
    SpeculativeRoundState(DeviceSpan backing, const SpeculativeRoundStateLayout& layout);
};

struct MtpRoundState {
    Tensor alignment_ids;
    Tensor position;
    Tensor ar_hidden;

    MtpRoundState() = default;
    MtpRoundState(DeviceSpan backing, const MtpRoundStateLayout& layout);
};

struct RoundState {
    Tensor token;
    Tensor pos;
    Tensor rope_pos;
    Tensor rope_delta;
    Tensor logits;
    Tensor verify_hidden;
    Tensor gdn_initial_slot;
    SpeculativeRoundState speculative;
    std::optional<MtpRoundState> mtp;

    RoundState() = default;
    RoundState(DeviceSpan backing, const RoundStateLayout& layout);
};

} // namespace ninfer::targets::qwen3_6
