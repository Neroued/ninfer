#pragma once

#include "core/layout.h"
#include "core/tensor.h"

#include <cstdint>

namespace ninfer::targets::qwen3_6 {

struct RoundStateSpec {
    std::int32_t hidden         = 0;
    std::int32_t output_rows    = 0;
    std::uint32_t draft_window  = 0;
    std::int32_t stats_counters = 0;
};

struct RoundStateLayout {
    RoundStateSpec spec;
    TensorRegion token;
    TensorRegion pos;
    TensorRegion rope_pos;
    TensorRegion rope_delta;
    TensorRegion logits;
    TensorRegion verify_hidden;
    TensorRegion target_tokens;
    TensorRegion drafts;
    TensorRegion sampled_out;
    TensorRegion num_sampled;
    TensorRegion verify_ids;
    TensorRegion shifted_ids;
    TensorRegion positions;
    TensorRegion window_base;
    TensorRegion accepted;
    TensorRegion gdn_initial_slot;
    TensorRegion ar_pos;
    TensorRegion mtp_ar_hidden;
    TensorRegion stats;
    bool complete = false;
};

// The two planning calls expose one deliberate exact-target extension seam after verify_hidden.
// This lets a target retain its schedule-sized prefill activation at the established physical
// address without making that activation part of the family round contract.
[[nodiscard]] RoundStateLayout begin_round_state_layout(LayoutBuilder& builder,
                                                        const RoundStateSpec& spec);
void complete_round_state_layout(LayoutBuilder& builder, RoundStateLayout& layout);

struct RoundState {
    Tensor token;
    Tensor pos;
    Tensor rope_pos;
    Tensor rope_delta;
    Tensor logits;
    Tensor verify_hidden;
    Tensor target_tokens;
    Tensor drafts;
    Tensor sampled_out;
    Tensor num_sampled;
    Tensor verify_ids;
    Tensor shifted_ids;
    Tensor positions;
    Tensor window_base;
    Tensor accepted;
    Tensor gdn_initial_slot;
    Tensor ar_pos;
    Tensor mtp_ar_hidden;
    Tensor stats;

    RoundState() = default;
    RoundState(DeviceSpan backing, const RoundStateLayout& layout);
};

} // namespace ninfer::targets::qwen3_6
