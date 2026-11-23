#pragma once

#include "models/common/speculative/cost_model.h"

namespace ninfer::models::speculative {

inline constexpr std::uint32_t kMaxSpeculativeDrafts = 15;

struct SequenceSpeculativeState {
    void observe_speculative_round(std::uint32_t attempted_extent, std::uint32_t accepted_drafts,
                                   std::span<const float> draft_confidences = {});
    void observe_ordinary_round(std::optional<float> target_top1_prob);
    void predict_conditional_survival(std::uint32_t max_drafts,
                                      std::span<float> out_probs) const noexcept;

private:
    void decay(float factor) noexcept;

    std::array<float, kMaxSpeculativeDrafts> attempts_{};
    std::array<float, kMaxSpeculativeDrafts> accepts_{};
    std::optional<float> last_target_top1_prob_;
    // Unorm16 keeps the value state within 160 bytes. Observations stay in FP32.
    std::array<std::uint16_t, kMaxSpeculativeDrafts> last_draft_confidences_{};
    std::uint8_t last_draft_count_ = 0;
};

static_assert(sizeof(SequenceSpeculativeState) <= 160);

struct AdaptiveRowInput {
    RoundMember member;
    std::uint32_t frontier                = 0;
    const SequenceSpeculativeState* state = nullptr;
};

[[nodiscard]] DecodeRoundPlan
select_adaptive_round(SpeculativeBackend backend, std::uint32_t max_drafts,
                      std::uint32_t owner_count, PhysicalBatchShape shape, ExecutionPath path,
                      const DecodeGraphRoutes* routes, const SpeculativeCostModel& cost_model,
                      std::span<const AdaptiveRowInput> rows, std::uint32_t max_context);

struct NarrowedTargetPlan {
    RoundRoute target_route;
    std::vector<std::uint32_t> verify_drafts;
};

// Confidence at row * confidence_stride + j must come from the draft distribution
// before its draw. DFlash2 block distributions supplied here must be independent
// of the block's sampled IDs, across both positions and rows.
[[nodiscard]] NarrowedTargetPlan
narrow_after_proposal(SpeculativeBackend backend, std::uint32_t max_drafts,
                      std::uint32_t owner_count, ExecutionPath path,
                      const DecodeGraphRoutes* routes, const SpeculativeCostModel& cost_model,
                      const DecodeRoundPlan& initial_plan, std::span<const AdaptiveRowInput> rows,
                      std::span<const float> live_draft_confidences,
                      std::uint32_t confidence_stride, std::uint32_t max_context);

} // namespace ninfer::models::speculative
