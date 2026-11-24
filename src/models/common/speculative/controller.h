#pragma once

#include "models/common/speculative/cost_model.h"
#include "runtime/contract/execution.h"

namespace ninfer::models::speculative {

inline constexpr std::uint32_t kMaxSpeculativeDrafts = 15;

struct SequenceSpeculativeState {
    void observe_speculative_round(std::uint32_t attempted_extent, std::uint32_t accepted_drafts,
                                   std::span<const float> draft_confidences = {});
    void observe_ordinary_round(std::optional<float> target_top1_prob);
    void predict_conditional_survival(SpeculativeBackend backend, std::uint32_t max_drafts,
                                      std::span<float> out_probs) const noexcept;
    [[nodiscard]] float draft_attempt_weight(std::uint32_t position) const noexcept;

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

// Record verification work even when the result is discarded at commit. Only an adopted
// result can update adaptive feedback. The Session supplies validated counts and statistics
// storage.
void record_speculative_commit(SpeculativeStats&, SequenceSpeculativeState* adaptive_state,
                               const runtime::CommitDecision&, bool could_speculate,
                               std::uint32_t proposed, std::uint32_t drafted,
                               std::uint32_t accepted, std::span<const float> draft_confidences,
                               std::optional<float> target_top1_probability);

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

struct DecodeRoundRequest {
    std::span<const runtime::RoundBudget> budgets;
    std::span<const std::uint32_t> extents;
    const RoundRoute* route = nullptr;
    bool record_online_cost = true;

    [[nodiscard]] bool uses_adaptive_length(SpeculativeLengthPolicy policy) const noexcept;
    [[nodiscard]] bool valid_membership(std::size_t rows) const noexcept;
    [[nodiscard]] std::uint32_t draft_budget(std::size_t row) const noexcept;
    [[nodiscard]] std::optional<DecodeRoundPlan>
    select(SpeculativeLengthPolicy, PhysicalBatchShape, ExecutionPath, const DecodeGraphRoutes*,
           const SpeculativeCostModel&, std::span<const RoundMember>,
           std::span<const AdaptiveRowInput>, std::uint32_t max_context) const;
};

struct NarrowedTargetPlan {
    RoundRoute target_route;
    std::vector<std::uint32_t> verify_drafts;
};

// Confidence at row * confidence_stride + j must come from the draft distribution before
// its draw. DFlash2 finalizes each row in prefix order before choosing the physical route.
[[nodiscard]] NarrowedTargetPlan
narrow_after_proposal(SpeculativeBackend backend, std::uint32_t max_drafts,
                      std::uint32_t owner_count, ExecutionPath path,
                      const DecodeGraphRoutes* routes, const SpeculativeCostModel& cost_model,
                      const DecodeRoundPlan& initial_plan, std::span<const AdaptiveRowInput> rows,
                      std::span<const float> live_draft_confidences,
                      std::uint32_t confidence_stride, std::uint32_t max_context);

} // namespace ninfer::models::speculative
