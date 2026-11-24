#pragma once

#include "runtime/contract/placement.h"
#include "runtime/contract/request.h"
#include <array>
#include <compare>

namespace ninfer::runtime {

enum class UnitReservationStatus : std::uint8_t {
    Granted,
    ContextUnavailable,
    ExecutionUnavailable,
};

struct LaneId {
    std::uint32_t value = 0;

    [[nodiscard]] friend constexpr bool operator==(LaneId, LaneId) noexcept  = default;
    [[nodiscard]] friend constexpr auto operator<=>(LaneId, LaneId) noexcept = default;
};

enum class ConsumeStatus : std::uint8_t {
    Consumed,
    InvariantMismatch,
};

enum class CommitDisposition : std::uint8_t {
    Active,
    Finishable,
    CancelledReleased,
    FailedReleased,
};

struct CommitDecision {
    std::uint32_t accepted_tokens = 0;
    bool terminal                 = false;
    bool cancelled                = false;
    bool failed                   = false;
};

struct ExecutionComplete {
    ExecutionSample sample;
    std::size_t row_count = 0;
};

struct BeginSummary {
    std::uint32_t prompt_tokens        = 0;
    std::uint32_t reused_prompt_tokens = 0;
    PrefixReusePath prefix_reuse_path  = PrefixReusePath::Root;

    [[nodiscard]] friend constexpr bool operator==(BeginSummary, BeginSummary) noexcept = default;
};

struct RoundBudget {
    std::uint32_t generated_tokens_remaining = 0;
};

struct PrefillBudgetPolicy {
    // Forward-token and sequence limits for one step, independent of physical ranks.
    std::uint32_t max_tokens_global           = 0;
    std::uint32_t max_tokens_per_placement    = 0;
    std::uint32_t max_sequences_per_step      = 1;
    std::uint32_t max_sequences_per_placement = 1;
    std::uint32_t chunk_alignment             = 64; // Alignment for a truncated model unit.
};

struct PrefillCandidateInfo {
    PlacementId placement{};
    std::uint32_t remaining_tokens = 0;     // Entire unprocessed prompt suffix.
    std::uint32_t unit_max_tokens  = 0;     // Current model window, including semantic boundaries.
    bool exclusive                 = false; // This window needs the whole step.
    bool zero_token_sample         = false; // The committed prompt tail needs its first sample.
};

struct PrefillGrant {
    std::uint32_t max_tokens = 0; // Forward-token cap for this row.
    bool constrained         = false;
};

} // namespace ninfer::runtime
