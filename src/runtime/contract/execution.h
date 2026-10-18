#pragma once

#include "runtime/contract/request.h"
#include "runtime/contract/timing.h"
#include <array>
#include <compare>

namespace ninfer::runtime {

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
    // Copied unchanged from the corresponding OutputDecision; still relative to this row's
    // accepted span.
    std::optional<std::uint32_t> prefix_execution_split_after;
};

struct CommitComplete {
    std::array<GenerationTimings, kMaximumConcurrency> timings{};
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

} // namespace ninfer::runtime
