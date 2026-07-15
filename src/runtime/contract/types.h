#pragma once

#include "ninfer/types.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace ninfer::runtime {

using ::ninfer::ExecutionOptions;
using ::ninfer::FinishReason;
using ::ninfer::OutputChannel;
using ::ninfer::RequestOptions;
using ::ninfer::SamplingParameters;
using ::ninfer::StopPolicy;
using ::ninfer::StopString;
using ::ninfer::TokenId;

struct OutputDecision {
    std::uint32_t accepted_tokens = 0;
    FinishReason finish_reason    = FinishReason::None;

    [[nodiscard]] bool finished() const noexcept { return finish_reason != FinishReason::None; }
};

struct RequestPlanSummary {
    std::uint32_t prompt_tokens           = 0;
    std::uint32_t reusable_prompt_tokens  = 0;
    std::uint32_t requested_output_tokens = 0;
    std::uint32_t effective_output_tokens = 0;
    FinishReason effective_limit_reason   = FinishReason::None;
    std::size_t transient_bytes           = 0;
    std::size_t transient_alignment       = 1;
};

struct BeginSummary {
    std::uint32_t prompt_tokens        = 0;
    std::uint32_t reused_prompt_tokens = 0;
};

struct GeneratedRound {
    std::span<const TokenId> tokens;
};

struct BeginResult {
    BeginSummary summary;
    GeneratedRound round;
};

struct GenerationSummary {
    std::optional<BeginSummary> begin;
    FinishReason finish_reason = FinishReason::None;
};

struct RoundBudget {
    std::uint32_t generated_tokens_remaining = 0;
};

} // namespace ninfer::runtime
