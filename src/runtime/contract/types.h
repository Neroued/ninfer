#pragma once

#include "ninfer/types.h"

#include <cstddef>
#include <cstdint>
#include <optional>

namespace ninfer::runtime {

using ::ninfer::ExecutionOptions;
using ::ninfer::FinishReason;
using ::ninfer::OutputChannel;
using ::ninfer::RequestOptions;
using ::ninfer::SamplingParameters;
using ::ninfer::StopPolicy;
using ::ninfer::StopString;
using ::ninfer::TokenId;

enum class RoundContinuation : std::uint8_t {
    Continue,
    Finish,
};

struct StagedChoiceId {
    std::uint32_t value = 0;
};

struct OutputResolution {
    std::uint32_t committed_tokens = 0;
    RoundContinuation continuation = RoundContinuation::Continue;
    FinishReason finish_reason     = FinishReason::None;
    StagedChoiceId staged_choice;
};

enum class FinishDisposition : std::uint8_t {
    Resident,
    Invalid,
};

enum class ProgramState : std::uint8_t {
    Empty,
    Active,
    PendingRound,
    Resident,
    Invalid,
};

struct SequenceSummary {
    ProgramState state                = ProgramState::Empty;
    std::uint64_t epoch               = 0;
    std::uint32_t materialized_tokens = 0;
    std::uint32_t logical_tokens      = 0;
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

struct GenerationSummary {
    std::optional<BeginSummary> begin;
    std::uint32_t committed_output_tokens = 0;
    FinishReason finish_reason            = FinishReason::None;
    std::optional<FinishDisposition> sequence;
};

struct RoundBudget {
    std::uint32_t generated_tokens_remaining = 0;
};

} // namespace ninfer::runtime
