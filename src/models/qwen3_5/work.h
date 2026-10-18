#pragma once

#include "models/common/context/contract.h"
#include "models/common/qwen/frontend/prepared_prompt.h"
#include "runtime/contract/resources.h"
#include "runtime/executor/contract.h"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

namespace ninfer::models::qwen {
struct VisionExecutionInput;
}

namespace ninfer::models::qwen3_5 {

struct ExecutionRow {
    runtime::ExecutionMember member;
    context::StateId source_state;
    context::StateId destination_state;
    context::StateSelectors state_slots;
    context::TableRow main_table;
    std::optional<context::TableRow> backend_table;
    std::uint32_t main_frontier    = 0;
    std::uint32_t backend_frontier = 0;
    std::int32_t rope_delta        = 0;
    std::array<std::int32_t, 3> tail_rope_position{};
};

struct SamplingInput {
    ResolvedSamplingParameters parameters;
    bool constrained = false;
};

struct InputWindow {
    std::vector<TokenId> tokens;
    // Axis-major positions for the exact input window.
    std::vector<std::int32_t> rope_positions;
    // Forward mathematical segments inside one scheduling unit; excludes zero and window end.
    std::vector<std::uint32_t> execution_splits;
    std::shared_ptr<const qwen::VisionExecutionInput> vision;
    std::uint32_t prompt_begin = 0;
};

enum class ForwardPurpose : std::uint8_t { Prefill, Replay, Control };
enum class SamplePurpose : std::uint8_t { None, FirstToken, NextToken };

struct ForwardWork {
    ForwardPurpose purpose = ForwardPurpose::Prefill;
    std::vector<ExecutionRow> rows;
    std::vector<InputWindow> inputs;
    std::vector<std::vector<TokenId>> committed_tokens;
    SamplePurpose sample = SamplePurpose::None;
    std::vector<SamplingInput> sampling;
};

struct DecodeRow {
    ExecutionRow binding;
    TokenId anchor                        = 0;
    std::uint32_t draft_extent            = 0;
    std::uint32_t output_tokens_remaining = 0;
    SamplingInput sampling;
};

struct DecodeWork {
    std::vector<DecodeRow> rows;
};

// DFlash target features occupy a residency's persistent plane until this range is appended.
struct NormalizeWork {
    std::vector<ExecutionRow> rows;
};

struct ResidencyBinding {
    runtime::ExecutionMember member;
    ResolvedSamplingParameters sampling;
    std::vector<TokenId> committed_generated_tokens;
};

struct ContextWork {
    std::vector<ResidencyBinding> bindings;
    std::vector<runtime::ResidencyId> releases;
};

struct ScoreWork {
    ExecutionRow row;
    InputWindow input;
    std::vector<TokenId> targets;
    std::uint32_t first_column = 0;
};

struct Work {
    context::ContextActions context;
    std::variant<ForwardWork, DecodeWork, NormalizeWork, ContextWork, ScoreWork> operation;
};

struct GeneratedResult {
    std::vector<TokenId> tokens;
    std::vector<std::uint32_t> counts;
    std::vector<std::uint32_t> draft_extents;
    std::vector<std::uint32_t> accepted_drafts;
    std::vector<std::uint32_t> dead_positions;
    std::uint32_t row_stride = 1;
};

struct ForwardResult {
    std::vector<std::uint32_t> processed_tokens;
    std::optional<GeneratedResult> generated;
    double vision_seconds = 0.0;
};

struct ContextResult {
    std::vector<runtime::ContextTransferObservation> transfers;
};

struct NormalizeResult {};

struct ScoreResult {
    std::uint32_t processed_tokens = 0;
    std::vector<float> logprobs;
};

struct Result {
    std::variant<ForwardResult, GeneratedResult, NormalizeResult, ContextResult, ScoreResult>
        output;
    runtime::ExecutionTiming timing;
};

struct CommitRow {
    std::uint32_t accepted_tokens = 0;
    bool terminal                 = false;
    bool discarded                = false;
};

struct Commit {
    std::vector<CommitRow> rows;
    context::ContextActions context;
};

struct ExecutionTypes {
    using Work   = qwen3_5::Work;
    using Result = qwen3_5::Result;
    using Commit = qwen3_5::Commit;
};

} // namespace ninfer::models::qwen3_5
