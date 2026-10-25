#pragma once

#include "models/common/context/contract.h"
#include "models/common/context/directory/state.h"
#include "models/common/context/directory/history.h"
#include "models/qwen4_exp/execution_shape.h"
#include "models/common/qwen/vision/control.h"
#include "ninfer/types.h"
#include "runtime/contract/resources.h"
#include "runtime/executor/contract.h"

#include <array>
#include <memory>
#include <map>
#include <optional>
#include <variant>
#include <vector>

namespace ninfer::models::qwen {
struct VisionExecutionInput;
}

namespace ninfer::models::qwen4_exp {

using StateBindings  = std::map<std::uint32_t, context::StateSelectors>;
using ContextActions = std::map<std::uint32_t, context::ContextActions>;

[[nodiscard]] const context::ContextActions& local_actions(const ContextActions&,
                                                           std::uint32_t rank);

struct WorkRow {
    runtime::ExecutionMember member;
    StateBindings states;
    context::TableRow main_table, mtp_table;
    std::uint32_t main_frontier = 0;
    std::uint32_t mtp_frontier  = 0;
    std::int32_t rope_delta     = 0;
    std::array<TokenId, 2> ple_raw_tail{};
    std::vector<TokenId> ids;
    std::vector<TokenId> count_tokens;
    // Required boundaries within ids, measured from this input window's beginning.
    std::vector<std::uint32_t> window_ends;
    // Axis-major [3,T], where T is ids.size().
    std::vector<std::int32_t> rope_positions;
    std::array<std::int32_t, 3> previous_rope{};
    ResolvedSamplingParameters sampling;
    std::uint32_t sampling_position = 0;
    bool constrained                = false;
};

struct PlePrefetch {
    runtime::ExecutionMember member;
    std::uint32_t frontier = 0;
    std::array<TokenId, 2> raw_tail{};
    std::vector<TokenId> ids;
};

enum class ForwardKind : std::uint8_t { Prefill, Replay, Ordinary, Control };
enum class SampleKind : std::uint8_t { None, FirstToken, NextToken };

struct ForwardWork {
    ForwardKind kind  = ForwardKind::Prefill;
    SampleKind sample = SampleKind::None;
    std::vector<WorkRow> rows;
    std::optional<PlePrefetch> prefetch;
    std::optional<qwen::VisionWork> visual;
};

struct MtpWork {
    std::vector<WorkRow> rows;
    std::vector<std::uint32_t> draft_extents;
};

struct ResidencyBinding {
    runtime::ExecutionMember member;
    ResolvedSamplingParameters sampling;
    std::vector<TokenId> generated;
};

struct ContextWork {
    std::vector<ResidencyBinding> bindings;
    std::vector<runtime::ResidencyId> releases;
};

struct ScoreWork {
    WorkRow row;
    std::vector<TokenId> targets;
    std::uint32_t first_target_column = 0;
};

struct Work {
    ContextActions actions;
    std::variant<ForwardWork, MtpWork, ContextWork, ScoreWork> operation;
};

struct RowResult {
    std::uint32_t processed       = 0;
    std::uint32_t drafts          = 0;
    std::uint32_t accepted_drafts = 0;
    std::vector<TokenId> tokens;
    std::vector<TokenId> input_tokens;
    std::uint32_t dead_positions = 0;
};

struct ComputeResult {
    std::vector<RowResult> rows;
};

struct ContextResult {
    std::vector<runtime::ContextTransferObservation> transfers;
};

struct ScoreResult {
    std::uint32_t processed = 0;
    std::vector<float> logprobs;
};

using Result = std::variant<ComputeResult, ContextResult, ScoreResult>;

struct CommitRow {
    runtime::ExecutionMember member;
    StateBindings states;
    std::uint32_t accepted = 0;
    bool terminal          = false;
    bool cancelled         = false;
    bool failed            = false;
};

struct Commit {
    std::vector<CommitRow> rows;
    ContextActions actions;
};

struct ExecutionTypes {
    [[nodiscard]] static ExecutionWork describe_work(const qwen4_exp::Work&);
    using Work   = qwen4_exp::Work;
    using Result = qwen4_exp::Result;
    using Commit = qwen4_exp::Commit;
};

// Value-level execution requirements shared by the coordinator and rank. Residency ownership,
// physical ranges and current device stages are checked by RankProgram.
struct WorkContract {
    ExecutionShape shape;
    std::uint32_t vocabulary, public_tokens;
    bool mtp, vision;

    void validate(const Work&, runtime::FrameKind) const;
    void validate(const Work&, const Result&, const Commit&) const;
    void validate_commit(const Work&, std::span<const RowResult>, const Commit&) const;
    void validate(const Work&, const runtime::DecisionRequest&,
                  const runtime::DecisionInput&) const;
};

[[nodiscard]] std::span<const WorkRow> work_rows(const Work&);
void validate_decision_rows(const runtime::DecisionInput&, std::uint32_t stage,
                            std::uint32_t vocabulary, std::span<const std::uint32_t> positions);

} // namespace ninfer::models::qwen4_exp
