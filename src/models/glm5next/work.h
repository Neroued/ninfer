#pragma once

#include "models/common/context/contract.h"
#include "models/common/context/directory/state.h"
#include "models/common/context/directory/history.h"
#include "models/glm5next/execution_shape.h"
#include "models/glm5next/vision/control.h"
#include "models/common/parallel.h"
#include "runtime/executor/contract.h"

#include <map>
#include <variant>

namespace ninfer::models::glm5next {

using StateBindings  = std::map<std::uint32_t, context::StateSelectors>;
using ContextActions = std::map<std::uint32_t, context::ContextActions>;

struct WorkRow {
    runtime::ExecutionMember member;
    std::uint32_t attention_rank = 0;
    StateBindings states;
    context::TableRow main_table, mtp_table;
    std::uint32_t main_frontier = 0, mtp_frontier = 0;
    std::vector<TokenId> ids, count_tokens;
    std::vector<std::uint32_t> window_ends;
    ResolvedSamplingParameters sampling;
    std::uint32_t sampling_position = 0;
    bool constrained                = false;
};

// Physical rows are owner-contiguous; all public results and decisions retain logical order.
struct BatchPlan {
    std::vector<std::uint32_t> members;
    std::vector<IndexRange> owners;
    bool operator==(const BatchPlan&) const = default;
};

[[nodiscard]] BatchPlan plan_batch(std::span<const WorkRow>, std::uint32_t ranks);

enum class ForwardKind : std::uint8_t { Prefill, Replay, Ordinary, Control };
enum class SampleKind : std::uint8_t { None, FirstToken, NextToken };

struct ForwardWork {
    ForwardKind kind  = ForwardKind::Prefill;
    SampleKind sample = SampleKind::None;
    std::vector<WorkRow> rows;
    std::optional<vision::Work> visual;
};

struct MtpWork {
    std::vector<WorkRow> rows;
    std::vector<std::uint32_t> draft_extents;
};

struct ResidencyBinding {
    runtime::ExecutionMember member;
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
    BatchPlan batch;
};

struct RowResult {
    std::uint32_t processed = 0;
    std::vector<TokenId> tokens;
    std::uint32_t dead_positions = 0, drafts = 0, accepted_drafts = 0;
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
    bool terminal = false, cancelled = false, failed = false;
};

struct Commit {
    std::vector<CommitRow> rows;
    ContextActions actions;
};

struct ExecutionTypes {
    using Work   = glm5next::Work;
    using Result = glm5next::Result;
    using Commit = glm5next::Commit;
    [[nodiscard]] static ExecutionWork describe_work(const Work&);
};

struct WorkContract {
    ExecutionShape shape;
    std::uint32_t vocabulary, public_tokens, ranks;
    std::uint32_t vision_max_features = 0;
    void validate(const Work&, runtime::FrameKind) const;
    void validate(const Work&, const Result&, const Commit&) const;
    void validate_commit(const Work&, std::span<const RowResult>, const Commit&) const;
    void validate(const Work&, const runtime::DecisionRequest&,
                  const runtime::DecisionInput&) const;
};

[[nodiscard]] std::span<const WorkRow> work_rows(const Work&);
[[nodiscard]] const context::ContextActions& local_actions(const ContextActions&,
                                                           std::uint32_t rank);

} // namespace ninfer::models::glm5next
