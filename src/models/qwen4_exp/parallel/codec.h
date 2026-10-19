#pragma once

#include "models/qwen4_exp/parallel/wire.h"
#include "models/qwen4_exp/wire.h"
#include "runtime/executor/protocol.h"

namespace ninfer::models::qwen4_exp {

struct GroupExecutionPlan {
    SessionPlan session;
    runtime::ExecutionCapacity capacity;
};

[[nodiscard]] GroupExecutionPlan plan_group_execution(const ParallelPlan&,
                                                      std::span<const RankExecutionDescription>);

class ExecutionCodec {
public:
    ExecutionCodec(const ModelDefinition&, const RankExecutionDescription&,
                   const distributed::ExecutionTopology&);

    [[nodiscard]] runtime::protocol::ControlBudget control_budget() const noexcept {
        return budget_;
    }

    void validate_work(const Work&, runtime::FrameKind) const;
    void validate_decision(const Work&, const runtime::DecisionRequest&,
                           const runtime::DecisionInput&) const;
    void validate_commit(const Work&, const Result&, const Commit&) const;
    [[nodiscard]] runtime::protocol::Scope scope(const Work&,
                                                 const distributed::ExecutionTopology&) const;

    void encode_work(distributed::wire::Encoder&, const Work&, distributed::RankId) const;
    [[nodiscard]] Work decode_work(distributed::wire::Decoder&) const;
    void encode_commit(distributed::wire::Encoder&, const Commit&, distributed::RankId) const;
    [[nodiscard]] Commit decode_commit(distributed::wire::Decoder&) const;
    void encode_result(distributed::wire::Encoder&, const Result&) const;
    [[nodiscard]] Result decode_result(distributed::wire::Decoder&) const;

    [[nodiscard]] Result merge_context_results(const runtime::protocol::Scope&,
                                               const std::vector<std::optional<Result>>&) const;
    [[nodiscard]] MemorySummary
    merge_memory(const distributed::ExecutionTopology&,
                 const std::vector<std::optional<MemorySummary>>&) const;

private:
    WorkContract contract_;
    runtime::protocol::ControlBudget budget_;
    runtime::protocol::Scope scope_;
    VisionInputBindings inputs_;
};

} // namespace ninfer::models::qwen4_exp
