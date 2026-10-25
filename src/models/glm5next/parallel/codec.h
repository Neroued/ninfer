#pragma once
#include "models/glm5next/parallel/wire.h"
#include "models/glm5next/wire.h"
#include "runtime/executor/protocol.h"

namespace ninfer::models::glm5next {

class ExecutionCodec {
public:
    ExecutionCodec(const ModelDefinition&, const RankResourceDescription&,
                   const distributed::ExecutionTopology&);

    runtime::protocol::ControlBudget control_budget() const noexcept { return budget_; }

    void validate_work(const Work&, runtime::FrameKind) const;
    void validate_decision(const Work&, const runtime::DecisionRequest&,
                           const runtime::DecisionInput&) const;
    void validate_commit(const Work&, const Result&, const Commit&) const;
    runtime::protocol::Scope scope(const Work&, const distributed::ExecutionTopology&) const;
    void encode_work(distributed::wire::Encoder&, const Work&, distributed::RankId) const;
    Work decode_work(distributed::wire::Decoder&) const;
    void encode_commit(distributed::wire::Encoder&, const Commit&, distributed::RankId) const;
    Commit decode_commit(distributed::wire::Decoder&) const;
    void encode_result(distributed::wire::Encoder&, const Result&) const;
    Result decode_result(distributed::wire::Decoder&) const;
    Result merge_context_results(const runtime::protocol::Scope&,
                                 const std::vector<std::optional<Result>>&) const;
private:
    WorkContract contract_;
    runtime::protocol::ControlBudget budget_;
    runtime::protocol::Scope scope_;
};

} // namespace ninfer::models::glm5next
