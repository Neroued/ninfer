#pragma once

#include "models/common/vision/input.h"
#include "models/qwen3_5/parallel/wire.h"
#include "models/qwen3_5/wire.h"
#include "runtime/executor/input.h"
#include "runtime/executor/protocol.h"

namespace ninfer::models::qwen3_5 {

class ExecutionCodec {
public:
    ExecutionCodec(const ModelDefinition&, const RankExecutionDescription&,
                   const distributed::ExecutionTopology&, const distributed::ReplicaTopology&);
    ExecutionCodec(const ModelDefinition&, const RankExecutionDescription&,
                   const distributed::ExecutionTopology&);

    [[nodiscard]] runtime::protocol::ControlBudget control_budget() const noexcept {
        return budget_;
    }

    void bind_input_storage(MemorySpan storage) { receiver_.bind(storage); }

    [[nodiscard]] std::vector<runtime::InputResource>
    prepare_inputs(const Work&, const distributed::ReplicaTopology&);
    [[nodiscard]] bool has_inputs(const Work&) const noexcept;
    void retire_inputs(const Work&);
    [[nodiscard]] std::optional<std::uint64_t> receive_input(distributed::wire::Kind,
                                                             std::span<const std::byte>);

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
                                               const distributed::ReplicaTopology&,
                                               const std::vector<std::optional<Result>>&) const;

private:
    VisionInputBindings inputs_;
    qwen::VisionInputReceiver receiver_;
    distributed::ReplicaTopology replica_;
    WorkContract contract_;
    runtime::protocol::ControlBudget budget_{};
    runtime::protocol::Scope scope_;
};

} // namespace ninfer::models::qwen3_5
