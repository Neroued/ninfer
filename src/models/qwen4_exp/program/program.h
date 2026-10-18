#pragma once

#include "models/qwen4_exp/program/planning.h"
#include "models/qwen4_exp/work.h"

namespace ninfer::models::qwen4_exp {
namespace execution {
class Parameters;
}

namespace detail {
class RankProgramImpl;
}

// Owns one rank's physical execution resources. Work and commits carry already-resolved
// context identities; logical sequence, checkpoint and allocation ownership stays in Session.
class RankProgram {
public:
    RankProgram(const execution::Parameters&, ExecutionPlan, DeviceContext&,
                const StartupObserver& observer = {});
    ~RankProgram();
    RankProgram(const RankProgram&)            = delete;
    RankProgram& operator=(const RankProgram&) = delete;

    [[nodiscard]] runtime::ExecutionCapacity capacity() const noexcept;
    [[nodiscard]] std::vector<context::KVTransferProfile> context_transfer_profiles() const;
    void validate_work(const Work&, runtime::FrameKind) const;
    void start(runtime::ExecutionRequest<ExecutionTypes>&&);
    void validate_decision(runtime::ExecutionTicket, const runtime::DecisionInput&) const;
    void supply(runtime::ExecutionTicket, runtime::CommandSequence, runtime::DecisionInput&&);
    void prepare_commit(runtime::ExecutionTicket, Commit&) const;
    void commit(runtime::ExecutionTicket, runtime::CommandSequence, Commit&&);
    void discard(runtime::ExecutionTicket, runtime::CommandSequence);
    [[nodiscard]] std::optional<runtime::ExecutionEvent<ExecutionTypes>> poll();
    void drain();
    [[nodiscard]] MemorySummary memory_summary() const;
    void reset_memory_peaks() noexcept;

private:
    std::unique_ptr<detail::RankProgramImpl> impl_;
};
} // namespace ninfer::models::qwen4_exp
