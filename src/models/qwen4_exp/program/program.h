#pragma once

#include "models/qwen4_exp/program/planning.h"
#include "models/qwen4_exp/session/planning.h"
#include "models/qwen4_exp/work.h"
#include "models/common/parallel_execution.h"

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
                const StartupObserver& observer = {}, ParallelBinding = {});
    ~RankProgram();
    RankProgram(const RankProgram&)            = delete;
    RankProgram& operator=(const RankProgram&) = delete;

    [[nodiscard]] runtime::ExecutionCapacity capacity() const noexcept;
    [[nodiscard]] SessionPlan session_plan() const;
    void prepare_execution();
    void release_execution();
    void validate_work(const Work&, runtime::FrameKind) const;
    void start(runtime::ExecutionRequest<ExecutionTypes>&&);
    void validate_decision(runtime::ExecutionTicket, const runtime::DecisionInput&) const;
    void supply(runtime::ExecutionTicket, runtime::CommandSequence, runtime::DecisionInput&&);
    void prepare_commit(runtime::ExecutionTicket, Commit&) const;
    void commit(runtime::ExecutionTicket, runtime::CommandSequence, Commit&&);
    void discard(runtime::ExecutionTicket, runtime::CommandSequence);
    [[nodiscard]] bool can_discard(runtime::ExecutionTicket) const;
    [[nodiscard]] std::optional<runtime::ExecutionEvent<ExecutionTypes>> poll();
    void drain();
    [[nodiscard]] RankMemoryUsage observe_resources() const;
    [[nodiscard]] std::optional<GraphMemorySummary> graph_memory_observation() const;
    [[nodiscard]] RankMemoryBudget resource_budget() const;
    void reset_measurement_peaks() noexcept;
    [[nodiscard]] MemorySpan input_storage();

private:
    std::unique_ptr<detail::RankProgramImpl> impl_;
};
} // namespace ninfer::models::qwen4_exp
