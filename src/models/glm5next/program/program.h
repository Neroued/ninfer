#pragma once
#include "models/common/parallel_execution.h"
#include "models/glm5next/execution/parameters.h"
#include "models/glm5next/program/resources.h"
#include "models/glm5next/work.h"

namespace ninfer::models::glm5next {
namespace detail {
class RankProgramImpl;
}

class RankProgram {
public:
    RankProgram(const execution::Parameters&, std::unique_ptr<Resources>, DeviceContext&,
                ParallelBinding = {});
    ~RankProgram();
    RankProgram(const RankProgram&)            = delete;
    RankProgram& operator=(const RankProgram&) = delete;

    [[nodiscard]] runtime::ExecutionCapacity capacity() const noexcept { return {1, 1}; }

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
    void prepare_execution();
    void release_execution();
    [[nodiscard]] const ExecutionPlan& plan() const;
    [[nodiscard]] MemorySpan input_storage();
    [[nodiscard]] RankMemoryUsage observe_resources() const;
    [[nodiscard]] RankMemoryBudget resource_budget() const;
    void reset_measurement_peaks() noexcept;
private:
    std::unique_ptr<detail::RankProgramImpl> impl_;
};
} // namespace ninfer::models::glm5next
