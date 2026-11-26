#pragma once

#include "models/common/graph_preparation.h"
#include "models/inkling2/execution/parameters.h"
#include "models/inkling2/program/resources.h"
#include "models/inkling2/work.h"

#include <memory>
#include <optional>

namespace ninfer::models::inkling2 {
namespace detail {
class RankProgramImpl;
}

class RankProgram {
public:
    RankProgram(const execution::Parameters&, std::unique_ptr<Resources>, DeviceContext&);
    ~RankProgram();
    RankProgram(const RankProgram&)            = delete;
    RankProgram& operator=(const RankProgram&) = delete;

    [[nodiscard]] static constexpr runtime::ExecutionCapacity capacity() noexcept { return {1, 1}; }

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
    GraphPreparationSample prepare_execution(GraphPreparationMode = GraphPreparationMode::Complete);
    void release_execution();
    [[nodiscard]] const ExecutionPlan& plan() const;
    [[nodiscard]] MemorySpan input_storage();
    [[nodiscard]] RankMemoryUsage observe_resources() const;
    [[nodiscard]] RankMemoryBudget resource_budget() const;
    void reset_measurement_peaks() noexcept;

private:
    std::unique_ptr<detail::RankProgramImpl> impl_;
};

} // namespace ninfer::models::inkling2
