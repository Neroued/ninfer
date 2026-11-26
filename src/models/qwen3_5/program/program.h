#pragma once

#include "models/qwen3_5/planning.h"
#include "models/qwen3_5/work.h"

#include <memory>
#include <optional>

namespace ninfer::models::qwen3_5 {

namespace detail {
class RankProgramImpl;
}

class RankProgram {
public:
    ~RankProgram();
    RankProgram(const RankProgram&)            = delete;
    RankProgram& operator=(const RankProgram&) = delete;

    [[nodiscard]] static constexpr runtime::ExecutionCapacity capacity() noexcept { return {1, 1}; }
    void validate_work(const Work&, runtime::FrameKind) const;
    void validate_decision(runtime::ExecutionTicket, const runtime::DecisionInput&) const;
    void prepare_commit(runtime::ExecutionTicket, Commit&) const;

    void start(runtime::ExecutionRequest<ExecutionTypes>&&);
    void supply(runtime::ExecutionTicket, runtime::CommandSequence, runtime::DecisionInput&&);
    void commit(runtime::ExecutionTicket, runtime::CommandSequence, Commit&&);
    void discard(runtime::ExecutionTicket, runtime::CommandSequence);
    [[nodiscard]] bool can_discard(runtime::ExecutionTicket) const noexcept;
    [[nodiscard]] MemorySpan input_storage() noexcept;
    [[nodiscard]] std::optional<runtime::ExecutionEvent<ExecutionTypes>> poll();
    void drain();

    GraphPreparationSample prepare_execution(GraphPreparationMode = GraphPreparationMode::Complete);
    void release_execution();

    [[nodiscard]] RankMemoryUsage observe_resources() const;
    [[nodiscard]] std::optional<GraphMemorySummary> graph_memory_observation() const;
    void reset_measurement_peaks() noexcept;

private:
    explicit RankProgram(std::unique_ptr<detail::RankProgramImpl>);
    std::unique_ptr<detail::RankProgramImpl> impl_;

    friend std::unique_ptr<RankProgram> create_rank_program(const execution::Parameters&,
                                                            SequencePlan&&, DeviceContext&,
                                                            const StartupObserver&,
                                                            GraphPreparationMode);
    friend std::unique_ptr<RankProgram>
    create_uninitialized_rank_program(const execution::Parameters&, SequencePlan&&, DeviceContext&,
                                      const StartupObserver&);
};

} // namespace ninfer::models::qwen3_5
