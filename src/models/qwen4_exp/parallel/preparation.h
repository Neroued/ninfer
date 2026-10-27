#pragma once

#include "models/qwen4_exp/load.h"
#include "models/qwen4_exp/parallel/codec.h"
#include "models/qwen4_exp/runtime_types.h"
#include "runtime/executor/preparation.h"
#include "runtime/executor/rank_worker.h"

namespace ninfer::models::qwen4_exp {

// The source Reader is borrowed until materialize() returns. The same stable instance owns
// weights, Parameters and the final Program through preparation and execution.
class RankPreparation {
public:
    using Types    = RuntimeTypes;
    using Instance = runtime::RankModelInstance<Types>;

    RankPreparation(const artifact::Reader&, std::span<const std::byte>,
                    const distributed::RankIdentity&, DeviceContext&,
                    const distributed::ExecutionTopology&, distributed::CommunicationContext&);

    [[nodiscard]] runtime::PreparationMemory memory() const;
    void materialize();
    void prepare_communication();
    void prepare_probe();
    void calibrate();
    void prepare_rebuild_probe();
    void calibrate_rebuild();
    void prepare_execution();
    [[nodiscard]] runtime::RankPreparationReport report(std::size_t available_bytes) const;

    void prepare(runtime::ContextCapacity capacity, std::size_t available_bytes,
                 std::size_t available_host_bytes);
    void release_resources();
    [[nodiscard]] RankExecutionDescription describe() const;
    [[nodiscard]] ExecutionCodec codec(const distributed::ExecutionTopology&) const;
    [[nodiscard]] std::unique_ptr<Instance> finish() &&;

private:
    void create_probe(GraphProbeBacking);
    EngineOptions options_;
    DeviceContext& device_;
    ParallelBinding binding_;
    std::size_t host_reservation_bytes_ = 0;
    bool control_owner_                 = false;
    std::optional<ResolvedLoadPlan> load_;
    std::shared_ptr<const execution::ModelProfile> profile_;
    runtime::ContextCapacityPlan capacity_plan_;
    std::unique_ptr<Instance> instance_;
    std::unique_ptr<RankProgram> probe_;
    GraphMemorySummary graph_memory_;
};

} // namespace ninfer::models::qwen4_exp
