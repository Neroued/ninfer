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
                    const distributed::RankIdentity&, DeviceContext&);

    [[nodiscard]] runtime::PreparationMemory memory() const;
    void materialize();
    [[nodiscard]] runtime::RankPreparationReport report(std::size_t available_bytes) const;

    void prepare(std::uint32_t pages, std::size_t available_bytes);
    void release_program();
    [[nodiscard]] RankExecutionDescription describe() const;
    [[nodiscard]] ExecutionCodec codec(const distributed::ExecutionTopology&) const;
    [[nodiscard]] std::unique_ptr<Instance> finish() &&;

private:
    EngineOptions options_;
    DeviceContext& device_;
    std::size_t host_reservation_bytes_ = 0;
    std::optional<ResolvedLoadPlan> load_;
    std::shared_ptr<const execution::ModelProfile> profile_;
    runtime::SequenceCapacityCurve curve_;
    std::unique_ptr<Instance> instance_;
};

} // namespace ninfer::models::qwen4_exp
