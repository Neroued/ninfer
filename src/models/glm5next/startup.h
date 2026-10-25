#pragma once

#include "models/common/startup.h"
#include "models/glm5next/execution/parameters.h"
#include "models/glm5next/frontend/frontend.h"
#include "models/glm5next/load.h"
#include "models/glm5next/parallel/preparation.h"
#include "models/glm5next/program/resources.h"

namespace ninfer::models::glm5next {

// Owns rank preparation through teardown. The Reader is borrowed only until materialize();
// DeviceContext and optional communication groups outlive this owner.
class RankPreparation {
public:
    RankPreparation(const artifact::Reader&, std::shared_ptr<const ModelDefinition>, RankPlan,
                    const EngineOptions&, DeviceContext&);
    ~RankPreparation();
    RankPreparation(const RankPreparation&)            = delete;
    RankPreparation& operator=(const RankPreparation&) = delete;

    [[nodiscard]] WeightCandidate weights() const noexcept { return weights_; }

    [[nodiscard]] const ExecutionPlan& plan() const noexcept { return plan_; }

    [[nodiscard]] runtime::RankCapacityOffer offer(std::size_t available,
                                                   std::size_t available_host) const;
    void materialize(const StartupObserver& = {});
    void allocate(runtime::ContextCapacityResolution, ContextCommunication = {});
    void release_resources();
    [[nodiscard]] const execution::Parameters& parameters() const;
    [[nodiscard]] Resources& resources();
    [[nodiscard]] RankResourceDescription describe() const;

private:
    DeviceContext& device_;
    WeightCandidate weights_;
    std::optional<ResolvedLoadPlan> load_;
    ExecutionPlan plan_;
    runtime::ContextCapacityPlan capacity_;
    std::unique_ptr<ModelShard> model_;
    std::unique_ptr<const execution::Parameters> parameters_;
    std::unique_ptr<Resources> resources_;
};

[[nodiscard]] Frontend create_frontend(const ModelDefinition&, const EngineOptions&);

} // namespace ninfer::models::glm5next
