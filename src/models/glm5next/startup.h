#pragma once

#include "models/glm5next/execution/parameters.h"
#include "models/glm5next/frontend/frontend.h"
#include "models/glm5next/load.h"
#include "models/glm5next/parallel/preparation.h"
#include "models/glm5next/program/resources.h"
#include "runtime/executor/preparation.h"

namespace ninfer::models::glm5next {

// Parameters borrow the immutable model; Resources may contain asynchronous device work.
// Their addresses remain stable when preparation hands ownership to execution.
struct PreparedRank {
    PreparedRank()                               = default;
    PreparedRank(const PreparedRank&)            = delete;
    PreparedRank& operator=(const PreparedRank&) = delete;

    std::unique_ptr<ModelShard> model;
    std::unique_ptr<const execution::Parameters> parameters;
    std::unique_ptr<Resources> resources;
};

// The Reader is borrowed until materialize(). DeviceContext and communication groups outlive
// both preparation and the returned PreparedRank.
class RankPreparation {
public:
    RankPreparation(const artifact::Reader&, RankLoadRequest, DeviceContext&,
                    ContextCommunication = {});
    RankPreparation(const artifact::Reader&, std::span<const std::byte>,
                    const distributed::RankIdentity&, DeviceContext&,
                    const distributed::ExecutionTopology&, distributed::CommunicationContext&);
    ~RankPreparation();
    RankPreparation(const RankPreparation&)            = delete;
    RankPreparation& operator=(const RankPreparation&) = delete;

    [[nodiscard]] runtime::PreparationMemory memory() const;
    [[nodiscard]] runtime::RankPreparationReport report(std::size_t available_bytes) const;
    void materialize();
    void prepare(runtime::ContextCapacity, std::size_t available_bytes,
                 std::size_t available_host_bytes);
    void release_resources();
    [[nodiscard]] RankResourceDescription describe() const;
    [[nodiscard]] std::unique_ptr<PreparedRank> finish() &&;

private:
    DeviceContext& device_;
    EngineOptions options_;
    ContextCommunication peer_;
    runtime::PreparationMemory memory_;
    bool control_owner_;
    std::optional<ResolvedLoadPlan> load_;
    ExecutionPlan plan_;
    runtime::ContextCapacityPlan capacity_;
    std::unique_ptr<PreparedRank> prepared_;
};

[[nodiscard]] Frontend create_frontend(const ModelDefinition&, const EngineOptions&);

} // namespace ninfer::models::glm5next
