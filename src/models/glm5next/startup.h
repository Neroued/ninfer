#pragma once

#include "models/glm5next/execution/parameters.h"
#include "models/glm5next/frontend/frontend.h"
#include "models/glm5next/load.h"
#include "models/glm5next/parallel/preparation.h"
#include "models/glm5next/program/resources.h"
#include "runtime/executor/preparation.h"
#include "runtime/executor/rank_worker.h"
#include "models/glm5next/runtime_types.h"
#include "models/registry.h"

namespace ninfer::models::glm5next {
class ExecutionCodec;

// Owns the same rank instance from weight materialization through execution activation.
// Reader is borrowed until materialize(); DeviceContext and communicators outlive the instance.
class RankPreparation {
public:
    using Types    = RuntimeTypes;
    using Instance = runtime::RankModelInstance<Types>;
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
    [[nodiscard]] ExecutionCodec codec(const distributed::ExecutionTopology&) const;
    [[nodiscard]] std::unique_ptr<Instance> finish() &&;

private:
    DeviceContext& device_;
    EngineOptions options_;
    ContextCommunication peer_;
    ParallelBinding binding_;
    runtime::PreparationMemory memory_;
    bool control_owner_;
    std::optional<ResolvedLoadPlan> load_;
    ExecutionPlan plan_;
    runtime::ContextCapacityPlan capacity_;
    std::unique_ptr<Instance> instance_;
};

class Startup {
public:
    using Types       = RuntimeTypes;
    using Preparation = RankPreparation;
    Startup(const artifact::Reader&, const artifact::Reader& rank, const EngineOptions&, Device);

    const auto& definition() const { return definition_; }

    Frontend frontend(const EngineOptions&) const;
    std::unique_ptr<Preparation> preparation(const artifact::Reader&, const EngineOptions&,
                                             DeviceContext&) const;
    PreparedSession session(const RankResourceDescription&,
                            std::uint32_t checkpoint_interval) const;

    Architecture architecture() const { return Architecture::Glm5Next; }

    std::string prefill_signature(std::uint32_t chunk) const;
private:
    std::shared_ptr<const ModelDefinition> definition_;
    RankPlan rank_;
};

[[nodiscard]] std::string prefill_signature(const ModelDefinition&, std::uint32_t chunk,
                                            std::uint32_t ranks);

[[nodiscard]] Frontend create_frontend(const ModelDefinition&, const EngineOptions&);

} // namespace ninfer::models::glm5next
