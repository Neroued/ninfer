#pragma once

#include "runtime/engine/context_capacity.h"
#include "runtime/engine/context_cache/context_cost.h"
#include "runtime/executor/executor.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace ninfer::runtime {

struct StartupMemorySummary {
    std::vector<ContextCapacityResolution> context_capacities;
    std::size_t device_budget_bytes = 0;
    std::optional<GraphMemorySummary> cuda_graph;
    std::int64_t startup_retained_bytes = 0;
    std::vector<RankMemoryBudget> rank_budgets;
};

using ContextCostModel = ContextMachineCostModel;

template <class Types>
struct ReplicaInstance : StartupMemorySummary {
    using ModelContract = Types;
    std::uint32_t replica_id = 0;
    const std::shared_ptr<const typename Types::ModelDefinition> definition;
    typename Types::Frontend& frontend;
    std::unique_ptr<Executor<typename Types::ExecutionTypes>> executor;
    std::unique_ptr<typename Types::Session> session;
    std::optional<ContextCostModel> context_cost;

    ReplicaInstance(std::uint32_t replica_id,
                    std::shared_ptr<const typename Types::ModelDefinition> definition,
                    typename Types::Frontend& frontend,
                    std::unique_ptr<Executor<typename Types::ExecutionTypes>> executor,
                    typename Types::PreparedSession prepared_session,
                    StartupMemorySummary memory = {}, std::optional<ContextCostModel> cost = {})
        : StartupMemorySummary(std::move(memory)), replica_id(replica_id),
          definition(std::move(definition)), frontend(frontend), executor(std::move(executor)),
          session(std::make_unique<typename Types::Session>(std::move(prepared_session),
                                                            *this->executor)),
          context_cost(std::move(cost)) {}

    ~ReplicaInstance() {
        try {
            if (executor) {
                executor->bind_to_current_thread();
                executor->drain();
            }
        } catch (...) {}
        session.reset();
        executor.reset();
    }

    ReplicaInstance(const ReplicaInstance&)            = delete;
    ReplicaInstance& operator=(const ReplicaInstance&) = delete;
};

template <class Types>
struct BuiltReplica {
    std::uint32_t replica_id = 0;
    std::unique_ptr<Executor<typename Types::ExecutionTypes>> executor;
    typename Types::PreparedSession prepared_session;
    StartupMemorySummary memory;
    std::optional<ContextCostModel> context_cost{};
};

template <class Types>
struct InferenceInstance : StartupMemorySummary {
    using ModelContract = Types;
    using Replica       = ReplicaInstance<Types>;

    const std::shared_ptr<const typename Types::ModelDefinition> definition;
    typename Types::Frontend frontend;
    std::vector<std::unique_ptr<Replica>> replicas;
    // Keeps shared process supervisor / cluster resources alive across all replicas.
    std::shared_ptr<void> cluster_owner;

    // Single-replica convenience constructor (used by local single-GPU and dp=1 startups).
    InferenceInstance(std::shared_ptr<const typename Types::ModelDefinition> model_definition,
                      typename Types::Frontend model_frontend,
                      std::unique_ptr<Executor<typename Types::ExecutionTypes>> exec,
                      typename Types::PreparedSession prepared_session, StartupMemorySummary memory,
                      std::shared_ptr<void> shared_cluster_owner = {})
        : StartupMemorySummary(memory), definition(std::move(model_definition)),
          frontend(std::move(model_frontend)), cluster_owner(std::move(shared_cluster_owner)) {
        replicas.push_back(std::make_unique<Replica>(0, this->definition, this->frontend,
                                                     std::move(exec), std::move(prepared_session),
                                                     std::move(memory)));
    }

    // Multi-replica constructor (1..dp independent DP replicas sharing definition & frontend).
    InferenceInstance(std::shared_ptr<const typename Types::ModelDefinition> model_definition,
                      typename Types::Frontend model_frontend,
                      std::vector<BuiltReplica<Types>> built_replicas,
                      StartupMemorySummary cluster_memory,
                      std::shared_ptr<void> shared_cluster_owner = {})
        : StartupMemorySummary(std::move(cluster_memory)), definition(std::move(model_definition)),
          frontend(std::move(model_frontend)), cluster_owner(std::move(shared_cluster_owner)) {
        replicas.reserve(built_replicas.size());
        for (auto& built : built_replicas) {
            replicas.push_back(std::make_unique<Replica>(
                built.replica_id, this->definition, this->frontend, std::move(built.executor),
                std::move(built.prepared_session), std::move(built.memory),
                std::move(built.context_cost)));
        }
    }

    ~InferenceInstance() {
        replicas.clear();
        cluster_owner.reset();
    }

    InferenceInstance(const InferenceInstance&)            = delete;
    InferenceInstance& operator=(const InferenceInstance&) = delete;
};

} // namespace ninfer::runtime
