#pragma once

#include "runtime/engine/context_capacity.h"
#include "runtime/executor/executor.h"

#include <memory>
#include <utility>

namespace ninfer::runtime {

struct StartupMemorySummary {
    ContextCapacityResolution context_capacity;
    std::size_t device_budget_bytes = 0;
    std::optional<GraphMemorySummary> cuda_graph;
    std::int64_t startup_retained_bytes = 0;
    std::vector<RankMemoryBudget> rank_budgets;
};

template <class Types>
struct InferenceInstance : StartupMemorySummary {
    using ModelContract = Types;
    const std::shared_ptr<const typename Types::ModelDefinition> definition;
    typename Types::Frontend frontend;
    std::unique_ptr<Executor<typename Types::ExecutionTypes>> executor;
    std::unique_ptr<typename Types::Session> session;

    InferenceInstance(std::shared_ptr<const typename Types::ModelDefinition> definition,
                      typename Types::Frontend frontend,
                      std::unique_ptr<Executor<typename Types::ExecutionTypes>> executor,
                      typename Types::PreparedSession prepared_session, StartupMemorySummary memory)
        : StartupMemorySummary(std::move(memory)), definition(std::move(definition)),
          frontend(std::move(frontend)), executor(std::move(executor)),
          session(std::make_unique<typename Types::Session>(std::move(prepared_session),
                                                            *this->executor)) {}

    ~InferenceInstance() {
        try {
            executor->bind_to_current_thread();
            executor->drain();
        } catch (...) {}
        session.reset();
        executor.reset();
    }

    InferenceInstance(const InferenceInstance&)            = delete;
    InferenceInstance& operator=(const InferenceInstance&) = delete;
};

} // namespace ninfer::runtime
