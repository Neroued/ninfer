#pragma once

#include "models/qwen4_exp/parallel/codec.h"
#include "models/qwen4_exp/runtime_types.h"
#include "models/qwen4_exp/startup.h"

namespace ninfer::models::qwen4_exp {

class ParallelStartup {
public:
    using Types = RuntimeTypes;
    using Codec = ExecutionCodec;
    ParallelStartup(const artifact::Reader&, const EngineOptions&);

    [[nodiscard]] const auto& definition() const { return definition_; }

    [[nodiscard]] const auto& topology() const { return parallel_.topology; }

    [[nodiscard]] qwen::Frontend frontend(const EngineOptions&) const;
    [[nodiscard]] std::vector<distributed::wire::Bytes> requests(const EngineOptions&) const;
    void resolve(std::span<const distributed::wire::Payload>, EngineOptions&);
    [[nodiscard]] PreparedSession session() const;
    [[nodiscard]] Codec codec() const;
    [[nodiscard]] std::vector<RankSnapshot> observations() const;
    [[nodiscard]] std::vector<RankMemoryBudget> memory_budgets() const;

    [[nodiscard]] runtime::ExecutionCapacity capacity() const { return plan_.capacity; }

    [[nodiscard]] std::string prefill_signature(std::uint32_t chunk) const;

    [[nodiscard]] Architecture architecture() const { return Architecture::Qwen4Exp; }

private:
    std::shared_ptr<const ModelDefinition> definition_;
    ParallelPlan parallel_;
    GroupExecutionPlan plan_;
    std::vector<RankExecutionDescription> ranks_;
};

} // namespace ninfer::models::qwen4_exp
