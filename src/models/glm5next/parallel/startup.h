#pragma once

#include "models/glm5next/frontend/frontend.h"
#include "models/glm5next/parallel/codec.h"
#include "models/glm5next/parallel/wire.h"
#include "models/glm5next/runtime_types.h"
#include "models/registry.h"

namespace ninfer::models::glm5next {

class ParallelStartup {
public:
    using Types = RuntimeTypes;
    using Codec = ExecutionCodec;
    ParallelStartup(const artifact::Reader&, const EngineOptions&);

    [[nodiscard]] const auto& definition() const { return definition_; }

    [[nodiscard]] const auto& topology() const { return parallel_.topology; }

    [[nodiscard]] Frontend frontend(const EngineOptions&) const;
    [[nodiscard]] std::vector<distributed::wire::Bytes> requests(const EngineOptions&) const;
    void resolve(std::span<const distributed::wire::Bytes>, EngineOptions&);
    [[nodiscard]] PreparedSession session() const;
    [[nodiscard]] Codec codec() const;
    [[nodiscard]] std::vector<RankSnapshot> observations() const;
    [[nodiscard]] std::vector<RankMemoryBudget> memory_budgets() const;

    [[nodiscard]] runtime::ExecutionCapacity capacity() const { return {1, 1}; }

    [[nodiscard]] std::string prefill_signature(std::uint32_t chunk) const;

    [[nodiscard]] Architecture architecture() const { return Architecture::Glm5Next; }

private:
    std::shared_ptr<const ModelDefinition> definition_;
    ParallelPlan parallel_;
    std::vector<RankResourceDescription> ranks_;
    std::uint32_t checkpoint_interval_ = 0;
};

} // namespace ninfer::models::glm5next
