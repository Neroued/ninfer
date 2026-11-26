#pragma once

#include "models/qwen3_5/parallel/codec.h"
#include "models/qwen3_5/parallel/wire.h"
#include "models/qwen3_5/runtime_types.h"

namespace ninfer::models::qwen3_5 {

class ParallelStartup {
public:
    using Types = RuntimeTypes;
    using Codec = ExecutionCodec;

    ParallelStartup(const artifact::Reader&, const EngineOptions&);

    [[nodiscard]] const auto& definition() const { return definition_; }
    [[nodiscard]] const auto& topology() const { return parallel_.topology; }
    [[nodiscard]] Architecture architecture() const noexcept { return architecture_; }

    [[nodiscard]] qwen::Frontend frontend(const EngineOptions&) const;
    [[nodiscard]] std::vector<distributed::wire::Bytes> requests(const EngineOptions&,
                                                                 bool narrow_targets = false) const;
    [[nodiscard]] bool narrow_targets(const EngineOptions&) const noexcept;
    [[nodiscard]] std::size_t coordinator_payload_bytes(const EngineOptions&) const;
    void resolve(std::span<const distributed::wire::Payload>, EngineOptions&);
    [[nodiscard]] std::vector<PreparedSession> sessions() const;
    [[nodiscard]] Codec codec(std::uint32_t replica_id) const;
    [[nodiscard]] runtime::ExecutionCapacity capacity(std::uint32_t replica_id = 0) const;
    [[nodiscard]] std::vector<RankSnapshot> observations() const;
    [[nodiscard]] std::vector<RankMemoryBudget> memory_budgets() const;
    [[nodiscard]] std::string prefill_signature(std::uint32_t chunk,
                                                std::uint32_t replica_id = 0) const;

private:
    std::shared_ptr<const ModelDefinition> definition_;
    std::shared_ptr<const execution::ModelProfile> profile_;
    Architecture architecture_;
    ParallelPlan parallel_;
    std::vector<RankExecutionDescription> ranks_;
};

} // namespace ninfer::models::qwen3_5
