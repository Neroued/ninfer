#pragma once

#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/load.h"
#include "models/qwen3_5/parallel/plan.h"
#include "models/qwen3_5/planning.h"
#include "models/qwen3_5/program/program.h"
#include "models/qwen3_5/runtime_types.h"
#include "ninfer/observability/resources.h"
#include "runtime/executor/preparation.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::distributed {
class CommunicationContext;
}

namespace ninfer::models::qwen3_5 {

class ExecutionCodec;

[[nodiscard]] runtime::WorkPayloadCapacity work_payload_capacity(const execution::ModelProfile&,
                                                                 const EngineOptions&);

struct RankLoadRequest {
    artifact::ArtifactId artifact{};
    RankPlan rank;
    EngineOptions options;
    std::size_t host_reservation_bytes = 0;
    bool control_owner                 = false;
    bool narrow_targets                = false;
};

struct RankExecutionDescription {
    std::uint32_t rank = 0;
    BatchExecutionSpec batches;
    std::vector<std::uint32_t> target_widths;
    ExecutionPath decode_path     = ExecutionPath::Eager;
    bool split_proposal_narrowing = false;
    context::ContextDirectorySpec context;
    std::vector<runtime::KvPoolDescriptor> kv_pools;
    std::uint32_t capacity                 = 0;
    std::uint32_t max_concurrency          = 1;
    std::uint32_t prefill_chunk            = 0;
    std::uint32_t draft_window             = 0;
    SpeculativeBackend speculative_backend = SpeculativeBackend::None;
    SpeculativeLengthPolicy length_policy  = SpeculativeLengthPolicy::Adaptive;
    std::uint32_t hybrid_max_tokens        = 0;
    ContextCacheOptions context_cache;
    KvCacheStorage kv_storage           = KvCacheStorage::BFloat16;
    LinearStateDType linear_state_dtype = LinearStateDType::Float32;
    ProposalHead proposal_head          = ProposalHead::Full;
    TransferWork state_transfer;
    std::optional<LayoutRegion> dflash_local_group;
    std::optional<TransferWork> dflash_local_transfer;
    std::uint32_t vision_max_features = 0;
    bool causal_scoring               = false;
    RankMemoryBudget budget;
    RankSnapshot observation;

    [[nodiscard]] SessionPlan to_session_plan(
        std::shared_ptr<const execution::ModelProfile> profile) const;
};

[[nodiscard]] std::vector<std::uint32_t> narrow_target_widths_for(const EngineOptions&);

class RankPreparation {
public:
    using Types = RuntimeTypes;

    struct Instance {
        std::unique_ptr<ModelShard> model;
        execution::Parameters parameters;
        std::unique_ptr<RankProgram> program;

        Instance(std::unique_ptr<ModelShard> shard,
                 std::shared_ptr<const execution::ModelProfile> profile)
            : model(std::move(shard)), parameters(*model, std::move(profile)) {}
    };

    RankPreparation(const artifact::Reader&, RankLoadRequest, DeviceContext&,
                    std::uint32_t global_rank = 0, std::uint32_t global_node = 0);
    RankPreparation(const artifact::Reader&, std::span<const std::byte>,
                    const distributed::RankIdentity&, DeviceContext&,
                    const distributed::ExecutionTopology&, distributed::CommunicationContext&,
                    const StartupObserver&);
    ~RankPreparation();

    RankPreparation(const RankPreparation&)            = delete;
    RankPreparation& operator=(const RankPreparation&) = delete;

    [[nodiscard]] runtime::PreparationMemory memory() const noexcept { return memory_; }
    [[nodiscard]] runtime::RankPreparationReport report(std::size_t available_bytes) const;
    void materialize();
    [[nodiscard]] std::vector<distributed::GroupRequirements> communication_requirements() const;
    void prepare_probe();
    void calibrate();
    void prepare_rebuild_probe();
    void calibrate_rebuild();
    void prepare_execution();
    void prepare(runtime::ContextCapacity, std::size_t available_bytes,
                 std::size_t available_host_bytes);
    void release_resources();
    void select_narrow_targets(bool enabled, GraphMemorySummary measured = {});

    [[nodiscard]] GraphMemorySummary graph_memory() const noexcept { return graph_memory_; }
    [[nodiscard]] RankExecutionDescription describe() const;
    [[nodiscard]] ExecutionCodec codec(const distributed::ExecutionTopology&) const;
    [[nodiscard]] std::unique_ptr<Instance> finish() &&;

private:
    void create_probe(GraphProbeBacking backing);

    DeviceContext& device_;
    EngineOptions options_;
    std::uint32_t global_rank_ = 0;
    std::uint32_t global_node_ = 0;
    bool control_owner_        = false;
    runtime::PreparationMemory memory_{};
    std::shared_ptr<const ModelDefinition> definition_;
    std::shared_ptr<const execution::ModelProfile> profile_;
    std::optional<ResolvedLoadPlan> load_;
    std::vector<std::uint32_t> target_widths_;
    runtime::ContextCapacityPlan capacity_;
    std::unique_ptr<Instance> instance_;
    std::unique_ptr<RankProgram> probe_;
    std::optional<SessionPlan> session_plan_;
    std::optional<runtime::ContextCapacityResolution> resolution_;
    GraphMemorySummary graph_memory_{};
};

} // namespace ninfer::models::qwen3_5
