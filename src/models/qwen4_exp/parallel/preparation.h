#pragma once

#include "models/qwen4_exp/load.h"
#include "models/qwen4_exp/program/resources.h"

namespace ninfer::models::qwen4_exp {

struct RankResourceLayout {
    std::uint64_t state_bytes = 0, workspace_bytes = 0, device_bytes = 0, physical_bytes = 0;
    std::uint32_t state_slots = 0, main_pages = 0, mtp_pages = 0;
    std::uint32_t gdn_key_heads = 0, gdn_value_heads = 0, qsa_query_heads = 0, qsa_kv_heads = 0;
    std::uint32_t vocabulary_begin = 0, vocabulary = 0, ple_shards = 0;
};

// Owns immutable rank data once; mutable resources can be rebuilt at a new common capacity.
class RankPreparation {
public:
    RankPreparation(ResolvedLoadPlan, EngineOptions, DeviceContext&);

    [[nodiscard]] const runtime::SequenceCapacityCurve& capacity_curve() const { return curve_; }

    [[nodiscard]] const ModelShard& model() const { return *model_; }

    [[nodiscard]] const Resources& resources() const { return *resources_; }

    [[nodiscard]] RankResourceLayout layout() const;

    void prepare(std::uint32_t pages, std::size_t available_bytes);
    void release_resources();
private:
    EngineOptions options_;
    DeviceContext& device_;
    std::shared_ptr<const execution::ModelProfile> profile_;
    runtime::SequenceCapacityCurve curve_;
    std::unique_ptr<ModelShard> model_;
    std::unique_ptr<execution::Parameters> parameters_;
    std::unique_ptr<Resources> resources_;
};

} // namespace ninfer::models::qwen4_exp
