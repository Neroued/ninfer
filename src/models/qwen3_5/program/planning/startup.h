#pragma once

#include "models/common/qwen/vision/vision.h"
#include "models/qwen3_5/planning.h"

#include "core/cyclic_kv_cache.h"
#include "core/dtype.h"
#include "core/gdn_replay_records.h"
#include "core/layout.h"
#include "core/tensor.h"
#include "models/qwen3_5/state/decoder_state.h"
#include "models/qwen3_5/program/resources.h"
#include "models/qwen3_5/state/state_image.h"
#include "models/load_options.h"
#include "models/qwen3_5/program/planning/moe.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

namespace ninfer::models::qwen3_5::detail {

inline constexpr std::uint32_t kCausalScoreTile = 1024;

struct SequencePlanningInputs {
    std::shared_ptr<const execution::ModelProfile> profile;
    std::shared_ptr<const moe::Plan> moe;
    std::uint32_t capacity                 = 0;
    std::uint32_t max_concurrency          = 1;
    std::uint32_t prefill_chunk            = 0;
    std::uint32_t draft_window             = 0;
    SpeculativeBackend speculative_backend = SpeculativeBackend::None;
    KvCacheStorage kv_storage              = KvCacheStorage::BFloat16;
    ProposalHead proposal_head             = ProposalHead::Full;
    models::LoadOptions features;
    bool gpu_timing                   = false;
    bool use_cuda_graph               = true;
    bool causal_scoring               = false;
    std::int32_t multiprocessor_count = 0;
    ContextCacheOptions context_cache;
};

struct SequencePlanImpl {
    std::shared_ptr<const execution::ModelProfile> profile;
    std::shared_ptr<const moe::Plan> moe;
    std::uint32_t capacity                 = 0;
    std::size_t context_bytes              = 0;
    std::size_t context_metadata_bytes     = 0;
    std::uint32_t max_concurrency          = 1;
    std::uint32_t prefill_chunk            = 0;
    std::uint32_t draft_window             = 0;
    SpeculativeBackend speculative_backend = SpeculativeBackend::None;
    KvCacheStorage kv_storage              = KvCacheStorage::BFloat16;
    ProposalHead proposal_head             = ProposalHead::Full;
    models::LoadOptions features;
    bool gpu_timing                   = false;
    bool use_cuda_graph               = true;
    bool causal_scoring               = false;
    std::int32_t multiprocessor_count = 0;
    ContextCacheOptions context_cache;
    ResourceLayout persistent;
    WorkspaceSpec workspace;
    bool graph_probe                     = false;
    std::size_t device_reservation_bytes = 0;
};

struct SequencePlannerImpl {
    SequencePlanningInputs inputs;
    runtime::ContextCapacityPlan capacity_plan;
    std::unique_ptr<SequencePlanImpl> minimum;
};

[[nodiscard]] std::unique_ptr<qwen3_5::detail::SequencePlannerImpl>
make_sequence_planner_impl(std::shared_ptr<const execution::ModelProfile> profile,
                           DeviceContext& device, const EngineOptions& options,
                           std::shared_ptr<const moe::Plan> moe = {});
[[nodiscard]] std::unique_ptr<SequencePlanImpl>
finalize_sequence_plan_impl(std::unique_ptr<qwen3_5::detail::SequencePlannerImpl> planner,
                            runtime::ContextCapacity capacity);

[[nodiscard]] std::unique_ptr<SequencePlanImpl>
make_graph_probe_plan(std::shared_ptr<const execution::ModelProfile>, DeviceContext&,
                      const EngineOptions&, std::shared_ptr<const moe::Plan>);

} // namespace ninfer::models::qwen3_5::detail
