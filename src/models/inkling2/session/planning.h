#pragma once

#include "models/common/batch.h"
#include "models/common/context/directory.h"
#include "models/common/speculative/round_plan.h"
#include "models/inkling2/execution_shape.h"
#include "models/inkling2/state/context.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace ninfer::models::inkling2 {

struct SessionModelInfo {
    std::uint32_t public_tokens       = 0;
    std::uint32_t control_token_limit = 0;
    std::uint32_t experts             = 0;
    std::uint32_t experts_per_token   = 0;
    std::optional<VisionConfig> vision;
};

struct SessionPlan {
    SessionModelInfo model;
    ExecutionShape shape;
    std::shared_ptr<const BatchExecutionPlan> batches;
    context::ContextDirectorySpec context;
    std::vector<StatePlacement> states;
    std::vector<KVPageLayout> caches;
    std::uint32_t checkpoint_interval = 8192;
    std::uint32_t vision_max_features = 0;
    std::shared_ptr<const speculative::DecodeGraphRoutes> decode_routes;
};

[[nodiscard]] runtime::ContextMetadataPlan
session_metadata_plan(const context::ContextDirectorySpec&, const ContextPoolGeometry&,
                      std::uint32_t rank, std::uint32_t count_vocabulary, std::uint32_t concurrency,
                      std::uint32_t max_context, std::uint32_t prefill_chunk, bool vision);

} // namespace ninfer::models::inkling2
