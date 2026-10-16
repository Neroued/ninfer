#pragma once

#include "models/common/moe/schedule.h"
#include "models/qwen4_exp/execution/parameters.h"

namespace ninfer::models::qwen4_exp {

struct GraphPreparationPlan {
    bool enabled = false, offload = false;
    std::uint32_t max_context = 0, max_concurrency = 0, draft_tokens = 0, prefill_chunk = 0;
    KvCacheStorage kv_cache{};
    bool vision = false;
    std::vector<moe::GraphPlan> ordinary, verify;
    friend bool operator==(const GraphPreparationPlan&, const GraphPreparationPlan&) = default;
};

struct GraphCalibration {
    GraphPreparationPlan plan;
    std::size_t rebuild_bytes                = 0;
    std::int64_t preparation_increment_bytes = 0, retained_increment_bytes = 0;
};

[[nodiscard]] GraphPreparationPlan describe_graph_preparation(const EngineOptions&,
                                                              const moe::Plan*);
[[nodiscard]] GraphCalibration calibrate_graphs(const execution::Parameters&, DeviceContext&,
                                                const EngineOptions&,
                                                std::shared_ptr<const moe::Plan>);

} // namespace ninfer::models::qwen4_exp
