#pragma once

#include "models/qwen4_exp/startup.h"

#include <algorithm>

namespace ninfer::test::flash_next {

// Exercise production weight placement with a bounded runtime allowance. For an automatic
// Context request the fixture needs only the model's declared startup working set.
inline models::qwen4_exp::Startup::ExecutionPlan plan_rank(models::qwen4_exp::Startup& startup,
                                                           DeviceContext& device,
                                                           const EngineOptions& options) {
    const auto memory = device_memory_snapshot(device.device());
    auto available    = memory.available_bytes();
    available -= std::min(available, options.memory_headroom_bytes);
    for (const auto& candidate : startup.placements()) {
        if (candidate.device_bytes > available ||
            candidate.host_bytes > memory.system_available_bytes)
            continue;
        auto remaining = available - candidate.device_bytes;
        if (memory.shared_physical_memory) {
            if (candidate.host_bytes > remaining) continue;
            remaining -= candidate.host_bytes;
        }
        remaining       = std::min<std::size_t>(remaining, 8ULL << 30);
        const auto host = memory.system_available_bytes - candidate.host_bytes;
        auto plan       = startup.plan(device, options, candidate.placement, remaining, host);
        if (!plan) continue;
        if (options.context_capacity.mode == ContextCapacityMode::Automatic) {
            auto minimum = options;
            minimum.context_capacity =
                ContextCapacityPolicy::explicit_capacity(plan->capacity.minimum_device_bytes);
            plan = startup.plan(device, minimum, candidate.placement, remaining, host);
        }
        if (plan) return std::move(*plan);
    }
    throw std::runtime_error("Flash Next test weights and Context working set do not fit");
}

} // namespace ninfer::test::flash_next
