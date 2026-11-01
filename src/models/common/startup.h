#pragma once

#include "runtime/engine/context_capacity.h"

namespace ninfer::models {

enum class WeightPlacement { Resident, HostExperts };

struct WeightCandidate {
    WeightPlacement placement;
    std::size_t device_bytes;
    std::size_t host_bytes;
};

template <class SequencePlan, class Preparation>
struct ExecutionPlan {
    SequencePlan sequence;
    runtime::ContextCapacityResolution capacity;
    WeightPlacement placement;
    std::uint32_t chunk;
    Preparation preparation;
};

} // namespace ninfer::models
