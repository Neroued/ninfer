#pragma once

#include "runtime/engine/context_capacity.h"

namespace ninfer::models {

enum class WeightPlacement { Resident, HostExperts };

template <class Placement>
struct PlacementCandidate {
    Placement placement;
    std::size_t device_bytes;
    std::size_t host_bytes;
};

using WeightCandidate = PlacementCandidate<WeightPlacement>;

template <class SequencePlan, class Preparation, class Placement = WeightPlacement>
struct ExecutionPlan {
    SequencePlan sequence;
    runtime::ContextCapacityResolution capacity;
    Placement placement;
    std::uint32_t chunk;
    Preparation preparation;
};

} // namespace ninfer::models
