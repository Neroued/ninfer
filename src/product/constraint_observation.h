#pragma once

#include "ninfer/types.h"
#include <nlohmann/json.hpp>

namespace ninfer::product {

inline nlohmann::ordered_json
constraint_observation_json(const std::optional<ConstraintObservation>& observation) {
    if (!observation) return nullptr;
    const auto& value = *observation;
    nlohmann::ordered_json result{
        {"branch", value.branch == ConstraintOutputBranch::Content ? "content"
                   : value.branch == ConstraintOutputBranch::Tools ? "tools"
                                                                   : "undecided"},
        {"complete", value.complete},
        {"terminated", value.terminated},
        {"cache", value.cache == ConstraintCacheAccess::Hit     ? "hit"
                  : value.cache == ConstraintCacheAccess::Built ? "built"
                                                                : "waited"},
        {"mask_positions", value.mask_positions},
        {"mask_upload_bytes", value.mask_upload_bytes}};
    if (value.timings_collected)
        result["timings_seconds"] = {{"prepare", value.prepare_seconds},
                                     {"mask", value.mask_seconds},
                                     {"matcher", value.matcher_seconds}};
    return result;
}

} // namespace ninfer::product
