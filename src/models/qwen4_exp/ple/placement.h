#pragma once

#include "models/common/parallel.h"
#include "models/qwen4_exp/ple/parameters.h"
#include "ninfer/types.h"

namespace ninfer::models::qwen4_exp::ple {

// Consumers in one group read the same logical windows in the same order.
// Returned choices are ordered by latency preference for ties in capacity.
[[nodiscard]] std::vector<Placement> placements(PleOptions, std::uint32_t shards,
                                                std::uint32_t consumers);
[[nodiscard]] IndexRange shard_range(Placement, std::uint32_t shards, ParallelCoordinate consumer);
[[nodiscard]] const char* name(Placement);

} // namespace ninfer::models::qwen4_exp::ple
