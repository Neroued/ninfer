#pragma once

#include "runtime/contract/resources.h"

#include <cstddef>
#include <span>
#include <vector>

namespace ninfer::runtime {

[[nodiscard]] ContextCapacityResolution
resolve_context_capacity(const KvCapacityPolicy& policy, const ContextCacheOptions& cache,
                         const ContextCapacityCurve& curve, std::size_t available_runtime_bytes);

struct RankCapacityOffer {
    ContextCapacityCurve curve;
    std::size_t available_bytes;
};

// One logical page/state capacity, with each owner's byte cost evaluated separately.
[[nodiscard]] std::vector<ContextCapacityResolution>
resolve_group_context_capacity(const KvCapacityPolicy&, const ContextCacheOptions&,
                               std::span<const RankCapacityOffer>);

} // namespace ninfer::runtime
