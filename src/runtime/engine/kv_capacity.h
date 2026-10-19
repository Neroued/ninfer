#pragma once

#include "runtime/contract/resources.h"

#include <cstddef>
#include <span>
#include <vector>

namespace ninfer::runtime {

[[nodiscard]] KvCapacityResolution resolve_kv_capacity(const KvCapacityPolicy& policy,
                                                       const SequenceCapacityCurve& curve,
                                                       std::size_t available_runtime_bytes);

struct RankCapacityOffer {
    SequenceCapacityCurve curve;
    std::size_t available_bytes;
};

// One logical page count, with each owner's physical byte cost evaluated separately.
[[nodiscard]] std::vector<KvCapacityResolution>
resolve_group_kv_capacity(const KvCapacityPolicy&, std::span<const RankCapacityOffer>);

} // namespace ninfer::runtime
