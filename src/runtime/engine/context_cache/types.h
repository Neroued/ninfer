#pragma once

#include <cstdint>

namespace ninfer::runtime {

using ContinuationOwnerToken = std::uint64_t;

struct CacheRetentionPriority {
    bool reused               = false;
    std::uint64_t last_demand = 0;
};

enum class ReclaimProgress : std::uint8_t { Blocked, Changed, Transferring };

enum class ReclaimPurpose : std::uint8_t { Execution, FreshAdmission, OptionalWrite };

struct ReclaimRights {
    ReclaimPurpose purpose = ReclaimPurpose::Execution;
    std::uint64_t request  = 0;
};

} // namespace ninfer::runtime
