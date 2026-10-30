#pragma once

#include <cstdint>

namespace ninfer::runtime {

using ContinuationOwnerToken = std::uint64_t;

struct CacheRetentionPriority {
    std::uint64_t last_demand = 0;

    static constexpr std::uint64_t request_window = 128;

    [[nodiscard]] bool reused() const noexcept { return last_demand != 0; }

    [[nodiscard]] CacheRetentionPriority at(std::uint64_t request_epoch) const noexcept {
        return reused() && request_epoch - last_demand < request_window ? *this
                                                                        : CacheRetentionPriority{};
    }
};

enum class ReclaimProgress : std::uint8_t { Blocked, Changed, Transferring, Deferred, Retry };

enum class ReclaimPurpose : std::uint8_t { Execution, FreshAdmission, OptionalWrite };

struct ReclaimRights {
    ReclaimPurpose purpose = ReclaimPurpose::Execution;
    std::uint64_t request  = 0;
};

} // namespace ninfer::runtime
