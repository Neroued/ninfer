#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>

namespace ninfer::models::qwen3_5::detail {

// Scratch belongs to the admission boundary, never to a numerical KV operation.
// Logical descriptors identify a Host replica exactly once even when many active
// addresses or an inactive catalog entry alias it.
[[nodiscard]] inline bool mark_unique_host_replica(std::span<std::uint8_t> seen,
                                                  std::size_t descriptor,
                                                  std::size_t& unique_pages) noexcept {
    if (descriptor >= seen.size()) { return false; }
    if (seen[descriptor] == 0) {
        seen[descriptor] = 1;
        ++unique_pages;
    }
    return true;
}

// This is a planning claim, not physical occupancy. Inactive replicas remain
// charged as actual bytes. Existing replicas of claimed active requests are
// credited once before their complete future peaks are charged instead.
[[nodiscard]] inline std::optional<std::size_t> sparse_host_budget_occupancy(
    std::size_t actual_bytes, std::size_t claimed_active_replica_bytes,
    std::span<const std::size_t> active_full_peaks) noexcept {
    if (claimed_active_replica_bytes > actual_bytes) { return std::nullopt; }
    std::size_t claims = 0;
    for (const std::size_t peak : active_full_peaks) {
        if (peak > std::numeric_limits<std::size_t>::max() - claims) {
            return std::nullopt;
        }
        claims += peak;
    }
    if (claimed_active_replica_bytes > claims) { return std::nullopt; }
    const std::size_t inactive = actual_bytes - claimed_active_replica_bytes;
    if (claims > std::numeric_limits<std::size_t>::max() - inactive) {
        return std::nullopt;
    }
    return inactive + claims;
}

[[nodiscard]] inline bool sparse_host_budget_fits(std::size_t used,
                                                 std::size_t additional,
                                                 std::size_t capacity) noexcept {
    return additional <= capacity && used <= capacity - additional;
}

} // namespace ninfer::models::qwen3_5::detail
