#include "runtime/engine/context_cache/capture_policy.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::runtime {

std::optional<std::uint32_t> intermediate_capture(std::uint32_t begin, std::uint32_t end,
                                                  std::uint32_t last_saved,
                                                  std::optional<std::uint32_t> next_capture,
                                                  std::span<const std::uint32_t> boundaries,
                                                  std::uint32_t interval) {
    if (!interval) throw std::invalid_argument("checkpoint interval must be positive");
    if (end <= begin || last_saved > begin) return {};
    const auto target = std::uint64_t(last_saved) + interval;
    if (end < target || (next_capture && *next_capture <= std::uint64_t(end) + interval / 4))
        return {};
    std::uint32_t chosen   = end;
    std::uint64_t distance = UINT64_MAX;
    for (auto it = std::upper_bound(boundaries.begin(), boundaries.end(), begin);
         it != boundaries.end() && *it <= end; ++it) {
        const auto difference = *it < target ? target - *it : *it - target;
        if (difference <= interval / 4 && difference < distance) {
            chosen   = *it;
            distance = difference;
        }
    }
    return chosen;
}

} // namespace ninfer::runtime
