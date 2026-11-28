#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::models::moe::exchange_detail {
inline int dimension(std::uint64_t n) {
    if (n > std::numeric_limits<int>::max())
        throw std::overflow_error("expert exchange extent exceeds int32");
    return static_cast<int>(n);
}

inline int hit_capacity(int b, int k, int slots) {
    return dimension(std::uint64_t(b) * std::min(k, slots));
}
} // namespace ninfer::models::moe::exchange_detail
