#pragma once

#include <cstdint>

namespace ninfer::ops::detail::gated_delta_net {

inline constexpr std::int32_t kStateDim = 128;

[[nodiscard]] constexpr bool are_head_counts_valid(std::int64_t qk_heads,
                                                   std::int64_t value_heads) noexcept {
    return qk_heads > 0 && value_heads >= qk_heads && (value_heads % qk_heads) == 0;
}

} // namespace ninfer::ops::detail::gated_delta_net
