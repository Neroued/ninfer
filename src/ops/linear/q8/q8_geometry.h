#pragma once

#include <cstdint>

namespace ninfer::ops::detail {

template <std::int32_t OutputRows, std::int32_t InputRows>
struct Q8LinearGeometry {
    static_assert(OutputRows > 0 && InputRows > 0);
    static_assert((OutputRows % 16) == 0);
    static_assert((InputRows % 32) == 0);

    static constexpr std::int32_t kOutputRows    = OutputRows;
    static constexpr std::int32_t kInputRows     = InputRows;
    static constexpr std::int32_t kGroupsPerRow  = InputRows / 32;
    static constexpr std::int32_t kCodeRowBytes  = InputRows;
    static constexpr std::int32_t kScaleRowBytes = kGroupsPerRow * sizeof(std::uint16_t);
};

} // namespace ninfer::ops::detail
