#pragma once

// Portable 128-bit unsigned multiply helpers. MSVC has no __int128; on x64 it
// provides _umul128 (full 128-bit product). Other platforms use __int128.

#include <cstdint>
#include <limits>

#if defined(_MSC_VER)
#include <intrin.h>
#pragma intrinsic(_umul128)
#endif

namespace ninfer::core {

// Full 128-bit product of two 64-bit values. high receives the upper half;
// returns the lower half.
[[nodiscard]] inline std::uint64_t u128_mul(std::uint64_t left, std::uint64_t right,
                                            std::uint64_t* high) noexcept {
#if defined(_MSC_VER)
    return _umul128(left, right, reinterpret_cast<unsigned __int64*>(high));
#else
    const unsigned __int128 product = static_cast<unsigned __int128>(left) * right;
    *high                          = static_cast<std::uint64_t>(product >> 64U);
    return static_cast<std::uint64_t>(product);
#endif
}

// Saturated product: returns max(uint64) on overflow.
[[nodiscard]] inline std::uint64_t saturating_u64_mul(std::uint64_t left,
                                                      std::uint64_t right) noexcept {
    std::uint64_t high = 0;
    const std::uint64_t low = u128_mul(left, right, &high);
    return high != 0 ? std::numeric_limits<std::uint64_t>::max() : low;
}

} // namespace ninfer::core
