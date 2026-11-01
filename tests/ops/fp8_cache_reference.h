#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace ninfer::test {

// Independent host reference: use real-number exponent decomposition rather than
// the CUDA encoder's FP32 bit arithmetic. Each consecutive 128 features is a group.
inline std::uint8_t fp8_cache_scale(float absmax) {
    if (absmax == 0) return 127;
    int exponent;
    const double fraction = std::frexp(double(absmax) / 448.0, &exponent);
    return static_cast<std::uint8_t>(127 + std::max(-127, exponent - (fraction == 0.5)));
}

inline double fp8_cache_multiplier(std::uint8_t scale) { return std::ldexp(1.0, int(scale) - 127); }

} // namespace ninfer::test
