#pragma once

#include <cstdint>

namespace ninfer::test::sampling_reference {
inline std::uint64_t mix(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

inline double uniform(std::uint64_t seed, int position, int purpose) {
    auto v = mix(seed ^ (std::uint64_t(std::uint32_t(position)) * 0xD1B54A32D192ED03ULL));
    v      = mix(v ^ (std::uint64_t(purpose) << 21));
    return double(v >> 40) / 16777216.;
}
} // namespace ninfer::test::sampling_reference
