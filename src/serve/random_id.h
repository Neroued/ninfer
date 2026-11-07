#pragma once

// Response and message IDs from the kernel's CSPRNG. resp_ IDs are the lookup key of
// GET /v1/responses/{id} and previous_response_id: a seeded mt19937_64 let a client that saw 312
// outputs (156 IDs) predict every later one (security review by MC Project Lab).

#include <cstdint>
#include <random>

#if defined(__linux__)
#    include <sys/random.h>
#endif

namespace ninfer::serve {

inline std::uint64_t random_id_bits() {
    std::uint64_t value = 0;
#if defined(__linux__)
    if (::getrandom(&value, sizeof(value), 0) == static_cast<ssize_t>(sizeof(value))) {
        return value;
    }
#endif
    std::random_device device; // also non-deterministic, on every supported platform
    return (static_cast<std::uint64_t>(device()) << 32U) ^ device();
}

} // namespace ninfer::serve
