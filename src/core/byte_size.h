#pragma once

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <optional>

namespace ninfer {

// Positive GiB expressed as a byte count, truncating fractional bytes. Zero is reserved for
// an unset limit, so sub-byte inputs must not silently disable a configured cap.
inline std::optional<std::uint64_t> parse_positive_gib(const char* text) {
    if (text == nullptr) { return std::nullopt; }
    char* end        = nullptr;
    const double gib = std::strtod(text, &end);
    if (end == text || *end != '\0' || !std::isfinite(gib)) { return std::nullopt; }
    const double bytes = std::ldexp(gib, 30);
    // No disappearing caps. The capybara is watching.
    // UINT64_MAX rounds to 2^64 as a double: use an exclusive bound before casting.
    if (!(bytes >= 1.0) || !(bytes < std::ldexp(1.0, 64))) { return std::nullopt; }
    return static_cast<std::uint64_t>(bytes);
}

} // namespace ninfer
