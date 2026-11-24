#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace ninfer::test::float_codec {
// Test-owned represented-value codecs, independent of device packing and arithmetic.
inline std::uint16_t f32_to_f16_bits(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));

    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    const std::uint32_t exp  = (bits >> 23) & 0xffu;
    std::uint32_t mantissa   = bits & 0x007fffffu;
    if (exp == 0xffu) {
        return static_cast<std::uint16_t>(sign | (mantissa == 0 ? 0x7c00u : 0x7e00u));
    }

    const int half_exp = static_cast<int>(exp) - 127 + 15;
    if (half_exp >= 31) { return static_cast<std::uint16_t>(sign | 0x7c00u); }
    if (half_exp <= 0) {
        if (half_exp < -10) { return static_cast<std::uint16_t>(sign); }
        mantissa |= 0x00800000u;
        const int shift             = 14 - half_exp;
        std::uint32_t half_mantissa = mantissa >> shift;
        const std::uint32_t halfway = 1u << (shift - 1);
        const std::uint32_t tail    = mantissa & ((1u << shift) - 1u);
        if (tail > halfway || (tail == halfway && (half_mantissa & 1u) != 0u)) { ++half_mantissa; }
        return static_cast<std::uint16_t>(sign | half_mantissa);
    }

    std::uint32_t half_mantissa = mantissa >> 13;
    const std::uint32_t tail    = mantissa & 0x1fffu;
    std::uint32_t rounded_exp   = static_cast<std::uint32_t>(half_exp);
    if (tail > 0x1000u || (tail == 0x1000u && (half_mantissa & 1u) != 0u)) {
        ++half_mantissa;
        if (half_mantissa == 0x400u) {
            half_mantissa = 0;
            ++rounded_exp;
            if (rounded_exp >= 31) { return static_cast<std::uint16_t>(sign | 0x7c00u); }
        }
    }
    return static_cast<std::uint16_t>(sign | (rounded_exp << 10) | half_mantissa);
}

inline float f16_bits_to_f32(std::uint16_t bits) {
    const bool negative = (bits & 0x8000u) != 0;
    const int exp       = (bits >> 10) & 0x1f;
    const int mantissa  = bits & 0x03ff;
    float magnitude     = 0.0f;
    if (exp == 0) {
        magnitude = std::ldexp(static_cast<float>(mantissa), -24);
    } else if (exp == 31) {
        magnitude = mantissa == 0 ? std::numeric_limits<float>::infinity()
                                  : std::numeric_limits<float>::quiet_NaN();
    } else {
        magnitude = std::ldexp(1.0f + static_cast<float>(mantissa) / 1024.0f, exp - 15);
    }
    return negative ? -magnitude : magnitude;
}

inline std::int32_t round_even_to_i32(float value) {
    const float lower_f  = std::floor(value);
    const float fraction = value - lower_f;
    std::int32_t lower   = static_cast<std::int32_t>(lower_f);
    if (fraction < 0.5f) return lower;
    if (fraction > 0.5f) return lower + 1;
    return (lower & 1) == 0 ? lower : lower + 1;
}

inline float decode_e4m3fn_positive(std::uint8_t code) {
    const int exponent = (code >> 3) & 0x0f;
    const int mantissa = code & 0x07;
    if (exponent == 0) return std::ldexp(static_cast<float>(mantissa), -9);
    return std::ldexp(1.0f + static_cast<float>(mantissa) / 8.0f, exponent - 7);
}

inline float decode_e4m3fn(std::uint8_t code) {
    const float magnitude = decode_e4m3fn_positive(code & 0x7fU);
    return (code & 0x80U) == 0 ? magnitude : -magnitude;
}

inline std::uint8_t encode_e4m3fn_rne_satfinite(float value) {
    static const auto codebook = [] {
        std::array<float, 127> values{};
        for (int code = 0; code < 127; ++code) values[code] = decode_e4m3fn_positive(code);
        return values;
    }();
    const bool negative   = std::signbit(value);
    const float magnitude = std::abs(value);
    int selected          = 126;
    if (magnitude < 448.0f) {
        const int upper =
            std::lower_bound(codebook.begin(), codebook.end(), magnitude) - codebook.begin();
        if (upper == 0)
            selected = 0;
        else {
            const int lower        = upper - 1;
            const float low_error  = magnitude - codebook[lower],
                        high_error = codebook[upper] - magnitude;
            selected               = low_error < high_error   ? lower
                                     : high_error < low_error ? upper
                                                              : (lower % 2 == 0 ? lower : upper);
        }
    }
    return static_cast<std::uint8_t>(selected | (negative ? 128 : 0));
}

inline float decode_e2m1(std::uint8_t code) {
    constexpr std::array<float, 8> magnitude{0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float value = magnitude.at(static_cast<std::size_t>(code & 0x07U));
    return (code & 0x08U) == 0 ? value : -value;
}

inline std::uint8_t encode_e2m1_rne_satfinite(float value) {
    const bool negative   = std::signbit(value);
    const float magnitude = std::abs(value);
    std::uint8_t selected = 7;
    if (magnitude < 6.0f) {
        for (std::uint8_t upper = 1; upper <= 7; ++upper) {
            const float upper_value = decode_e2m1(upper);
            if (upper_value < magnitude) continue;
            const std::uint8_t lower = static_cast<std::uint8_t>(upper - 1);
            const float lower_value  = decode_e2m1(lower);
            const float lower_error  = magnitude - lower_value;
            const float upper_error  = upper_value - magnitude;
            selected                 = lower_error < upper_error   ? lower
                                       : upper_error < lower_error ? upper
                                                                   : ((lower & 1U) == 0U ? lower : upper);
            break;
        }
    }
    return static_cast<std::uint8_t>(selected | (negative ? 0x08U : 0U));
}

} // namespace ninfer::test::float_codec
