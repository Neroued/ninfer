#pragma once

#include <type_traits>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#if defined(__CUDACC__)
#    define NINFER_KERNEL_HD __host__ __device__
#else
#    define NINFER_KERNEL_HD
#endif

namespace ninfer::ops::common {

// Exact division for numerator and positive divisor in [0, 2^31). The bounded
// numerator keeps the high-product addition representable in 32 bits.
struct FastDivisorU31 {
    std::uint32_t multiplier, shift, divisor;

    static FastDivisorU31 make(std::uint32_t divisor) {
        if (divisor == 0 || divisor >= (std::uint32_t{1} << 31))
            throw std::invalid_argument("FastDivisorU31: divisor must be in [1,2^31)");
        std::uint32_t shift = 0;
        while ((std::uint32_t{1} << shift) < divisor) ++shift;
        const auto multiplier = static_cast<std::uint32_t>(
            ((std::uint64_t{1} << 32) * ((std::uint64_t{1} << shift) - divisor)) / divisor + 1);
        return {multiplier, shift, divisor};
    }

    NINFER_KERNEL_HD std::uint32_t divide(std::uint32_t numerator) const {
#if defined(__CUDA_ARCH__)
        return (__umulhi(numerator, multiplier) + numerator) >> shift;
#else
        return numerator / divisor;
#endif
    }
};

// Host FP32 SiLU without exp overflow for large negative finite inputs.
inline float silu_stable(float value) noexcept {
    const float exponential = std::exp(-std::abs(value));
    return value >= 0 ? value / (1.0F + exponential) : (value * exponential) / (1.0F + exponential);
}

inline constexpr float kLog2E = 1.4426950408889634F;

template <class T>
NINFER_KERNEL_HD constexpr T div_up(T x, T d) {
    static_assert(std::is_integral_v<T>, "div_up requires an integral type");
    return x / d + static_cast<T>(x % d != 0);
}

template <class T>
NINFER_KERNEL_HD constexpr T round_up(T x, T multiple) {
    static_assert(std::is_integral_v<T>, "round_up requires an integral type");
    return div_up(x, multiple) * multiple;
}

template <auto Alignment, class T>
NINFER_KERNEL_HD constexpr T align_up(T x) {
    static_assert(std::is_integral_v<T>, "align_up requires an integral type");
    static_assert(Alignment > 0 && (Alignment & (Alignment - 1)) == 0,
                  "align_up requires a power-of-two alignment");
    constexpr T a = static_cast<T>(Alignment);
    return (x + a - 1) & ~(a - 1);
}

template <int Bits>
NINFER_KERNEL_HD constexpr int sign_extend(int x) {
    static_assert(Bits > 0 && Bits < 32, "sign_extend supports bit widths in [1, 31]");
    constexpr int sign = 1 << (Bits - 1);
    return (x ^ sign) - sign;
}

} // namespace ninfer::ops::common

#undef NINFER_KERNEL_HD
