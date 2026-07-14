#pragma once

#include <type_traits>

#if defined(__CUDACC__)
#    define NINFER_KERNEL_HD __host__ __device__
#else
#    define NINFER_KERNEL_HD
#endif

namespace ninfer::kernels {

template <class T>
NINFER_KERNEL_HD constexpr T div_up(T x, T d) {
    static_assert(std::is_integral_v<T>, "div_up requires an integral type");
    return (x + d - 1) / d;
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

} // namespace ninfer::kernels

#undef NINFER_KERNEL_HD
