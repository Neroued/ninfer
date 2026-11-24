#pragma once

#include <array>
#include <bit>
#include <cstdint>

namespace ninfer::ops::common {

// Byte offset within the BlockScaleK16M128x4 scale plane.
#if defined(__CUDACC__)
__host__ __device__
#endif
    constexpr std::int64_t
    nvfp4_scale_byte_offset(int row, int group, int k) noexcept {
    return static_cast<std::int64_t>((row / 128) * (k / 64) + group / 4) * 512 + (row & 31) * 16 +
           ((row & 127) >> 5) * 4 + (group & 3);
}

// Exact E2M1 * nonnegative finite E4M3FN, before the matrix divisor.
struct Nvfp4Tables {
    alignas(64) std::array<std::array<float, 16>, 128> fp32{};
    alignas(64) std::array<std::array<std::uint16_t, 16>, 128> bf16{};

    constexpr Nvfp4Tables() {
        constexpr float codes[]{0, 0.5F, 1, 1.5F, 2, 3, 4, 6};
        for (unsigned s = 0; s < 127; ++s) {
            const float scale =
                s < 8 ? float(s) / 512
                      : std::bit_cast<float>(((s >> 3) + 120U) << 23 | ((s & 7U) << 20));
            for (unsigned c = 0; c < 16; ++c) {
                const float value = (c & 8 ? -codes[c & 7] : codes[c]) * scale;
                fp32[s][c]        = value;
                bf16[s][c]        = std::bit_cast<std::uint32_t>(value) >> 16;
            }
        }
    }
};

inline constexpr Nvfp4Tables nvfp4_tables;

} // namespace ninfer::ops::common
