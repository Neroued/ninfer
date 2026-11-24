#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer {

inline constexpr std::array<std::uint32_t, 16> kNf4CodebookWords{
    0xbf800000, 0xbf3239b1, 0xbf066b30, 0xbeca32a0, 0xbe91a24d, 0xbe3d353f, 0xbdba7871, 0x00000000,
    0x3da2faff, 0x3e24cae3, 0x3e7c04dd, 0x3ead033a, 0x3ee1a4b8, 0x3f1007ab, 0x3f3913b3, 0x3f800000};

inline constexpr float nf4_code_value(std::uint8_t code) {
    return std::bit_cast<float>(kNf4CodebookWords[code & 15]);
}

struct Nf4Bf16Lut {
    std::array<std::uint16_t, 128 * 16> words{};
};

// BF16 RNE of the exact codebook * E4M3FN(scale) / divisor, preserving signed zero.
// The divisor is positive finite and every valid entry must produce finite BF16.
// Scale words 0..126 are valid; row 127 is reserved and remains zero.
[[nodiscard]] Nf4Bf16Lut make_nf4_bf16_lut(float divisor);

// One packed row: K/2 low-even code bytes followed by K/16 scale bytes, producing
// K BF16 words. K is positive, divisible by 16, and the encoded row fits 4096 B.
// Input, LUT and output are disjoint Host ranges.
// Invalid geometry/scales are rejected before writing.
void decode_nf4_row(std::span<const std::byte> encoded, const Nf4Bf16Lut& lut,
                    std::span<std::uint16_t> output);

} // namespace ninfer
