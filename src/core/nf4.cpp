#include "core/nf4.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer {
namespace {

struct BinaryValue {
    std::uint64_t significand;
    int exponent;
};

BinaryValue binary32(std::uint32_t magnitude) {
    const auto exponent = magnitude >> 23;
    return {exponent ? (magnitude & 0x7fffffU) | 0x800000U : magnitude,
            exponent ? int(exponent) - 150 : -149};
}

std::uint16_t rounded_ratio(std::uint64_t numerator, std::uint64_t denominator, int exponent) {
    int ratio_exponent = int(std::bit_width(numerator)) - int(std::bit_width(denominator));
    if (ratio_exponent >= 0 ? numerator < (denominator << ratio_exponent)
                            : (numerator << -ratio_exponent) < denominator)
        --ratio_exponent;
    const int power = ratio_exponent + exponent;
    if (power < -134) return 0;
    if (power > 127) throw std::invalid_argument("NF4 LUT overflows BF16");

    const int quantum = std::max(power - 7, -133);
    const int shift   = exponent - quantum;
    // The inputs have at most 28/24 significant bits. Scaling to one BF16 ulp
    // (or its subnormal quantum) keeps both integers within 33 bits here.
    if (shift >= 0)
        numerator <<= shift;
    else
        denominator <<= -shift;
    auto value           = numerator / denominator;
    const auto remainder = numerator % denominator;
    if (2 * remainder > denominator || (2 * remainder == denominator && (value & 1))) ++value;
    const auto word = power < -126 ? value : ((power + 127) << 7) + value - 128;
    if (word >= 0x7f80) throw std::invalid_argument("NF4 LUT overflows BF16");
    return static_cast<std::uint16_t>(word);
}

} // namespace

Nf4Bf16Lut make_nf4_bf16_lut(float divisor) {
    const auto bits = std::bit_cast<std::uint32_t>(divisor);
    if (!bits || bits >= 0x7f800000U)
        throw std::invalid_argument("NF4 requires a positive finite divisor");
    const auto d = binary32(bits);
    Nf4Bf16Lut result;
    for (unsigned s = 0; s < 127; ++s) {
        const auto exp = s >> 3;
        const BinaryValue scale{exp ? 8U + (s & 7) : s & 7, exp ? int(exp) - 10 : -9};
        for (unsigned c = 0; c < 16; ++c) {
            const auto word = kNf4CodebookWords[c];
            const auto code = binary32(word & 0x7fffffffU);
            const auto sign = static_cast<std::uint16_t>((word >> 16) & 0x8000);
            result.words[s * 16 + c] =
                sign | (code.significand && scale.significand
                            ? rounded_ratio(code.significand * scale.significand, d.significand,
                                            code.exponent + scale.exponent - d.exponent)
                            : 0);
        }
    }
    return result;
}

void decode_nf4_row(std::span<const std::byte> encoded, const Nf4Bf16Lut& lut,
                    std::span<std::uint16_t> output) {
    const auto k = output.size();
    if (!k || k % 16 || k / 2 + k / 16 > 4096 || encoded.size() != k / 2 + k / 16)
        throw std::invalid_argument("invalid NF4 row geometry");
    const auto* row = reinterpret_cast<const std::uint8_t*>(encoded.data());
    for (std::size_t group = 0; group < k / 16; ++group)
        if (row[k / 2 + group] >= 127) throw std::invalid_argument("invalid NF4 scale word");
    for (std::size_t group = 0; group < k / 16; ++group) {
        const auto* table = lut.words.data() + row[k / 2 + group] * 16;
        for (unsigned j = 0; j < 8; ++j) {
            const auto code                = row[group * 8 + j];
            output[group * 16 + 2 * j]     = table[code & 15];
            output[group * 16 + 2 * j + 1] = table[code >> 4];
        }
    }
}

} // namespace ninfer
