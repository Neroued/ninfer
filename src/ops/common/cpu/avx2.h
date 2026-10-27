#pragma once

#include "ops/common/quant/nvfp4.h"

#include <immintrin.h>
#include <cstdint>
#include <cstring>

namespace ninfer::ops::common::cpu::avx2 {

// Private primitives for an isolated avx2/fma/f16c translation unit. Load/store
// pointers need only element alignment and must cover the complete vector extent.
inline __m256 load_bf16(const std::uint16_t* source) noexcept {
    const auto words = _mm_loadu_si128(reinterpret_cast<const __m128i*>(source));
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(words), 16));
}

// Half-precision bits to scalar float. MSVC's STL has no scalar _cvtsh_ss, so F16C
// conversion goes through the vector intrinsic there.
inline float half_to_float(std::uint16_t bits) noexcept {
#if defined(_MSC_VER)
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(bits)));
#else
    return _cvtsh_ss(bits);
#endif
}

// 16 packed bytes -> 32 unsigned bytes, low nibble then high nibble for each byte.
inline __m256i unpack_u4(const std::uint8_t* source) noexcept {
    const auto words =
        _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(source)));
    return _mm256_or_si256(
        _mm256_and_si256(words, _mm256_set1_epi16(15)),
        _mm256_and_si256(_mm256_slli_epi16(words, 4), _mm256_set1_epi16(0x0f00)));
}

inline float reduce_sum(__m256 values) noexcept {
    auto sum = _mm_add_ps(_mm256_castps256_ps128(values), _mm256_extractf128_ps(values, 1));
    sum      = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(_mm_hadd_ps(sum, sum));
}

inline __m256 decode_nvfp4(const std::uint8_t* codes, std::uint8_t scale) noexcept {
    std::uint32_t packed;
    std::memcpy(&packed, codes, sizeof(packed));
    const auto words = _mm_cvtepu8_epi16(_mm_cvtsi32_si128(packed));
    const auto bytes =
        _mm_or_si128(_mm_and_si128(words, _mm_set1_epi16(15)),
                     _mm_and_si128(_mm_slli_epi16(words, 4), _mm_set1_epi16(0x0f00)));
    const auto indices = _mm256_cvtepu8_epi32(bytes);
    const auto values =
        _mm256_permutevar8x32_ps(_mm256_load_ps(nvfp4_tables.fp32[scale].data()), indices);
    return _mm256_xor_ps(values, _mm256_castsi256_ps(_mm256_slli_epi32(
                                     _mm256_and_si256(indices, _mm256_set1_epi32(8)), 28)));
}

struct F32 {
    using Float                = __m256;
    static constexpr int lanes = 8;

    static Float zero() noexcept { return _mm256_setzero_ps(); }

    static Float load(const float* p) noexcept { return _mm256_loadu_ps(p); }

    static void store(float* p, Float v) noexcept { _mm256_storeu_ps(p, v); }

    static Float bf16(const std::uint16_t* p) noexcept { return load_bf16(p); }

    static Float fma(Float a, Float b, Float c) noexcept { return _mm256_fmadd_ps(a, b, c); }

    static float sum(Float v) noexcept { return reduce_sum(v); }
};

template <int Bits>
inline __m256 decode_signed_q(const std::uint8_t* code, const std::uint8_t* high) noexcept {
    static_assert(Bits >= 4 && Bits <= 6);
    std::uint32_t packed;
    std::memcpy(&packed, code, sizeof(packed));
    const auto words = _mm_cvtepu8_epi16(_mm_cvtsi32_si128(packed));
    const auto bytes =
        _mm_or_si128(_mm_and_si128(words, _mm_set1_epi16(15)),
                     _mm_and_si128(_mm_slli_epi16(words, 4), _mm_set1_epi16(0x0f00)));
    auto values = _mm256_cvtepu8_epi32(bytes);
    if constexpr (Bits == 6) {
        std::uint16_t packed_high;
        std::memcpy(&packed_high, high, sizeof(packed_high));
        auto upper = _mm256_srlv_epi32(_mm256_set1_epi32(packed_high),
                                       _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14));
        upper      = _mm256_srai_epi32(_mm256_slli_epi32(upper, 30), 26);
        values     = _mm256_add_epi32(values, upper);
    } else if constexpr (Bits == 5) {
        auto bits =
            _mm256_srlv_epi32(_mm256_set1_epi32(*high), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
        bits   = _mm256_slli_epi32(_mm256_and_si256(bits, _mm256_set1_epi32(1)), 4);
        values = _mm256_sub_epi32(values, bits);
    } else {
        values =
            _mm256_sub_epi32(_mm256_xor_si256(values, _mm256_set1_epi32(8)), _mm256_set1_epi32(8));
    }
    return _mm256_cvtepi32_ps(values);
}

} // namespace ninfer::ops::common::cpu::avx2
