#pragma once

#include "core/dtype.h"
#include "ninfer/types.h"
#include "ops/fp8_cache_reference.h"
#include "ops/float_codec.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace ninfer::test::qsa::prepare {

enum class Encoding { Bf16, Fp16, Int8, Fp8, Nvfp4 };

struct VectorEncoding {
    Encoding encoding;
    DType data_dtype, scale_dtype;
    int data_extent, scale_extent, group;
    bool rotate;
};

struct CacheEncoding {
    VectorEncoding key, value;
};

inline CacheEncoding cache_encoding(KvCacheStorage storage) {
    const VectorEncoding i8{Encoding::Int8, DType::I8, DType::FP16, 256, 4, 64, true};
    const VectorEncoding fp8{Encoding::Fp8, DType::FP8_E4M3FN, DType::U8, 256, 2, 128, true};
    const VectorEncoding nvfp4{Encoding::Nvfp4, DType::U8, DType::U8, 128, 16, 16, true};
    switch (storage) {
    case KvCacheStorage::BFloat16:
        return {{Encoding::Bf16, DType::BF16, DType::U8, 256, 0, 0, false},
                {Encoding::Fp16, DType::FP16, DType::U8, 256, 0, 0, false}};
    case KvCacheStorage::Int8Group64: {
        auto value   = i8;
        value.rotate = false;
        return {i8, value};
    }
    case KvCacheStorage::Fp8E4m3fnGroup128Ue8m0: {
        auto value   = fp8;
        value.rotate = false;
        return {fp8, value};
    }
    case KvCacheStorage::Nvfp4Group16:
        return {nvfp4, nvfp4};
    case KvCacheStorage::Fp8KeyNvfp4Value:
        return {fp8, nvfp4};
    }
    throw std::invalid_argument("test cache storage");
}

inline std::uint16_t read_u16(const std::vector<std::uint8_t>& bytes, std::size_t offset) {
    std::uint16_t word;
    std::memcpy(&word, bytes.data() + offset, sizeof(word));
    return word;
}

inline void write_u16(std::vector<std::uint8_t>& bytes, std::size_t offset, std::uint16_t word) {
    std::memcpy(bytes.data() + offset, &word, sizeof(word));
}

inline std::uint16_t fp16_encode(float x) {
    return std::bit_cast<std::uint16_t>(static_cast<_Float16>(x));
}

inline float fp16_decode(std::uint16_t x) { return float(std::bit_cast<_Float16>(x)); }

inline double positive_fp8(int code) {
    static const auto values = [] {
        std::array<double, 128> result;
        for (int i = 0; i < 127; ++i) {
            const int exponent = i >> 3, fraction = i & 7;
            result[i] = exponent == 0 ? std::ldexp(double(fraction), -9)
                                      : std::ldexp(1.0 + fraction / 8.0, exponent - 7);
        }
        result[127] = std::numeric_limits<double>::quiet_NaN();
        return result;
    }();
    return values[code];
}

inline double positive_fp4(int code) {
    constexpr double values[]{0, 0.5, 1, 1.5, 2, 3, 4, 6};
    return values[code & 7];
}

inline std::uint8_t encode_minifloat(float x, bool fp4) {
    const double magnitude = std::abs(double(x));
    int selected           = 0;
    if (!fp4 && magnitude < 448.0) {
        const int lower          = float_codec::e4m3fn_floor_code(float(magnitude));
        const double lower_error = magnitude - positive_fp8(lower);
        const double upper_error = positive_fp8(lower + 1) - magnitude;
        selected =
            lower + (upper_error < lower_error || (upper_error == lower_error && (lower & 1)));
    } else if (!fp4) {
        selected = std::isnan(magnitude) ? 0 : 126;
    } else {
        int low = 0, high = 7;
        while (low < high) {
            const int mid = (low + high) / 2;
            if (positive_fp4(mid) < magnitude)
                low = mid + 1;
            else
                high = mid;
        }
        selected = low;
        if (low > 0) {
            const double upper_error = std::abs(positive_fp4(low) - magnitude);
            const double lower_error = std::abs(positive_fp4(low - 1) - magnitude);
            if (lower_error < upper_error || (lower_error == upper_error && (low & 1))) --selected;
        }
    }
    return std::uint8_t(selected | (std::signbit(x) ? (fp4 ? 8 : 128) : 0));
}

inline double decode_code(std::uint8_t code, Encoding encoding) {
    if (encoding == Encoding::Int8) return std::bit_cast<std::int8_t>(code);
    if (encoding == Encoding::Nvfp4) return (code & 8 ? -1 : 1) * positive_fp4(code);
    return (code & 128 ? -1 : 1) * positive_fp8(code & 127);
}

// The scalar butterfly uses ordinary contiguous dimensions, independently of CUDA lanes.
// Double evaluates the ideal transform; float evaluates the existing persistent V codec.
template <class T>
void hadamard(std::array<T, 256>& values) {
    for (int width = 1; width < 256; width *= 2)
        for (int block = 0; block < 256; block += 2 * width)
            for (int j = 0; j < width; ++j) {
                const T lo = values[block + j], hi = values[block + j + width];
                values[block + j]         = lo + hi;
                values[block + j + width] = lo - hi;
            }
    for (auto& value : values) value *= T(1.0 / 16);
}

struct ReferenceScale {
    std::uint16_t bits;
    float value;
};

inline ReferenceScale encode_scale(float absmax, Encoding encoding) {
    if (encoding == Encoding::Fp8) {
        const auto bits = fp8_cache_scale(absmax);
        return {bits, float(fp8_cache_multiplier(bits))};
    }
    if (absmax == 0) return {0, 0};
    if (encoding == Encoding::Nvfp4) {
        const float bounded = std::clamp(absmax / 6.0F, 0x1p-9F, 448.0F);
        const auto bits     = encode_minifloat(bounded, false);
        return {bits, float(positive_fp8(bits))};
    }
    const float raw = absmax / 127.0F;
    const auto bits = fp16_encode(raw);
    return {bits, fp16_decode(bits)};
}

inline std::uint8_t encode_code(float x, float scale, Encoding encoding) {
    if (scale == 0) return 0;
    const float normalized = encoding == Encoding::Nvfp4 ? x / scale : x * (1.0F / scale);
    if (encoding != Encoding::Int8)
        return encode_minifloat(normalized, encoding == Encoding::Nvfp4);
    const float clamped   = std::clamp(normalized, -127.0F, 127.0F);
    const int lower       = int(std::floor(clamped));
    const float remainder = clamped - lower;
    const int rounded     = lower + (remainder > 0.5F || (remainder == 0.5F && (lower & 1)));
    return std::bit_cast<std::uint8_t>(std::int8_t(rounded));
}

struct EncodedRow {
    std::vector<std::uint8_t> data, scales;
};

inline void encode_row_into(std::array<float, 256> values, VectorEncoding format,
                            std::span<std::uint8_t> data, std::span<std::uint8_t> scales) {
    if (format.rotate) hadamard(values);
    if (format.encoding == Encoding::Nvfp4) std::fill(data.begin(), data.end(), 0);
    for (int begin = 0; begin < 256; begin += format.group) {
        float absmax = 0;
        for (int d = begin; d < begin + format.group; ++d)
            absmax = std::max(absmax, std::abs(values[d]));
        const auto scale = encode_scale(absmax, format.encoding);
        const int group  = begin / format.group;
        if (format.scale_dtype == DType::U8)
            scales[group] = std::uint8_t(scale.bits);
        else
            std::memcpy(scales.data() + 2 * group, &scale.bits, sizeof(scale.bits));
        for (int d = begin; d < begin + format.group; ++d) {
            const auto code = encode_code(values[d], scale.value, format.encoding);
            if (format.encoding == Encoding::Nvfp4)
                data[d / 2] |= code << (4 * (d % 2));
            else
                data[d] = code;
        }
    }
}

inline EncodedRow encode_row(std::array<float, 256> values, VectorEncoding format) {
    EncodedRow out{std::vector<std::uint8_t>(format.data_extent * dtype_size(format.data_dtype)),
                   std::vector<std::uint8_t>(format.scale_extent * dtype_size(format.scale_dtype))};
    encode_row_into(values, format, out.data, out.scales);
    return out;
}

// Norm/RoPE and Hadamard are approximate FP32 computations. Their fixed input-error envelope
// admits adjacent quantizer bins at a threshold, not an extra BF16 rounding before encoding.
inline bool compatible_key_encoding(std::array<double, 256> ideal, VectorEncoding format,
                                    const EncodedRow& actual) {
    hadamard(ideal);
    double magnitude = 0;
    for (double x : ideal) magnitude = std::max(magnitude, std::abs(x));
    constexpr double kArithmeticRelative = 2e-5;
    const double delta =
        kArithmeticRelative * magnitude + 64 * double(std::numeric_limits<float>::denorm_min());
    double error2 = 0, nearest_error2 = 0, allowance2 = 0;
    for (int begin = 0; begin < 256; begin += format.group) {
        double absmax = 0;
        for (int d = begin; d < begin + format.group; ++d)
            absmax = std::max(absmax, std::abs(ideal[d]));
        const auto low    = encode_scale(float(std::max(0.0, absmax - delta)), format.encoding);
        const auto high   = encode_scale(float(absmax + delta), format.encoding);
        const int group   = begin / format.group;
        const float scale =
            format.encoding == Encoding::Nvfp4 ? float(positive_fp8(actual.scales[group]))
            : format.encoding == Encoding::Fp8 ? float(fp8_cache_multiplier(actual.scales[group]))
                                               : fp16_decode(read_u16(actual.scales, 2 * group));
        // The zero FP8 group uses unit scale; the first nonzero group uses 2^-127.
        if (!std::isfinite(scale) || scale < std::min(low.value, high.value) ||
            scale > std::max(low.value, high.value))
            return false;
        for (int d = begin; d < begin + format.group; ++d) {
            const auto code  = format.encoding == Encoding::Nvfp4
                                   ? (actual.data[d / 2] >> (4 * (d % 2))) & 15
                                   : actual.data[d];
            const double q   = decode_code(code, format.encoding);
            const double qlo = decode_code(
                encode_code(float(ideal[d] - delta), scale, format.encoding), format.encoding);
            const double qhi = decode_code(
                encode_code(float(ideal[d] + delta), scale, format.encoding), format.encoding);
            if (!std::isfinite(q) || q < qlo || q > qhi) return false;
            const double nearest =
                decode_code(encode_code(float(ideal[d]), scale, format.encoding), format.encoding) *
                scale;
            const double decoded = q * scale;
            error2 += (decoded - ideal[d]) * (decoded - ideal[d]);
            nearest_error2 += (nearest - ideal[d]) * (nearest - ideal[d]);
            // Crossing a nearest-rounding midpoint by delta changes squared error by at
            // most 2*delta*the distance between the two decoded codes.
            allowance2 += 2 * delta * std::abs(decoded - nearest) + delta * delta;
        }
    }
    return error2 <= nearest_error2 + allowance2 + 1e-12 * magnitude * magnitude;
}

} // namespace ninfer::test::qsa::prepare
