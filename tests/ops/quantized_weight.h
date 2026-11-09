#pragma once

// Test-owned packed-weight fixtures for every registered quantized Weight format. Physical
// row-split and block-scale codecs remain format-specific beneath one logical fixture interface.

#include "core/weight.h"
#include "core/tensor.h"
#include "core/weight_bank.h"
#include "float_codec.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace ninfer::test::quantized_weight {
namespace detail {

inline std::int32_t align_up(std::int32_t x, std::int32_t m) { return ((x + m - 1) / m) * m; }

inline std::size_t align_up_size(std::size_t x, std::size_t m) { return ((x + m - 1) / m) * m; }

inline std::uint64_t mix64(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

inline std::uint32_t float_bits(float f) {
    std::uint32_t u = 0;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}

inline float bits_float(std::uint32_t u) {
    float f = 0.0f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

inline std::uint32_t round_shift_rne(std::uint32_t v, int shift) {
    if (shift <= 0) { return v; }
    const std::uint32_t mask = (1u << shift) - 1u;
    const std::uint32_t half = 1u << (shift - 1);
    const std::uint32_t base = v >> shift;
    const std::uint32_t rem  = v & mask;
    return base + ((rem > half || (rem == half && (base & 1u))) ? 1u : 0u);
}

inline std::uint16_t f32_to_f16(float f) {
    const std::uint32_t x    = float_bits(f);
    const std::uint32_t sign = (x >> 16) & 0x8000u;
    const std::uint32_t ax   = x & 0x7fffffffu;

    if (ax >= 0x7f800000u) {
        const std::uint32_t mant = ax & 0x007fffffu;
        return static_cast<std::uint16_t>(sign | 0x7c00u | (mant ? 0x0200u : 0u));
    }

    int exp            = static_cast<int>((ax >> 23) & 0xffu) - 127 + 15;
    std::uint32_t mant = ax & 0x007fffffu;
    if (exp <= 0) {
        if (exp < -10) { return static_cast<std::uint16_t>(sign); }
        mant |= 0x00800000u;
        const std::uint32_t hm = round_shift_rne(mant, 14 - exp);
        return static_cast<std::uint16_t>(sign | hm);
    }
    if (exp >= 31) { return static_cast<std::uint16_t>(sign | 0x7c00u); }

    std::uint32_t hm = round_shift_rne(mant, 13);
    if (hm == 0x0400u) {
        hm = 0;
        ++exp;
        if (exp >= 31) { return static_cast<std::uint16_t>(sign | 0x7c00u); }
    }
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exp) << 10) | hm);
}

inline float f16_to_f32(std::uint16_t h) {
    const std::uint32_t sign = (static_cast<std::uint32_t>(h) & 0x8000u) << 16;
    std::uint32_t exp        = (static_cast<std::uint32_t>(h) >> 10) & 0x1fu;
    std::uint32_t mant       = static_cast<std::uint32_t>(h) & 0x03ffu;

    if (exp == 0) {
        if (mant == 0) { return bits_float(sign); }
        int e = -14;
        while ((mant & 0x0400u) == 0) {
            mant <<= 1;
            --e;
        }
        mant &= 0x03ffu;
        return bits_float(sign | (static_cast<std::uint32_t>(e + 127) << 23) | (mant << 13));
    }
    if (exp == 31) { return bits_float(sign | 0x7f800000u | (mant << 13)); }
    exp = exp - 15 + 127;
    return bits_float(sign | (exp << 23) | (mant << 13));
}

inline float bf16_to_f32(std::uint16_t h) {
    return bits_float(static_cast<std::uint32_t>(h) << 16);
}

inline void store_u16_le(std::vector<std::uint8_t>& payload, std::size_t off, std::uint16_t v) {
    payload[off]     = static_cast<std::uint8_t>(v & 0xffu);
    payload[off + 1] = static_cast<std::uint8_t>(v >> 8);
}

inline void store_u32_le(std::vector<std::uint8_t>& payload, std::size_t off, std::uint32_t v) {
    payload[off]     = static_cast<std::uint8_t>(v & 0xffu);
    payload[off + 1] = static_cast<std::uint8_t>((v >> 8) & 0xffu);
    payload[off + 2] = static_cast<std::uint8_t>((v >> 16) & 0xffu);
    payload[off + 3] = static_cast<std::uint8_t>(v >> 24);
}

inline std::uint16_t load_u16_le(const std::vector<std::uint8_t>& payload, std::size_t off) {
    return static_cast<std::uint16_t>(payload[off]) |
           static_cast<std::uint16_t>(static_cast<std::uint16_t>(payload[off + 1]) << 8);
}

inline std::uint32_t load_u32_le(const std::vector<std::uint8_t>& payload, std::size_t off) {
    return std::uint32_t(payload[off]) | (std::uint32_t(payload[off + 1]) << 8) |
           (std::uint32_t(payload[off + 2]) << 16) | (std::uint32_t(payload[off + 3]) << 24);
}

inline double decode_e2m1(std::uint8_t word) {
    constexpr double magnitudes[]{0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
    const double magnitude = magnitudes[word & 0x07U];
    return (word & 0x08U) == 0 ? magnitude : -magnitude;
}

inline double decode_e4m3fn(std::uint8_t word) {
    const bool negative   = (word & 0x80U) != 0;
    const std::uint32_t e = (word >> 3) & 0x0fU;
    const std::uint32_t m = word & 0x07U;
    double magnitude      = 0.0;
    if (e == 0) {
        magnitude = m == 0 ? 0.0 : static_cast<double>(m) * std::ldexp(1.0, -9);
    } else {
        if (e == 0x0fU && m == 0x07U) {
            throw std::invalid_argument("quantized-weight fixture: E4M3FN NaN scale");
        }
        magnitude = (1.0 + static_cast<double>(m) / 8.0) * std::ldexp(1.0, static_cast<int>(e) - 7);
    }
    return negative ? -magnitude : magnitude;
}

struct QuantSpec {
    int bits;
    int group_size;
    int qmax;
    int qmin;
};

inline QuantSpec quant_spec(QType qtype) {
    switch (qtype) {
    case QType::Q4_G64_FP16:
        return {4, 64, 7, -8};
    case QType::Q5_G64_FP16:
        return {5, 64, 15, -16};
    case QType::Q6_G64_FP16:
        return {6, 64, 31, -32};
    case QType::Q8_G32_FP16:
        return {8, 32, 127, -127};
    default:
        throw std::invalid_argument("row-split test packer: unsupported qtype");
    }
}

inline int nibble_bytes_per_group(const QuantSpec& spec) {
    return spec.bits == 8 ? spec.group_size : spec.group_size / 2;
}

inline int high_bytes_per_group(const QuantSpec& spec) {
    if (spec.bits == 8) { return 0; }
    return spec.bits <= 4 ? 0 : spec.group_size * (spec.bits - 4) / 8;
}

inline void pack_lowbit_group(const std::int8_t* codes, const QuantSpec& spec,
                              std::uint8_t* nibble_out, std::uint8_t* high_out) {
    const int nib  = nibble_bytes_per_group(spec);
    const int high = high_bytes_per_group(spec);
    std::fill(nibble_out, nibble_out + nib, static_cast<std::uint8_t>(0));
    if (high != 0) { std::fill(high_out, high_out + high, static_cast<std::uint8_t>(0)); }
    if (spec.bits == 8) {
        for (int i = 0; i < spec.group_size; ++i) {
            nibble_out[i] = static_cast<std::uint8_t>(codes[i]);
        }
        return;
    }
    const std::uint32_t mask = (1u << spec.bits) - 1u;
    for (int i = 0; i < spec.group_size; ++i) {
        const std::uint32_t u   = static_cast<std::uint32_t>(codes[i]) & mask;
        const std::uint32_t low = u & 0x0fu;
        if ((i & 1) == 0) {
            nibble_out[i >> 1] |= static_cast<std::uint8_t>(low);
        } else {
            nibble_out[i >> 1] |= static_cast<std::uint8_t>(low << 4);
        }
        if (high != 0) {
            const std::uint32_t hi = u >> 4;
            if (spec.bits == 5) {
                high_out[i >> 3] |= static_cast<std::uint8_t>((hi & 0x01u) << (i & 7));
            } else {
                const int bit_pos = i * 2;
                high_out[bit_pos >> 3] |= static_cast<std::uint8_t>((hi & 0x03u) << (bit_pos & 7));
            }
        }
    }
}

inline int unpack_lowbit_code(const std::uint8_t* nibble, const std::uint8_t* high,
                              const QuantSpec& spec, int index) {
    if (spec.bits == 8) { return static_cast<std::int8_t>(nibble[index]); }
    const std::uint8_t low_byte = nibble[index >> 1];
    const std::uint32_t low     = (index & 1) ? (low_byte >> 4) : (low_byte & 0x0fu);
    std::uint32_t hi            = 0;
    if (spec.bits == 5) {
        hi = (high[index >> 3] >> (index & 7)) & 0x01u;
    } else if (spec.bits == 6) {
        const int bitpos = index * 2;
        hi               = (high[bitpos >> 3] >> (bitpos & 7)) & 0x03u;
    }
    const std::uint32_t u    = low | (hi << 4);
    const std::uint32_t sign = 1u << (spec.bits - 1);
    const std::uint32_t span = 1u << spec.bits;
    return (u & sign) ? static_cast<int>(u) - static_cast<int>(span) : static_cast<int>(u);
}

} // namespace detail

struct PackedWeight {
    std::vector<std::uint8_t> payload;
    std::vector<float> dequant;
    Weight weight{};
    std::uint64_t code_plane_bytes    = 0;
    std::uint64_t high_plane_offset   = 0;
    std::uint64_t high_plane_bytes    = 0;
    std::uint64_t scale_plane_offset  = 0;
    std::uint64_t scale_plane_bytes   = 0;
    std::uint64_t matrix_scale_offset = 0;
    WeightGeometry geometry;

    [[nodiscard]] std::uint64_t model_weight_bytes() const {
        if (geometry.layout == QuantLayout::Csf32) {
            const auto& c = geometry.csf;
            return geometry.code_bytes + 4 + geometry.shape[0] + c.nibble_bytes_per_matrix +
                   c.offsets_per_matrix * 2 + c.pages_per_matrix * 4 + c.exception_count * 2;
        }
        return code_plane_bytes + high_plane_bytes + scale_plane_bytes +
               (matrix_scale_operation(weight.qtype) == MatrixScaleOperation::None ? 0 : 4);
    }

    Weight device_weight(ConstMemorySpan storage) const {
        const auto* device_payload = static_cast<const std::uint8_t*>(storage.data);
        if (storage.bytes < payload.size()) throw std::out_of_range("weight backing is too small");
        if (geometry.elements && device_payload) {
            const std::span<const float> scales = geometry.matrix_scale_count
                                                      ? std::span(&weight.weight_scale, 1)
                                                      : std::span<const float>{};
            const WeightParent parent{geometry, storage.device,
                                      static_cast<const std::byte*>(storage.data), scales};
            return native_weight({geometry.shape, {{&parent, 0, geometry.elements}}},
                                 weight.input_scale);
        }
        Weight w           = weight;
        w.device           = storage.device;
        w.payload          = device_payload;
        w.high_plane_bytes = high_plane_bytes;
        if (device_payload != nullptr) {
            w.qdata  = device_payload;
            w.qhigh  = high_plane_bytes == 0
                           ? nullptr
                           : static_cast<const std::uint8_t*>(device_payload) + high_plane_offset;
            w.scales = w.qtype == QType::BF16
                           ? nullptr
                           : static_cast<const std::uint8_t*>(device_payload) + scale_plane_offset;
        } else {
            w.qdata  = nullptr;
            w.qhigh  = nullptr;
            w.scales = nullptr;
        }
        return w;
    }

    Weight device_row_view(ConstMemorySpan storage, std::int32_t row_begin,
                           std::int32_t row_count) const {
        Weight w = device_weight(storage);
        if (row_begin < 0 || row_count <= 0 || row_begin + row_count > w.n) {
            throw std::invalid_argument("quantized-weight fixture: invalid row view");
        }
        if (geometry.elements) {
            const std::span<const float> scales = geometry.matrix_scale_count
                                                      ? std::span(&weight.weight_scale, 1)
                                                      : std::span<const float>{};
            const WeightParent parent{geometry, storage.device,
                                      static_cast<const std::byte*>(storage.data), scales};
            return native_weight({{std::uint64_t(row_count), std::uint64_t(w.k)},
                                  {{&parent, std::uint64_t(row_begin) * w.k,
                                    std::uint64_t(row_begin + row_count) * w.k}}},
                                 weight.input_scale);
        }
        if (w.layout == QuantLayout::RowSplit) {
            const detail::QuantSpec spec      = detail::quant_spec(w.qtype);
            const std::int32_t groups_per_row = w.padded_shape[1] / spec.group_size;
            const std::size_t code_row =
                static_cast<std::size_t>(groups_per_row) * detail::nibble_bytes_per_group(spec);
            const std::size_t high_row =
                static_cast<std::size_t>(groups_per_row) * detail::high_bytes_per_group(spec);
            const std::size_t scale_row =
                static_cast<std::size_t>(groups_per_row) * sizeof(std::uint16_t);
            w.qdata = static_cast<const std::uint8_t*>(w.qdata) +
                      static_cast<std::size_t>(row_begin) * code_row;
            w.qhigh  = high_row == 0 ? nullptr
                                     : static_cast<const std::uint8_t*>(w.qhigh) +
                                          static_cast<std::size_t>(row_begin) * high_row;
            w.scales = static_cast<const std::uint8_t*>(w.scales) +
                       static_cast<std::size_t>(row_begin) * scale_row;
        } else if (w.layout == QuantLayout::BlockScaleK16M128x4) {
            if ((row_begin % 128) != 0 || (row_count % 128) != 0) {
                throw std::invalid_argument(
                    "quantized-weight fixture: NVFP4 row view must align to 128 rows");
            }
            w.qdata = static_cast<const std::uint8_t*>(w.qdata) +
                      static_cast<std::size_t>(row_begin) * w.k / 2;
            w.scales = static_cast<const std::uint8_t*>(w.scales) +
                       static_cast<std::size_t>(row_begin / 128) *
                           static_cast<std::size_t>(w.k / 64) * 512;
        } else {
            throw std::invalid_argument("quantized-weight fixture: unsupported row-view layout");
        }
        w.n               = row_count;
        w.shape[0]        = row_count;
        w.padded_shape[0] = row_count;
        return w;
    }
};

enum class RowSplitScalePattern : std::uint8_t {
    Unit,
    Small,
    Tiny,
};

enum class RowSplitCodePattern : std::uint8_t {
    Coordinate,
    Hashed,
};

struct PatternedWeightOptions {
    RowSplitScalePattern row_split_scale = RowSplitScalePattern::Unit;
    RowSplitCodePattern row_split_codes  = RowSplitCodePattern::Coordinate;
    float weight_scale                   = 0.0F;
    float input_scale                    = 0.0F;
    std::optional<QuantLayout> layout;
    // Zero disables exceptions; a period of 389 approximates the checkpoint's sparse scales.
    std::uint32_t csf_exception_period     = 389;
    bool csf_dense_exception_block         = false;
    std::uint32_t outlier_expert_period    = 0;
    std::uint32_t outlier_exception_period = 12;
    float native_extent                    = 0.05F;
};

namespace detail {

inline std::uint64_t coordinate(std::uint32_t seed, int row, int column) {
    return mix64((std::uint64_t(std::uint32_t(row)) << 32) ^ std::uint32_t(column) ^
                 (std::uint64_t(seed) * 0x632be59bd9b4e019ULL));
}

inline std::uint8_t native_scale_base(std::uint32_t seed, int row) {
    return 28 + coordinate(seed ^ 0x251ca1U, row, 0) % 8;
}

inline bool native_exception(std::uint32_t seed, int row, int group,
                             const PatternedWeightOptions& o) {
    return (o.csf_dense_exception_block && row < 32 && group < 2) ||
           (o.csf_exception_period &&
            coordinate(seed ^ 0x5c5fU, row, group) % o.csf_exception_period == 0);
}

inline std::uint8_t native_scale_word(std::uint32_t seed, int row, int group,
                                      const PatternedWeightOptions& o) {
    if (native_exception(seed, row, group, o)) {
        constexpr std::uint8_t values[]{0, 4, 8, 80};
        return values[coordinate(seed ^ 0xe4ceU, row, group) & 3];
    }
    return native_scale_base(seed, row) + (coordinate(seed ^ 0xb10cU, row, group) & 15);
}

inline std::uint8_t native_fp8_code(std::uint32_t seed, int row, int column) {
    constexpr std::uint8_t codes[]{0x00, 0x80, 0x01, 0x81, 0x18, 0x98, 0x38, 0xb8,
                                   0x44, 0xc4, 0x60, 0xe0, 0x70, 0xf0, 0x7e, 0xfe};
    return codes[coordinate(seed, row, column) & 15];
}

inline std::uint64_t native_exception_count(int n, int k, std::uint32_t seed,
                                            const PatternedWeightOptions& o) {
    std::uint64_t count = 0;
    for (int r = 0; r < n; ++r)
        for (int g = 0; g < k / 16; ++g) count += native_exception(seed, r, g, o);
    return count;
}

template <class F>
void each_csf_block(int rows, int columns, F&& function) {
    std::uint64_t tile = 0, block = 0, nibble = 0;
    for (int rt = 0; rt < rows; rt += 128)
        for (int ct = 0; ct < columns; ct += 4, ++tile)
            for (int c = ct; c < std::min(ct + 4, columns); c += 2)
                for (int r = rt; r < std::min(rt + 128, rows); r += 32, ++block) {
                    const auto width = std::min(2, columns - c);
                    function(block, tile / 32, nibble, r, c, width);
                    nibble += 16 * width;
                }
}

inline PackedWeight make_native_patterned_weight(QType type, int n, int k, std::uint32_t seed,
                                                 const PatternedWeightOptions& o) {
    const auto layout = o.layout.value_or(type == QType::MXFP8 ? QuantLayout::RowGroupScale
                                          : type == QType::FP8_E4M3FN_BLOCK128X128_FP32
                                              ? QuantLayout::BlockScaleM128K128
                                              : QuantLayout::BlockScaleK16M128x4);
    const bool fp4    = type == QType::NVFP4_MUL;
    if (!std::isfinite(o.native_extent) || o.native_extent <= 0)
        throw std::invalid_argument("native weight fixture extent must be positive finite");
    if (!fp4 && (o.weight_scale != 0 || o.input_scale != 0))
        throw std::invalid_argument("native FP8 fixture has no matrix/activation scalar");
    const float multiplier =
        o.weight_scale ? o.weight_scale : bits_float(0x3c800003U + seed % 8191);
    if (fp4 && (!std::isfinite(multiplier) || multiplier <= 0 || !std::isfinite(o.input_scale) ||
                o.input_scale < 0))
        throw std::invalid_argument("native FP4 fixture has invalid multipliers");
    const std::array<std::uint64_t, 2> shape{std::uint64_t(n), std::uint64_t(k)};
    WeightLayoutParameters parameters;
    if (layout == QuantLayout::Csf32)
        parameters.exception_count = native_exception_count(n, k, seed, o);
    PackedWeight result;
    result.geometry = weight_geometry(type, layout, shape, parameters);
    const auto& g   = result.geometry;
    result.payload.resize(g.bytes);
    result.code_plane_bytes    = g.code_bytes;
    result.scale_plane_offset  = g.scale_offset;
    result.scale_plane_bytes   = g.scale_bytes;
    result.matrix_scale_offset = g.matrix_scale_offset;
    for (int r = 0; r < n; ++r)
        if (fp4) {
            for (int c = 0; c < k; c += 2)
                result.payload[std::uint64_t(r) * (k / 2) + c / 2] =
                    (coordinate(seed, r, c) & 15) | ((coordinate(seed, r, c + 1) & 15) << 4);
        } else {
            for (int c = 0; c < k; ++c)
                result.payload[std::uint64_t(r) * k + c] = native_fp8_code(seed, r, c);
        }
    if (type == QType::MXFP8) {
        const int exponent = std::ilogb(static_cast<double>(o.native_extent) / 224.0) + 127;
        for (int r = 0; r < n; ++r)
            for (int c = 0; c < k / 32; ++c)
                result.payload[g.scale_offset + std::uint64_t(r) * (k / 32) + c] =
                    std::clamp(exponent - 2 + int(coordinate(seed ^ 0x51ca1eU, r, c) % 5), 0, 254);
    } else if (!fp4) {
        for (int r = 0; r < n / 128; ++r)
            for (int c = 0; c < k / 128; ++c) {
                const float fraction =
                    float(coordinate(seed ^ 0x51ca1eU, r, c) & 0xffffffU) / 16777216.0F;
                const float scale = o.native_extent / 224.0F * (0.5F + fraction);
                store_u32_le(result.payload,
                             g.scale_offset + (std::uint64_t(r) * (k / 128) + c) * 4,
                             float_bits(scale));
            }
    } else if (layout == QuantLayout::Csf32) {
        const auto& csf = g.csf;
        for (int r = 0; r < n; ++r)
            result.payload[csf.bases_offset + r] = native_scale_base(seed, r);
        std::uint64_t exception = 0, page_start = 0, previous_page = UINT64_MAX;
        each_csf_block(n, k / 16, [&](auto block, auto page, auto nibble, int r, int c, int width) {
            if (page != previous_page) {
                page_start = exception;
                store_u32_le(result.payload, csf.page_bases_offset + page * 4, exception * 2);
                previous_page = page;
            }
            store_u16_le(result.payload, csf.offsets_offset + (block + page) * 2,
                         exception - page_start);
            for (int i = 0; i < 32 * width; ++i) {
                const int row = r + i / width, col = c + i % width;
                const auto value   = native_scale_word(seed, row, col, o);
                const bool outside = native_exception(seed, row, col, o);
                const auto delta   = outside ? 0 : value - native_scale_base(seed, row);
                result.payload[csf.nibbles_offset + nibble + i / 2] |= delta << (i % 2 * 4);
                if (outside) {
                    store_u16_le(result.payload, csf.exceptions_offset + exception * 2,
                                 (std::uint16_t(value) << 8) | ((row - r) * 2 + col - c));
                    ++exception;
                }
            }
            store_u16_le(result.payload, csf.offsets_offset + (block + page + 1) * 2,
                         exception - page_start);
        });
    } else {
        for (int r = 0; r < n; ++r)
            for (int c = 0; c < k / 16; ++c) {
                const auto offset = ((std::uint64_t(r / 128) * (k / 64) + c / 4) * 512) +
                                    (r % 32) * 16 + (r % 128 / 32) * 4 + c % 4;
                result.payload[g.scale_offset + offset] = native_scale_word(seed, r, c, o);
            }
    }
    result.weight.n = result.weight.shape[0] = n;
    result.weight.k = result.weight.shape[1] = k;
    result.weight.qtype                      = type;
    result.weight.weight_scale               = fp4 ? multiplier : 0;
    result.weight.input_scale                = o.input_scale;
    if (fp4) store_u32_le(result.payload, g.matrix_scale_offset, float_bits(multiplier));
    result.weight =
        result.device_weight({result.payload.data(), result.payload.size(), Device::cpu()});
    return result;
}
} // namespace detail

struct PackedWeightBank {
    WeightGeometry geometry;
    std::vector<std::uint8_t> payload;
    std::vector<float> matrix_scales;
    std::vector<float> activation_scales;
    std::vector<std::uint32_t> expert_indices;
    std::vector<std::uint64_t> matrix_bytes;

    [[nodiscard]] ExpertMatrixDeviceView device_view(ConstMemorySpan storage,
                                                     ConstMemorySpan indices = {}) const {
        if (storage.bytes < geometry.bytes || !storage.data)
            throw std::out_of_range("native bank fixture backing is incomplete");
        const auto count = static_cast<std::uint32_t>(geometry.shape[0]);
        const WeightParent parent{geometry, storage.device,
                                  static_cast<const std::byte*>(storage.data), matrix_scales};
        auto result   = device_weight_bank(bind_weight_bank(
            parent,
            describe_weight_bank(geometry, {0, count, geometry.shape[1], geometry.shape[1]})));
        bool identity = expert_indices.size() == count;
        for (std::uint32_t i = 0; identity && i < count; ++i) identity = expert_indices[i] == i;
        if (!identity && (!indices.data || indices.bytes < expert_indices.size() * 4))
            throw std::invalid_argument("native bank fixture requires its logical expert map");
        result.expert_indices = static_cast<const std::uint32_t*>(indices.data);
        return result;
    }

    [[nodiscard]] std::uint64_t model_weight_bytes() const {
        std::uint64_t bytes = 0;
        for (const auto value : matrix_bytes) bytes += value;
        return bytes;
    }

    [[nodiscard]] std::uint64_t expert_weight_bytes(std::uint32_t expert) const {
        const auto index = expert_indices.at(expert);
        if (index == UINT32_MAX) throw std::out_of_range("expert is absent from native fixture");
        return matrix_bytes.at(index);
    }

    void release_payload() { std::vector<std::uint8_t>().swap(payload); }
};

inline PackedWeightBank make_patterned_bank(QType type, int n, int k,
                                            std::span<const std::uint32_t> logical_ids,
                                            std::uint32_t logical_count, std::uint32_t seed,
                                            PatternedWeightOptions options = {}) {
    if (logical_ids.empty() || !logical_count || n <= 0 || k <= 0 ||
        (type != QType::NVFP4_MUL && type != QType::MXFP8 &&
         type != QType::FP8_E4M3FN_BLOCK128X128_FP32))
        throw std::invalid_argument("native bank fixture has invalid shape/format");
    PackedWeightBank result;
    result.expert_indices.assign(logical_count, UINT32_MAX);
    result.activation_scales.resize(logical_count);
    const auto local_options = [&](std::uint32_t expert) {
        auto o = options;
        if (o.layout == QuantLayout::BatchedBlockScaleK16M128x4)
            o.layout = QuantLayout::BlockScaleK16M128x4;
        if (type == QType::NVFP4_MUL) {
            o.weight_scale =
                options.weight_scale > 0
                    ? options.weight_scale
                    : detail::bits_float(0x3c800003U +
                                         (detail::coordinate(seed, expert, 1) & 0x003fffffU));
            o.input_scale = result.activation_scales[expert];
            if (o.outlier_expert_period && expert % o.outlier_expert_period == 0)
                o.csf_exception_period = o.outlier_exception_period;
        }
        return o;
    };
    for (std::uint32_t expert = 0; expert < logical_count; ++expert)
        result.activation_scales[expert] =
            type == QType::NVFP4_MUL
                ? (options.input_scale > 0
                       ? options.input_scale
                       : detail::bits_float(0x3b000005U +
                                            (detail::coordinate(seed, expert, 2) & 0x003fffffU)))
                : 0;
    auto layout = options.layout.value_or(type == QType::MXFP8 ? QuantLayout::RowGroupScale
                                          : type == QType::FP8_E4M3FN_BLOCK128X128_FP32
                                              ? QuantLayout::BlockScaleM128K128
                                              : QuantLayout::BlockScaleK16M128x4);
    std::uint64_t exception_count = 0;
    for (std::uint32_t index = 0; index < logical_ids.size(); ++index) {
        const auto expert = logical_ids[index];
        if (expert >= logical_count || result.expert_indices[expert] != UINT32_MAX)
            throw std::invalid_argument("native bank fixture repeats/exceeds logical expert IDs");
        result.expert_indices[expert] = index;
        if (layout == QuantLayout::Csf32)
            exception_count +=
                detail::native_exception_count(n, k, seed + expert * 65537U, local_options(expert));
    }
    if (layout == QuantLayout::BlockScaleK16M128x4)
        layout = QuantLayout::BatchedBlockScaleK16M128x4;
    const std::array<std::uint64_t, 3> shape{logical_ids.size(), std::uint64_t(n),
                                             std::uint64_t(k)};
    result.geometry =
        weight_geometry(type, layout, shape,
                        layout == QuantLayout::Csf32 ? WeightLayoutParameters{exception_count}
                                                     : WeightLayoutParameters{});
    const auto& g = result.geometry;
    result.payload.resize(g.bytes);
    if (g.matrix_scale_count) result.matrix_scales.resize(logical_ids.size());
    result.matrix_bytes.resize(logical_ids.size());
    std::uint64_t exceptions_written = 0;
    for (std::size_t slot = 0; slot < logical_ids.size(); ++slot) {
        const auto expert = logical_ids[slot];
        auto matrix       = detail::make_native_patterned_weight(type, n, k, seed + expert * 65537U,
                                                                 local_options(expert));
        const auto& m     = matrix.geometry;
        result.matrix_bytes[slot] = matrix.model_weight_bytes();
        std::memcpy(result.payload.data() + slot * m.code_bytes, matrix.payload.data(),
                    m.code_bytes);
        if (g.matrix_scale_count) {
            result.matrix_scales[slot] = matrix.weight.weight_scale;
            detail::store_u32_le(result.payload, g.matrix_scale_offset + slot * 4,
                                 detail::float_bits(matrix.weight.weight_scale));
        }
        if (layout == QuantLayout::Csf32) {
            const auto copy = [&](auto dst, auto src, auto bytes) {
                std::memcpy(result.payload.data() + dst + slot * bytes, matrix.payload.data() + src,
                            bytes);
            };
            copy(g.csf.bases_offset, m.csf.bases_offset, std::uint64_t(n));
            copy(g.csf.nibbles_offset, m.csf.nibbles_offset, m.csf.nibble_bytes_per_matrix);
            copy(g.csf.offsets_offset, m.csf.offsets_offset, m.csf.offsets_per_matrix * 2);
            for (std::uint64_t p = 0; p < m.csf.pages_per_matrix; ++p)
                detail::store_u32_le(
                    result.payload,
                    g.csf.page_bases_offset + (slot * m.csf.pages_per_matrix + p) * 4,
                    detail::load_u32_le(matrix.payload, m.csf.page_bases_offset + p * 4) +
                        exceptions_written * 2);
            std::memcpy(result.payload.data() + g.csf.exceptions_offset + exceptions_written * 2,
                        matrix.payload.data() + m.csf.exceptions_offset, m.csf.exception_count * 2);
            exceptions_written += m.csf.exception_count;
        } else {
            std::memcpy(result.payload.data() + g.scale_offset + slot * m.scale_bytes,
                        matrix.payload.data() + m.scale_offset, m.scale_bytes);
        }
    }
    return result;
}

namespace detail {
inline std::uint8_t csf_scale_word(const std::vector<std::uint8_t>& bytes, const WeightGeometry& g,
                                   std::uint64_t matrix, int row, int group) {
    const auto rows = g.shape[g.shape.size() - 2], columns = g.shape.back() / 16;
    const auto row_tile = row / 128, column_tile = group / 4;
    const auto row_blocks = std::min<std::uint64_t>(4, rows / 32 - row_tile * 4);
    const auto width      = std::min<std::uint64_t>(2, columns - (group / 2) * 2);
    const auto block      = row_tile * 4 * ((columns + 1) / 2) + column_tile * 2 * row_blocks +
                       (group % 4 / 2) * row_blocks + row % 128 / 32;
    const auto page  = (row_tile * ((columns + 3) / 4) + column_tile) / 32;
    const auto field = row_tile * 128 * columns + column_tile * row_blocks * 128 +
                       (group % 4 / 2) * row_blocks * 64 + (row % 128 / 32) * 32 * width +
                       (row % 32) * width + group % 2;
    const auto packed =
        bytes[g.csf.nibbles_offset + matrix * g.csf.nibble_bytes_per_matrix + field / 2];
    std::uint8_t value =
        bytes[g.csf.bases_offset + matrix * rows + row] + ((packed >> (field % 2 * 4)) & 15);
    const auto base =
        load_u32_le(bytes, g.csf.page_bases_offset + (matrix * g.csf.pages_per_matrix + page) * 4) /
        2;
    const auto offset =
        g.csf.offsets_offset + (matrix * g.csf.offsets_per_matrix + block + page) * 2;
    const auto begin = base + load_u16_le(bytes, offset),
               end   = base + load_u16_le(bytes, offset + 2);
    for (auto i = begin; i < end; ++i) {
        const auto record = load_u16_le(bytes, g.csf.exceptions_offset + i * 2);
        if ((record & 255) == (row % 32) * 2 + group % 2) return record >> 8;
    }
    return value;
}

inline double native_value(const std::vector<std::uint8_t>& bytes, const WeightGeometry& g,
                           std::uint64_t matrix, int row, int column) {
    const auto n = g.shape[g.shape.size() - 2], k = g.shape.back();
    const auto index = (matrix * n + row) * k + column;
    if (g.format == QType::BF16)
        return static_cast<double>(bf16_to_f32(load_u16_le(bytes, index * 2)));
    if (g.format == QType::Q4_G64_FP16) {
        const auto nibble = (bytes[index / 2] >> (column % 2 * 4)) & 15;
        const int code    = (static_cast<int>(nibble) ^ 8) - 8;
        const auto offset = g.scale_offset + ((matrix * n + row) * (k / 64) + column / 64) * 2;
        return static_cast<double>(code) * f16_to_f32(load_u16_le(bytes, offset));
    }
    if (g.format == QType::MXFP8)
        return decode_e4m3fn(bytes[index]) *
               float_codec::decode_ue8m0(
                   bytes[g.scale_offset + (matrix * n + row) * (k / 32) + column / 32]);
    if (g.format == QType::FP8_E4M3FN_BLOCK128X128_FP32) {
        const auto offset =
            g.scale_offset + ((matrix * (n / 128) + row / 128) * (k / 128) + column / 128) * 4;
        return decode_e4m3fn(bytes[index]) *
               static_cast<double>(bits_float(load_u32_le(bytes, offset)));
    }
    const auto packed     = bytes[index / 2];
    const auto code       = (packed >> (column % 2 * 4)) & 15;
    const int group       = column / 16;
    const auto scale      = g.layout == QuantLayout::Csf32
                                ? csf_scale_word(bytes, g, matrix, row, group)
                                : bytes[g.scale_offset + matrix * (n * k / 16) +
                                   ((row / 128) * (k / 64) + group / 4) * 512 + (row % 32) * 16 +
                                   (row % 128 / 32) * 4 + group % 4];
    const auto multiplier = bits_float(load_u32_le(bytes, g.matrix_scale_offset + matrix * 4));
    return decode_e2m1(code) * decode_e4m3fn(scale) * static_cast<double>(multiplier);
}
} // namespace detail

inline double logical_weight_fp64(const PackedWeightBank& bank, std::uint32_t expert,
                                  std::int32_t row, std::int32_t column) {
    const auto slot = bank.expert_indices.at(expert);
    if (slot == UINT32_MAX || row < 0 || column < 0 ||
        std::uint64_t(row) >= bank.geometry.shape[1] ||
        std::uint64_t(column) >= bank.geometry.shape[2] ||
        bank.payload.size() != bank.geometry.bytes)
        throw std::out_of_range("native bank oracle index or payload is invalid");
    return detail::native_value(bank.payload, bank.geometry, slot, row, column);
}

// Builds a deterministic full-shape payload without allocating a source or dequantized matrix.
// Large registered-shape conformance tests can then verify deterministic FP64 row samples.
inline PackedWeight make_patterned_weight(QType qtype, std::int32_t n, std::int32_t k,
                                          std::uint32_t seed, PatternedWeightOptions options = {}) {
    if (n <= 0 || k <= 0) {
        throw std::invalid_argument("quantized-weight fixture: shape must be positive");
    }
    if (qtype == QType::MXFP8 || qtype == QType::NVFP4_MUL ||
        qtype == QType::FP8_E4M3FN_BLOCK128X128_FP32)
        return detail::make_native_patterned_weight(qtype, n, k, seed, options);
    if (qtype == QType::FP8_E4M3FN_ROW_BF16) {
        if (options.weight_scale != 0.0F || options.input_scale != 0.0F) {
            throw std::invalid_argument(
                "quantized-weight fixture: divisors do not belong to FP8 weights");
        }

        PackedWeight packed;
        packed.code_plane_bytes = static_cast<std::uint64_t>(n) * k;
        packed.scale_plane_offset =
            detail::align_up_size(static_cast<std::size_t>(packed.code_plane_bytes), 256);
        packed.scale_plane_bytes = static_cast<std::uint64_t>(n) * 2;
        packed.payload.assign(
            static_cast<std::size_t>(packed.scale_plane_offset + packed.scale_plane_bytes), 0);

        constexpr std::uint8_t kCodes[]{
            0x00U, 0x80U, 0x01U, 0x81U, 0x07U, 0x87U, 0x08U, 0x88U,
            0x18U, 0x98U, 0x38U, 0xb8U, 0x44U, 0xc4U, 0x7eU, 0xfeU,
        };
        for (std::int32_t row = 0; row < n; ++row) {
            const std::uint32_t row_mix =
                static_cast<std::uint32_t>(row) ^ (static_cast<std::uint32_t>(row) >> 7);
            for (std::int32_t column = 0; column < k; ++column) {
                const std::uint32_t pattern =
                    row_mix * 13U + static_cast<std::uint32_t>(column) * 7U + seed;
                packed.payload[static_cast<std::size_t>(row) * k + column] = kCodes[pattern & 15U];
            }
        }
        constexpr std::uint16_t kScales[]{0x3b00U, 0x3b40U, 0x3b80U, 0x3bc0U};
        for (std::int32_t row = 0; row < n; ++row) {
            detail::store_u16_le(packed.payload,
                                 packed.scale_plane_offset + static_cast<std::size_t>(row) * 2,
                                 kScales[(static_cast<std::uint32_t>(row) + seed) & 3U]);
        }

        packed.weight.qtype            = QType::FP8_E4M3FN_ROW_BF16;
        packed.weight.layout           = QuantLayout::RowScale;
        packed.weight.scale_dtype      = DType::BF16;
        packed.weight.device           = Device::cpu();
        packed.weight.payload          = packed.payload.data();
        packed.weight.payload_bytes    = packed.payload.size();
        packed.weight.high_plane_bytes = 0;
        packed.weight.qdata            = packed.payload.data();
        packed.weight.qhigh            = nullptr;
        packed.weight.scales           = packed.payload.data() + packed.scale_plane_offset;
        packed.weight.group_size       = static_cast<std::uint32_t>(k);
        packed.weight.group            = k;
        packed.weight.ndim             = 2;
        packed.weight.shape[0]         = n;
        packed.weight.shape[1]         = k;
        packed.weight.shape[2]         = 1;
        packed.weight.shape[3]         = 1;
        packed.weight.padded_shape[0]  = n;
        packed.weight.padded_shape[1]  = k;
        packed.weight.padded_shape[2]  = 1;
        packed.weight.padded_shape[3]  = 1;
        packed.weight.scale_ne[0]      = n;
        packed.weight.scale_ne[1]      = 1;
        packed.weight.scale_ne[2]      = 1;
        packed.weight.scale_ne[3]      = 1;
        packed.weight.scale_nb[0]      = 2;
        packed.weight.scale_nb[1]      = static_cast<std::int64_t>(n) * 2;
        packed.weight.scale_nb[2]      = packed.weight.scale_nb[1];
        packed.weight.scale_nb[3]      = packed.weight.scale_nb[1];
        packed.weight.n                = n;
        packed.weight.k                = k;
        return packed;
    }
    if (qtype == QType::NVFP4) {
        if ((n % 128) != 0 || (k % 64) != 0) {
            throw std::invalid_argument(
                "quantized-weight fixture: NVFP4 N must be divisible by 128 and K by 64");
        }
        if (options.row_split_scale != RowSplitScalePattern::Unit ||
            options.row_split_codes != RowSplitCodePattern::Coordinate) {
            throw std::invalid_argument(
                "quantized-weight fixture: row-split patterns do not apply to NVFP4");
        }
        if (!std::isfinite(options.weight_scale) || options.weight_scale <= 0.0F ||
            !std::isfinite(options.input_scale) || options.input_scale <= 0.0F) {
            throw std::invalid_argument(
                "quantized-weight fixture: NVFP4 divisors must be finite and positive");
        }

        PackedWeight packed;
        packed.code_plane_bytes = static_cast<std::uint64_t>(n) * k / 2;
        packed.scale_plane_offset =
            detail::align_up_size(static_cast<std::size_t>(packed.code_plane_bytes), 256);
        packed.scale_plane_bytes   = static_cast<std::uint64_t>(n) * k / 16;
        packed.matrix_scale_offset = packed.scale_plane_offset + packed.scale_plane_bytes;
        packed.payload.assign(static_cast<std::size_t>(packed.matrix_scale_offset) + 4U, 0);

        for (std::int32_t row = 0; row < n; ++row) {
            for (std::int32_t column = 0; column < k; column += 2) {
                const auto code = [&](std::int32_t logical_column) {
                    return static_cast<std::uint8_t>(
                        (static_cast<std::uint32_t>(row) * 13U +
                         static_cast<std::uint32_t>(logical_column) * 7U + seed) &
                        0x0fU);
                };
                packed.payload[static_cast<std::size_t>(row) * k / 2 +
                               static_cast<std::size_t>(column / 2)] =
                    static_cast<std::uint8_t>(code(column) | (code(column + 1) << 4));
            }
        }

        constexpr std::uint8_t kScaleWords[]{
            0x00U, // +0
            0x04U, // 0.0078125
            0x08U, // 0.015625
            0x0bU, // 0.021484375
            0x10U, // 0.03125
            0x13U, // 0.04296875
            0x18U, // 0.0625
            0x1dU, // 0.1015625
        };
        const std::int32_t groups_per_row = k / 16;
        const std::int32_t k_tiles        = k / 64;
        for (std::int32_t row = 0; row < n; ++row) {
            const std::int32_t row_tile  = row / 128;
            const std::int32_t row_inner = row % 128;
            for (std::int32_t group = 0; group < groups_per_row; ++group) {
                const std::int32_t scale_tile = group / 4;
                const std::int32_t scale_lane = group % 4;
                const std::size_t stored_offset =
                    packed.scale_plane_offset +
                    static_cast<std::size_t>(row_tile * k_tiles + scale_tile) * 512U +
                    static_cast<std::size_t>(row_inner % 32) * 16U +
                    static_cast<std::size_t>(row_inner / 32) * 4U +
                    static_cast<std::size_t>(scale_lane);
                const std::size_t pattern = (static_cast<std::uint32_t>(row) * 5U +
                                             static_cast<std::uint32_t>(group) * 3U + seed) &
                                            7U;
                packed.payload[stored_offset] = kScaleWords[pattern];
            }
        }
        detail::store_u32_le(packed.payload, packed.matrix_scale_offset,
                             detail::float_bits(options.weight_scale));

        packed.weight.qtype            = QType::NVFP4;
        packed.weight.layout           = QuantLayout::BlockScaleK16M128x4;
        packed.weight.scale_dtype      = DType::FP8_E4M3FN;
        packed.weight.device           = Device::cpu();
        packed.weight.payload          = packed.payload.data();
        packed.weight.payload_bytes    = packed.payload.size();
        packed.weight.high_plane_bytes = 0;
        packed.weight.qdata            = packed.payload.data();
        packed.weight.qhigh            = nullptr;
        packed.weight.scales           = packed.payload.data() + packed.scale_plane_offset;
        packed.weight.group_size       = 16;
        packed.weight.group            = 16;
        packed.weight.ndim             = 2;
        packed.weight.shape[0]         = n;
        packed.weight.shape[1]         = k;
        packed.weight.shape[2]         = 1;
        packed.weight.shape[3]         = 1;
        packed.weight.padded_shape[0]  = n;
        packed.weight.padded_shape[1]  = k;
        packed.weight.padded_shape[2]  = 1;
        packed.weight.padded_shape[3]  = 1;
        packed.weight.n                = n;
        packed.weight.k                = k;
        packed.weight.weight_scale     = options.weight_scale;
        packed.weight.input_scale      = options.input_scale;
        return packed;
    }
    if (options.weight_scale != 0.0F || options.input_scale != 0.0F) {
        throw std::invalid_argument(
            "quantized-weight fixture: divisors are defined only for NVFP4");
    }
    const detail::QuantSpec spec      = detail::quant_spec(qtype);
    const std::int32_t padded_k       = detail::align_up(k, 128);
    const std::int32_t kg             = padded_k / spec.group_size;
    const std::int32_t logical_groups = (k + spec.group_size - 1) / spec.group_size;
    const std::uint64_t groups        = static_cast<std::uint64_t>(n) * kg;

    PackedWeight packed;
    packed.code_plane_bytes =
        groups * static_cast<std::uint64_t>(detail::nibble_bytes_per_group(spec));
    packed.high_plane_offset =
        detail::align_up_size(static_cast<std::size_t>(packed.code_plane_bytes), 256);
    packed.high_plane_bytes =
        groups * static_cast<std::uint64_t>(detail::high_bytes_per_group(spec));
    packed.scale_plane_offset =
        packed.high_plane_offset +
        detail::align_up_size(static_cast<std::size_t>(packed.high_plane_bytes), 256);
    packed.scale_plane_bytes = groups * 2;
    packed.payload.assign(
        static_cast<std::size_t>(packed.scale_plane_offset + packed.scale_plane_bytes), 0);

    const std::uint64_t code_bytes_per_group = detail::nibble_bytes_per_group(spec);
    const std::uint64_t high_bytes_per_group = detail::high_bytes_per_group(spec);
    if (options.row_split_codes == RowSplitCodePattern::Coordinate) {
        for (std::uint64_t group_index = 0; group_index < groups; ++group_index) {
            const std::uint64_t row   = group_index / static_cast<std::uint64_t>(kg);
            const std::uint64_t group = group_index % static_cast<std::uint64_t>(kg);
            if (group >= static_cast<std::uint64_t>(logical_groups)) { continue; }
            const std::uint32_t row_mix =
                static_cast<std::uint32_t>(row ^ (row >> 8) ^ (row >> 16));
            for (std::uint64_t byte = 0; byte < code_bytes_per_group; ++byte) {
                std::uint8_t code = static_cast<std::uint8_t>(
                    (row_mix * 37u + group * 29u + byte * 17u + seed) & 0xffu);
                if (qtype == QType::Q8_G32_FP16 && code == 0x80u) { code = 0x81u; }
                packed
                    .payload[static_cast<std::size_t>(group_index * code_bytes_per_group + byte)] =
                    code;
            }
            for (std::uint64_t byte = 0; byte < high_bytes_per_group; ++byte) {
                packed.payload[static_cast<std::size_t>(
                    packed.high_plane_offset + group_index * high_bytes_per_group + byte)] =
                    static_cast<std::uint8_t>(
                        (row_mix * 43u + group * 31u + byte * 13u + seed * 3u) & 0xffu);
            }
        }
    } else {
        std::uint8_t code_patterns[256][32]{};
        std::uint8_t high_patterns[256][16]{};
        for (std::size_t pattern = 0; pattern < 256; ++pattern) {
            std::uint64_t state = detail::mix64(static_cast<std::uint64_t>(seed) << 32 | pattern);
            for (std::size_t byte = 0; byte < code_bytes_per_group; ++byte) {
                state              = detail::mix64(state + byte);
                std::uint8_t value = static_cast<std::uint8_t>(state >> 56);
                if (qtype == QType::Q8_G32_FP16 && value == 0x80U) { value = 0x81U; }
                code_patterns[pattern][byte] = value;
            }
            for (std::size_t byte = 0; byte < high_bytes_per_group; ++byte) {
                state                        = detail::mix64(state + 0x100U + byte);
                high_patterns[pattern][byte] = static_cast<std::uint8_t>(state >> 56);
            }
        }
        for (std::uint64_t group_index = 0; group_index < groups; ++group_index) {
            const std::uint64_t group = group_index % static_cast<std::uint64_t>(kg);
            if (group >= static_cast<std::uint64_t>(logical_groups)) { continue; }
            const std::uint64_t mixed =
                detail::mix64(group_index ^ (static_cast<std::uint64_t>(seed) << 17));
            const std::size_t pattern = static_cast<std::size_t>(mixed & 0xffU);
            std::memcpy(packed.payload.data() + group_index * code_bytes_per_group,
                        code_patterns[pattern], code_bytes_per_group);
            if (high_bytes_per_group != 0) {
                std::memcpy(packed.payload.data() + packed.high_plane_offset +
                                group_index * high_bytes_per_group,
                            high_patterns[pattern], high_bytes_per_group);
            }
        }
    }
    const std::int32_t partial_group_lanes = k % spec.group_size;
    if (partial_group_lanes != 0) {
        std::int8_t codes[64]{};
        const std::int32_t group = logical_groups - 1;
        for (std::int32_t row = 0; row < n; ++row) {
            const std::size_t group_index = static_cast<std::size_t>(row) * kg + group;
            std::uint8_t* nibble = packed.payload.data() + group_index * code_bytes_per_group;
            std::uint8_t* high   = high_bytes_per_group == 0
                                       ? nullptr
                                       : packed.payload.data() + packed.high_plane_offset +
                                           group_index * high_bytes_per_group;
            for (std::int32_t lane = 0; lane < partial_group_lanes; ++lane) {
                codes[lane] =
                    static_cast<std::int8_t>(detail::unpack_lowbit_code(nibble, high, spec, lane));
            }
            std::fill(codes + partial_group_lanes, codes + spec.group_size,
                      static_cast<std::int8_t>(0));
            detail::pack_lowbit_group(codes, spec, nibble, high);
        }
    }
    constexpr std::uint16_t kUnitScales[]  = {0x3800u, 0x3a00u, 0x3c00u, 0x3d00u};
    constexpr std::uint16_t kSmallScales[] = {0x2040u, 0x2440u, 0x2840u, 0x2c40u};
    constexpr std::uint16_t kTinyScales[]  = {0x1840u, 0x1c40u, 0x2040u, 0x2440u};
    const std::uint16_t* scales            = nullptr;
    switch (options.row_split_scale) {
    case RowSplitScalePattern::Unit:
        scales = kUnitScales;
        break;
    case RowSplitScalePattern::Small:
        scales = kSmallScales;
        break;
    case RowSplitScalePattern::Tiny:
        scales = kTinyScales;
        break;
    }
    for (std::uint64_t i = 0; i < groups; ++i) {
        const std::uint64_t row   = i / static_cast<std::uint64_t>(kg);
        const std::uint64_t group = i % static_cast<std::uint64_t>(kg);
        if (group >= static_cast<std::uint64_t>(logical_groups)) { continue; }
        const std::uint64_t scale_index =
            options.row_split_codes == RowSplitCodePattern::Hashed
                ? (detail::mix64(i ^ (static_cast<std::uint64_t>(seed) << 17)) >> 8) & 3U
                : (row ^ (row >> 8) ^ group ^ seed) & 3U;
        detail::store_u16_le(packed.payload,
                             static_cast<std::size_t>(packed.scale_plane_offset + i * 2),
                             scales[scale_index]);
    }

    packed.weight.qtype            = qtype;
    packed.weight.layout           = QuantLayout::RowSplit;
    packed.weight.scale_dtype      = DType::FP16;
    packed.weight.device           = Device::cpu();
    packed.weight.payload          = packed.payload.data();
    packed.weight.payload_bytes    = packed.payload.size();
    packed.weight.high_plane_bytes = packed.high_plane_bytes;
    packed.weight.qdata            = packed.payload.data();
    packed.weight.qhigh =
        packed.high_plane_bytes == 0 ? nullptr : packed.payload.data() + packed.high_plane_offset;
    packed.weight.scales          = packed.payload.data() + packed.scale_plane_offset;
    packed.weight.group_size      = static_cast<std::uint32_t>(spec.group_size);
    packed.weight.group           = spec.group_size;
    packed.weight.ndim            = 2;
    packed.weight.shape[0]        = n;
    packed.weight.shape[1]        = k;
    packed.weight.shape[2]        = 1;
    packed.weight.shape[3]        = 1;
    packed.weight.padded_shape[0] = n;
    packed.weight.padded_shape[1] = padded_k;
    packed.weight.padded_shape[2] = 1;
    packed.weight.padded_shape[3] = 1;
    packed.weight.n               = n;
    packed.weight.k               = k;
    return packed;
}

// Independent logical decode for numerical oracles. The packed representation is materialized as
// its specified logical FP32 weight before the complete Op oracle receives it.
inline double logical_weight_fp64(const PackedWeight& packed, std::int32_t row,
                                  std::int32_t column) {
    const Weight& weight = packed.weight;
    if (row < 0 || row >= weight.shape[0] || column < 0 || column >= weight.shape[1]) {
        throw std::out_of_range("quantized-weight fixture: logical index out of range");
    }
    if (packed.geometry.elements)
        return detail::native_value(packed.payload, packed.geometry, 0, row, column);

    if (weight.qtype == QType::BF16) {
        if (weight.layout != QuantLayout::Contiguous || weight.padded_shape[0] != weight.n ||
            weight.padded_shape[1] != weight.k)
            throw std::invalid_argument("linear-weight fixture: invalid contiguous BF16 metadata");
        const auto offset = (static_cast<std::size_t>(row) * weight.k + column) * 2;
        return static_cast<double>(
            detail::bf16_to_f32(detail::load_u16_le(packed.payload, offset)));
    }

    if (weight.qtype == QType::FP8_E4M3FN_ROW_BF16) {
        if (weight.layout != QuantLayout::RowScale || weight.scale_dtype != DType::BF16 ||
            weight.group != weight.k || weight.group_size != static_cast<std::uint32_t>(weight.k)) {
            throw std::invalid_argument("quantized-weight fixture: invalid FP8 metadata");
        }
        const std::uint8_t code = packed.payload[static_cast<std::size_t>(row) * weight.k + column];
        const std::uint16_t scale_bits = detail::load_u16_le(
            packed.payload, packed.scale_plane_offset + static_cast<std::size_t>(row) * 2);
        const float decoded =
            static_cast<float>(detail::decode_e4m3fn(code)) * detail::bf16_to_f32(scale_bits);
        return static_cast<double>(decoded);
    }

    if (weight.qtype == QType::NVFP4) {
        if (weight.layout != QuantLayout::BlockScaleK16M128x4 ||
            weight.scale_dtype != DType::FP8_E4M3FN || weight.group != 16 ||
            !std::isfinite(weight.weight_scale) || weight.weight_scale <= 0.0F) {
            throw std::invalid_argument("quantized-weight fixture: invalid NVFP4 metadata");
        }
        const std::size_t code_offset =
            static_cast<std::size_t>(row) * weight.k / 2 + static_cast<std::size_t>(column / 2);
        const std::uint8_t packed_codes = packed.payload[code_offset];
        const std::uint8_t code = (column & 1) == 0 ? (packed_codes & 0x0fU) : (packed_codes >> 4);

        const std::int32_t group      = column / 16;
        const std::int32_t row_tile   = row / 128;
        const std::int32_t row_inner  = row % 128;
        const std::int32_t scale_tile = group / 4;
        const std::int32_t scale_lane = group % 4;
        const std::int32_t k_tiles    = weight.k / 64;
        const std::size_t scale_offset =
            packed.scale_plane_offset +
            static_cast<std::size_t>(row_tile * k_tiles + scale_tile) * 512U +
            static_cast<std::size_t>(row_inner % 32) * 16U +
            static_cast<std::size_t>(row_inner / 32) * 4U + static_cast<std::size_t>(scale_lane);
        const double decoded = detail::decode_e2m1(code) *
                               detail::decode_e4m3fn(packed.payload[scale_offset]) /
                               static_cast<double>(weight.weight_scale);
        return static_cast<double>(static_cast<float>(decoded));
    }

    const detail::QuantSpec spec = detail::quant_spec(weight.qtype);
    const std::int32_t padded_k  = weight.padded_shape[1];
    if (padded_k < weight.shape[1] || padded_k % spec.group_size != 0) {
        throw std::invalid_argument("quantized-weight fixture: invalid padded K");
    }
    const std::int32_t groups_per_row = padded_k / spec.group_size;
    const std::int32_t group          = column / spec.group_size;
    const std::int32_t lane           = column - group * spec.group_size;
    const std::size_t group_index     = static_cast<std::size_t>(row) * groups_per_row + group;
    const int nibble_bytes            = detail::nibble_bytes_per_group(spec);
    const int high_bytes              = detail::high_bytes_per_group(spec);

    const std::uint8_t* nibble       = packed.payload.data() + group_index * nibble_bytes;
    const std::uint8_t* high         = high_bytes == 0 ? nullptr
                                                       : packed.payload.data() + packed.high_plane_offset +
                                                     group_index * high_bytes;
    const std::uint16_t stored_scale = detail::load_u16_le(
        packed.payload, packed.scale_plane_offset + group_index * sizeof(std::uint16_t));
    const int signed_code                = detail::unpack_lowbit_code(nibble, high, spec, lane);
    const double exact_stored_fp16_scale = static_cast<double>(detail::f16_to_f32(stored_scale));
    return static_cast<double>(signed_code) * exact_stored_fp16_scale;
}

inline double dot_fp64(const PackedWeight& packed, std::int32_t row, const float* logical_input,
                       std::int32_t k) {
    if (logical_input == nullptr) {
        throw std::invalid_argument("quantized-weight fixture: input must be non-null");
    }
    if (k != packed.weight.shape[1]) {
        throw std::invalid_argument("quantized-weight fixture: logical K mismatch");
    }
    double acc = 0.0;
    for (std::int32_t column = 0; column < k; ++column) {
        acc +=
            logical_weight_fp64(packed, row, column) * static_cast<double>(logical_input[column]);
    }
    return acc;
}

inline std::vector<float> materialize_rows_fp32(const PackedWeight& packed,
                                                std::span<const std::int32_t> rows) {
    if (rows.empty()) {
        throw std::invalid_argument("quantized-weight fixture: row selection is empty");
    }
    std::vector<float> result(static_cast<std::size_t>(packed.weight.k) * rows.size());
    for (std::size_t selected = 0; selected < rows.size(); ++selected) {
        const std::int32_t row = rows[selected];
        for (std::int32_t column = 0; column < packed.weight.k; ++column) {
            result[selected * static_cast<std::size_t>(packed.weight.k) +
                   static_cast<std::size_t>(column)] =
                static_cast<float>(logical_weight_fp64(packed, row, column));
        }
    }
    return result;
}

inline std::vector<float> decode_row_split_lowbit(const std::vector<std::uint8_t>& payload,
                                                  std::int32_t n, std::int32_t k,
                                                  std::int32_t padded_k, QType qtype) {
    const detail::QuantSpec spec = detail::quant_spec(qtype);
    const int nib                = detail::nibble_bytes_per_group(spec);
    const int high_bpr           = detail::high_bytes_per_group(spec);
    const std::int32_t kg        = padded_k / spec.group_size;
    const std::size_t nibble_bytes =
        static_cast<std::size_t>(n) * static_cast<std::size_t>(kg) * static_cast<std::size_t>(nib);
    const std::size_t high_bytes = static_cast<std::size_t>(n) * static_cast<std::size_t>(kg) *
                                   static_cast<std::size_t>(high_bpr);
    const std::size_t high_off  = detail::align_up_size(nibble_bytes, 256);
    const std::size_t scale_off = high_off + detail::align_up_size(high_bytes, 256);

    std::vector<float> deq(static_cast<std::size_t>(n) * k);
    for (std::int32_t row = 0; row < n; ++row) {
        for (std::int32_t g = 0; g < kg; ++g) {
            const std::size_t group_index = static_cast<std::size_t>(row) * kg + g;
            const std::uint16_t scale_h = detail::load_u16_le(payload, scale_off + group_index * 2);
            const float scale           = detail::f16_to_f32(scale_h);
            const std::uint8_t* nibble  = payload.data() + group_index * nib;
            const std::uint8_t* high =
                high_bpr == 0 ? nullptr : payload.data() + high_off + group_index * high_bpr;
            for (std::int32_t lane = 0; lane < spec.group_size; ++lane) {
                const std::int32_t kk = g * spec.group_size + lane;
                if (kk >= k) { continue; }
                const int code = detail::unpack_lowbit_code(nibble, high, spec, lane);
                deq[static_cast<std::size_t>(row) * k + kk] = static_cast<float>(code) * scale;
            }
        }
    }
    return deq;
}

inline PackedWeight pack_row_split_lowbit(const std::vector<float>& source, std::int32_t n,
                                          std::int32_t k, QType qtype) {
    if (n <= 0 || k <= 0) {
        throw std::invalid_argument("row-split test packer: shape must be positive");
    }
    if (source.size() != static_cast<std::size_t>(n) * k) {
        throw std::invalid_argument("row-split test packer: source size mismatch");
    }

    const detail::QuantSpec spec = detail::quant_spec(qtype);
    const std::int32_t padded_k  = detail::align_up(k, 128);
    const std::int32_t kg        = padded_k / spec.group_size;
    const int nib                = detail::nibble_bytes_per_group(spec);
    const int high_bpr           = detail::high_bytes_per_group(spec);

    PackedWeight out;
    out.code_plane_bytes = static_cast<std::uint64_t>(n) * static_cast<std::uint64_t>(kg) *
                           static_cast<std::uint64_t>(nib);
    out.high_plane_offset =
        detail::align_up_size(static_cast<std::size_t>(out.code_plane_bytes), 256);
    out.high_plane_bytes = static_cast<std::uint64_t>(n) * static_cast<std::uint64_t>(kg) *
                           static_cast<std::uint64_t>(high_bpr);
    out.scale_plane_offset =
        out.high_plane_offset +
        detail::align_up_size(static_cast<std::size_t>(out.high_plane_bytes), 256);
    out.scale_plane_bytes = static_cast<std::uint64_t>(n) * static_cast<std::uint64_t>(kg) * 2ULL;
    out.payload.assign(static_cast<std::size_t>(out.scale_plane_offset + out.scale_plane_bytes), 0);

    std::int8_t codes[64];
    float vals[64];
    for (std::int32_t row = 0; row < n; ++row) {
        for (std::int32_t g = 0; g < kg; ++g) {
            float maxabs = 0.0f;
            for (std::int32_t lane = 0; lane < spec.group_size; ++lane) {
                const std::int32_t kk = g * spec.group_size + lane;
                const float v = (kk < k) ? source[static_cast<std::size_t>(row) * k + kk] : 0.0f;
                vals[lane]    = v;
                maxabs        = std::max(maxabs, std::abs(v));
            }

            std::uint16_t scale_h = detail::f32_to_f16(maxabs / static_cast<float>(spec.qmax));
            if (scale_h == 0 && maxabs > 0.0f) { scale_h = 0x0001u; }
            const float scale = detail::f16_to_f32(scale_h);
            const float inv   = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
            for (std::int32_t lane = 0; lane < spec.group_size; ++lane) {
                const float q = std::nearbyint(vals[lane] * inv);
                const int qi  = std::min(spec.qmax, std::max(spec.qmin, static_cast<int>(q)));
                codes[lane]   = static_cast<std::int8_t>(qi);
            }

            const std::size_t group_index = static_cast<std::size_t>(row) * kg + g;
            detail::pack_lowbit_group(codes, spec, out.payload.data() + group_index * nib,
                                      high_bpr == 0 ? nullptr
                                                    : out.payload.data() + out.high_plane_offset +
                                                          group_index * high_bpr);
            detail::store_u16_le(out.payload, out.scale_plane_offset + group_index * 2, scale_h);
        }
    }

    out.dequant = decode_row_split_lowbit(out.payload, n, k, padded_k, qtype);

    out.weight.qtype            = qtype;
    out.weight.layout           = QuantLayout::RowSplit;
    out.weight.scale_dtype      = DType::FP16;
    out.weight.device           = Device::cpu();
    out.weight.payload          = out.payload.data();
    out.weight.payload_bytes    = out.payload.size();
    out.weight.high_plane_bytes = out.high_plane_bytes;
    out.weight.qdata            = out.payload.data();
    out.weight.qhigh =
        out.high_plane_bytes == 0 ? nullptr : out.payload.data() + out.high_plane_offset;
    out.weight.scales          = out.payload.data() + out.scale_plane_offset;
    out.weight.group_size      = static_cast<std::uint32_t>(spec.group_size);
    out.weight.group           = spec.group_size;
    out.weight.ndim            = 2;
    out.weight.shape[0]        = n;
    out.weight.shape[1]        = k;
    out.weight.shape[2]        = 1;
    out.weight.shape[3]        = 1;
    out.weight.padded_shape[0] = n;
    out.weight.padded_shape[1] = padded_k;
    out.weight.padded_shape[2] = 1;
    out.weight.padded_shape[3] = 1;
    out.weight.n               = n;
    out.weight.k               = k;
    return out;
}

inline PackedWeight pack_q4_row_split(const std::vector<float>& source, std::int32_t n,
                                      std::int32_t k) {
    return pack_row_split_lowbit(source, n, k, QType::Q4_G64_FP16);
}

inline PackedWeight pack_q5_row_split(const std::vector<float>& source, std::int32_t n,
                                      std::int32_t k) {
    return pack_row_split_lowbit(source, n, k, QType::Q5_G64_FP16);
}

inline PackedWeight pack_q6_row_split(const std::vector<float>& source, std::int32_t n,
                                      std::int32_t k) {
    return pack_row_split_lowbit(source, n, k, QType::Q6_G64_FP16);
}

inline PackedWeight pack_q8_g32_row_split(const std::vector<float>& source, std::int32_t n,
                                          std::int32_t k) {
    return pack_row_split_lowbit(source, n, k, QType::Q8_G32_FP16);
}

} // namespace ninfer::test::quantized_weight
