#pragma once

// Test-only q5090 ROW_SPLIT packing helpers. Mirrors
// tools/q5090_convert/{quantize.py,packing.py,layouts.py} for low-bit weights:
// fp16-round per-row/group scales, signed two's-complement LSB-first low-bit
// codes, a contiguous code plane, 256-byte padding, and a separate scale plane.

#include "qus/core/tensor.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace qus::test::q5090 {
namespace detail {

inline std::int32_t align_up(std::int32_t x, std::int32_t m) { return ((x + m - 1) / m) * m; }

inline std::size_t align_up_size(std::size_t x, std::size_t m) {
    return ((x + m - 1) / m) * m;
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

inline std::uint16_t f32_to_bf16(float f) {
    std::uint32_t u = float_bits(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) { return static_cast<std::uint16_t>((u >> 16) | 0x0040u); }
    const std::uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    return static_cast<std::uint16_t>(u >> 16);
}

inline float bf16_to_f32(std::uint16_t h) {
    return bits_float(static_cast<std::uint32_t>(h) << 16);
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

inline void store_u16_le(std::vector<std::uint8_t>& payload, std::size_t off, std::uint16_t v) {
    payload[off]     = static_cast<std::uint8_t>(v & 0xffu);
    payload[off + 1] = static_cast<std::uint8_t>(v >> 8);
}

inline std::uint16_t load_u16_le(const std::vector<std::uint8_t>& payload, std::size_t off) {
    return static_cast<std::uint16_t>(payload[off]) |
           static_cast<std::uint16_t>(static_cast<std::uint16_t>(payload[off + 1]) << 8);
}

struct QuantSpec {
    int bits;
    int group_size;
    int qmax;
    int qmin;
};

inline QuantSpec quant_spec(QType qtype) {
    switch (qtype) {
    case QType::Q4G64_F16S:
        return {4, 64, 7, -8};
    case QType::Q5G64_F16S:
        return {5, 64, 15, -16};
    case QType::Q6G64_F16S:
        return {6, 64, 31, -32};
    default:
        throw std::invalid_argument("q5090 test packer: unsupported qtype");
    }
}

inline int bytes_per_group(const QuantSpec& spec) { return (spec.group_size * spec.bits + 7) / 8; }

inline void pack_lowbit_group(const std::int8_t* codes, const QuantSpec& spec, std::uint8_t* out) {
    const int bpr = bytes_per_group(spec);
    std::fill(out, out + bpr, static_cast<std::uint8_t>(0));
    const std::uint32_t mask = (1u << spec.bits) - 1u;
    for (int i = 0; i < spec.group_size; ++i) {
        const std::uint32_t u = static_cast<std::uint32_t>(codes[i]) & mask;
        const int bit_pos     = i * spec.bits;
        for (int b = 0; b < spec.bits; ++b) {
            if ((u & (1u << b)) != 0) {
                out[(bit_pos + b) >> 3] |= static_cast<std::uint8_t>(1u << ((bit_pos + b) & 7));
            }
        }
    }
}

inline int unpack_lowbit_code(const std::uint8_t* packed, const QuantSpec& spec, int index) {
    std::uint32_t u  = 0;
    const int bitpos = index * spec.bits;
    for (int b = 0; b < spec.bits; ++b) {
        if ((packed[(bitpos + b) >> 3] & (1u << ((bitpos + b) & 7))) != 0) { u |= 1u << b; }
    }
    const std::uint32_t sign = 1u << (spec.bits - 1);
    const std::uint32_t span = 1u << spec.bits;
    return (u & sign) ? static_cast<int>(u) - static_cast<int>(span) : static_cast<int>(u);
}

} // namespace detail

struct PackedWeight {
    std::vector<std::uint8_t> payload;
    std::vector<float> dequant;
    Weight weight{};
    std::uint64_t code_plane_bytes = 0;
    std::uint64_t scale_plane_offset = 0;
    std::uint64_t scale_plane_bytes = 0;

    Weight device_weight(void* device_payload) const {
        Weight w  = weight;
        w.payload = device_payload;
        if (device_payload != nullptr) {
            w.qdata  = device_payload;
            w.scales = static_cast<std::uint8_t*>(device_payload) + scale_plane_offset;
        } else {
            w.qdata  = nullptr;
            w.scales = nullptr;
        }
        return w;
    }
};

inline std::vector<float> decode_row_split_lowbit(const std::vector<std::uint8_t>& payload,
                                                  std::int32_t n, std::int32_t k,
                                                  std::int32_t padded_k, QType qtype) {
    const detail::QuantSpec spec = detail::quant_spec(qtype);
    const int bpr                = detail::bytes_per_group(spec);
    const std::int32_t kg        = padded_k / spec.group_size;
    const std::size_t code_bytes =
        static_cast<std::size_t>(n) * static_cast<std::size_t>(kg) * static_cast<std::size_t>(bpr);
    const std::size_t scale_off = detail::align_up_size(code_bytes, 256);

    std::vector<float> deq(static_cast<std::size_t>(n) * k);
    for (std::int32_t row = 0; row < n; ++row) {
        for (std::int32_t g = 0; g < kg; ++g) {
            const std::size_t group_index = static_cast<std::size_t>(row) * kg + g;
            const std::uint16_t scale_h =
                detail::load_u16_le(payload, scale_off + group_index * 2);
            const float scale = detail::f16_to_f32(scale_h);
            const std::uint8_t* packed = payload.data() + group_index * bpr;
            for (std::int32_t lane = 0; lane < spec.group_size; ++lane) {
                const std::int32_t kk = g * spec.group_size + lane;
                if (kk >= k) { continue; }
                const int code = detail::unpack_lowbit_code(packed, spec, lane);
                deq[static_cast<std::size_t>(row) * k + kk] = static_cast<float>(code) * scale;
            }
        }
    }
    return deq;
}

inline PackedWeight pack_row_split_lowbit(const std::vector<float>& source, std::int32_t n,
                                          std::int32_t k, QType qtype) {
    if (n <= 0 || k <= 0) {
        throw std::invalid_argument("q5090 test packer: shape must be positive");
    }
    if (source.size() != static_cast<std::size_t>(n) * k) {
        throw std::invalid_argument("q5090 test packer: source size mismatch");
    }

    const detail::QuantSpec spec = detail::quant_spec(qtype);
    const std::int32_t padded_k  = detail::align_up(k, spec.group_size);
    const std::int32_t kg        = padded_k / spec.group_size;
    const int bpr                = detail::bytes_per_group(spec);

    PackedWeight out;
    out.code_plane_bytes =
        static_cast<std::uint64_t>(n) * static_cast<std::uint64_t>(kg) * static_cast<std::uint64_t>(bpr);
    out.scale_plane_offset = detail::align_up_size(static_cast<std::size_t>(out.code_plane_bytes), 256);
    out.scale_plane_bytes =
        static_cast<std::uint64_t>(n) * static_cast<std::uint64_t>(kg) * 2ULL;
    out.payload.assign(static_cast<std::size_t>(out.scale_plane_offset + out.scale_plane_bytes), 0);

    std::int8_t codes[64];
    float vals[64];
    for (std::int32_t row = 0; row < n; ++row) {
        for (std::int32_t g = 0; g < kg; ++g) {
            float maxabs = 0.0f;
            for (std::int32_t lane = 0; lane < spec.group_size; ++lane) {
                const std::int32_t kk = g * spec.group_size + lane;
                const float v =
                    (kk < k) ? source[static_cast<std::size_t>(row) * k + kk] : 0.0f;
                vals[lane] = v;
                maxabs     = std::max(maxabs, std::abs(v));
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
            detail::pack_lowbit_group(codes, spec, out.payload.data() + group_index * bpr);
            detail::store_u16_le(out.payload, out.scale_plane_offset + group_index * 2, scale_h);
        }
    }

    out.dequant = decode_row_split_lowbit(out.payload, n, k, padded_k, qtype);

    out.weight.qtype             = qtype;
    out.weight.layout            = QuantLayout::RowSplit;
    out.weight.q5090_scale_dtype = ScaleDType::FP16;
    out.weight.payload           = out.payload.data();
    out.weight.payload_bytes     = out.payload.size();
    out.weight.qdata             = out.payload.data();
    out.weight.scales            = out.payload.data() + out.scale_plane_offset;
    out.weight.group_size        = static_cast<std::uint32_t>(spec.group_size);
    out.weight.group             = spec.group_size;
    out.weight.ndim              = 2;
    out.weight.shape[0]          = n;
    out.weight.shape[1]          = k;
    out.weight.shape[2]          = 1;
    out.weight.shape[3]          = 1;
    out.weight.padded_shape[0]   = n;
    out.weight.padded_shape[1]   = padded_k;
    out.weight.padded_shape[2]   = 1;
    out.weight.padded_shape[3]   = 1;
    out.weight.n                 = n;
    out.weight.k                 = k;
    return out;
}

inline PackedWeight pack_q4_row_split(const std::vector<float>& source, std::int32_t n,
                                      std::int32_t k) {
    return pack_row_split_lowbit(source, n, k, QType::Q4G64_F16S);
}

inline PackedWeight pack_q5_row_split(const std::vector<float>& source, std::int32_t n,
                                      std::int32_t k) {
    return pack_row_split_lowbit(source, n, k, QType::Q5G64_F16S);
}

inline PackedWeight pack_q6_row_split(const std::vector<float>& source, std::int32_t n,
                                      std::int32_t k) {
    return pack_row_split_lowbit(source, n, k, QType::Q6G64_F16S);
}

} // namespace qus::test::q5090
