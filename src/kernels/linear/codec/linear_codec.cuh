#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace qus::kernels::detail {

struct Q4Codec {
    static constexpr int kBits = 4;
    static constexpr int kGroupK = 64;
    static constexpr int kBytesPerRowPerGroup = 32;            // ceil(64*4/8)
    __device__ static void load_group(const std::uint8_t* codes, const std::uint8_t* high,
                                      const std::uint8_t* scales, std::int32_t row,
                                      std::int32_t group, std::int32_t kg, float out[kGroupK]) {
        (void)high;
        const std::int64_t group_index = static_cast<std::int64_t>(row) * kg + group;
        const std::uint16_t sb  = static_cast<std::uint16_t>(scales[group_index * 2]) |
                                  static_cast<std::uint16_t>(
                                      static_cast<std::uint16_t>(scales[group_index * 2 + 1]) << 8);
        const float scale = __half2float(__ushort_as_half(sb));
        const std::uint8_t* packed = codes + group_index * kBytesPerRowPerGroup;
        for (int lane = 0; lane < kGroupK; ++lane) {
            const std::uint8_t byte = packed[lane >> 1];
            const int u = (lane & 1) ? (byte >> 4) : (byte & 0x0f);
            const int s = (u ^ 0x08) - 0x08;                  // sign-extend bit 3
            out[lane] = static_cast<float>(s) * scale;
        }
    }

    // Dequantize the two group values a warp lane owns (values 2*lane, 2*lane+1)
    // for the group at group_index. Identical math to load_group; used by the
    // warp-per-row multi-step GEMM so weight dequant is shared across T columns.
    __device__ static void load_pair(const std::uint8_t* codes, const std::uint8_t* /*high*/,
                                     const std::uint8_t* scales, std::int64_t group_index, int lane,
                                     float& w0, float& w1) {
        const std::uint16_t sb =
            static_cast<std::uint16_t>(scales[group_index * 2]) |
            static_cast<std::uint16_t>(static_cast<std::uint16_t>(scales[group_index * 2 + 1]) << 8);
        const float         scale = __half2float(__ushort_as_half(sb));
        const std::uint8_t  byte  = codes[group_index * kBytesPerRowPerGroup + lane];
        const int           u0    = byte & 0x0f;
        const int           u1    = byte >> 4;
        const int           s0    = (u0 ^ 0x08) - 0x08;
        const int           s1    = (u1 ^ 0x08) - 0x08;
        w0 = static_cast<float>(s0) * scale;
        w1 = static_cast<float>(s1) * scale;
    }
};

struct Q5Codec {
    static constexpr int kBits = 5;
    static constexpr int kGroupK = 64;
    static constexpr int kNibbleBytesPerRowPerGroup = 32;
    static constexpr int kHighBytesPerRowPerGroup = 8;
    __device__ static void load_group(const std::uint8_t* codes, const std::uint8_t* high,
                                      const std::uint8_t* scales, std::int32_t row,
                                      std::int32_t group, std::int32_t kg, float out[kGroupK]) {
        const std::int64_t group_index = static_cast<std::int64_t>(row) * kg + group;
        const std::uint16_t sb  = static_cast<std::uint16_t>(scales[group_index * 2]) |
                                  static_cast<std::uint16_t>(
                                      static_cast<std::uint16_t>(scales[group_index * 2 + 1]) << 8);
        const float scale = __half2float(__ushort_as_half(sb));
        const std::uint8_t* nibble = codes + group_index * kNibbleBytesPerRowPerGroup;
        const std::uint8_t* high_bits = high + group_index * kHighBytesPerRowPerGroup;
#pragma unroll
        for (int lane = 0; lane < kGroupK; ++lane) {
            const std::uint8_t byte = nibble[lane >> 1];
            const int low = (lane & 1) ? (byte >> 4) : (byte & 0x0f);
            const int hi = (high_bits[lane >> 3] >> (lane & 7)) & 0x01;
            const int u = low | (hi << 4);
            const int s = (u & 0x10) ? (u - 32) : u;
            out[lane] = static_cast<float>(s) * scale;
        }
    }

    __device__ static void load_pair(const std::uint8_t* codes, const std::uint8_t* high,
                                     const std::uint8_t* scales, std::int64_t group_index, int lane,
                                     float& w0, float& w1) {
        const std::uint16_t sb =
            static_cast<std::uint16_t>(scales[group_index * 2]) |
            static_cast<std::uint16_t>(static_cast<std::uint16_t>(scales[group_index * 2 + 1]) << 8);
        const float          scale     = __half2float(__ushort_as_half(sb));
        const std::uint8_t   byte      = codes[group_index * kNibbleBytesPerRowPerGroup + lane];
        const std::uint8_t*  high_bits = high + group_index * kHighBytesPerRowPerGroup;
        const int            v0        = lane * 2;
        const int            v1        = v0 + 1;
        const int            lo0       = byte & 0x0f;
        const int            lo1       = byte >> 4;
        const int            hi0       = (high_bits[v0 >> 3] >> (v0 & 7)) & 0x01;
        const int            hi1       = (high_bits[v1 >> 3] >> (v1 & 7)) & 0x01;
        const int            u0        = lo0 | (hi0 << 4);
        const int            u1        = lo1 | (hi1 << 4);
        const int            s0        = (u0 & 0x10) ? (u0 - 32) : u0;
        const int            s1        = (u1 & 0x10) ? (u1 - 32) : u1;
        w0 = static_cast<float>(s0) * scale;
        w1 = static_cast<float>(s1) * scale;
    }
};

struct Q6Codec {
    static constexpr int kBits = 6;
    static constexpr int kGroupK = 64;
    static constexpr int kNibbleBytesPerRowPerGroup = 32;
    static constexpr int kHighBytesPerRowPerGroup = 16;
    __device__ static void load_group(const std::uint8_t* codes, const std::uint8_t* high,
                                      const std::uint8_t* scales, std::int32_t row,
                                      std::int32_t group, std::int32_t kg, float out[kGroupK]) {
        const std::int64_t group_index = static_cast<std::int64_t>(row) * kg + group;
        const std::uint16_t sb  = static_cast<std::uint16_t>(scales[group_index * 2]) |
                                  static_cast<std::uint16_t>(
                                      static_cast<std::uint16_t>(scales[group_index * 2 + 1]) << 8);
        const float scale = __half2float(__ushort_as_half(sb));
        const std::uint8_t* nibble = codes + group_index * kNibbleBytesPerRowPerGroup;
        const std::uint8_t* high_bits = high + group_index * kHighBytesPerRowPerGroup;
#pragma unroll
        for (int lane = 0; lane < kGroupK; ++lane) {
            const std::uint8_t byte = nibble[lane >> 1];
            const int low = (lane & 1) ? (byte >> 4) : (byte & 0x0f);
            const int bitpos = lane * 2;
            const int hi = (high_bits[bitpos >> 3] >> (bitpos & 7)) & 0x03;
            const int u = low | (hi << 4);
            const int s = (u & 0x20) ? (u - 64) : u;
            out[lane] = static_cast<float>(s) * scale;
        }
    }

    __device__ static void load_pair(const std::uint8_t* codes, const std::uint8_t* high,
                                     const std::uint8_t* scales, std::int64_t group_index, int lane,
                                     float& w0, float& w1) {
        const std::uint16_t sb =
            static_cast<std::uint16_t>(scales[group_index * 2]) |
            static_cast<std::uint16_t>(static_cast<std::uint16_t>(scales[group_index * 2 + 1]) << 8);
        const float          scale     = __half2float(__ushort_as_half(sb));
        const std::uint8_t   byte      = codes[group_index * kNibbleBytesPerRowPerGroup + lane];
        const std::uint8_t*  high_bits = high + group_index * kHighBytesPerRowPerGroup;
        const int            lo0       = byte & 0x0f;
        const int            lo1       = byte >> 4;
        const int            bp0       = (lane * 2) * 2;       // value 2*lane, bitpos = value*2
        const int            bp1       = (lane * 2 + 1) * 2;   // value 2*lane+1
        const int            hi0       = (high_bits[bp0 >> 3] >> (bp0 & 7)) & 0x03;
        const int            hi1       = (high_bits[bp1 >> 3] >> (bp1 & 7)) & 0x03;
        const int            u0        = lo0 | (hi0 << 4);
        const int            u1        = lo1 | (hi1 << 4);
        const int            s0        = (u0 & 0x20) ? (u0 - 64) : u0;
        const int            s1        = (u1 & 0x20) ? (u1 - 64) : u1;
        w0 = static_cast<float>(s0) * scale;
        w1 = static_cast<float>(s1) * scale;
    }
};

} // namespace qus::kernels::detail
