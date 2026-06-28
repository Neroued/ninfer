#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace qus::kernels::detail {

struct Q4Codec {
    static constexpr int kBits = 4;
    static constexpr int kGroupK = 64;
    static constexpr int kBytesPerRowPerGroup = 32;            // ceil(64*4/8)
    static constexpr int kTileBytes = 64 * 2 + 64 * kBytesPerRowPerGroup;  // 2176
    __device__ static void load_group(const std::uint8_t* payload, std::int32_t row,
                                      std::int32_t group, std::int32_t kg, float out[kGroupK]) {
        const std::int32_t tile = row / 64;
        const std::int32_t rit  = row - tile * 64;
        const std::int64_t off  = (static_cast<std::int64_t>(tile) * kg + group) * kTileBytes;
        const std::uint16_t sb  = static_cast<std::uint16_t>(payload[off + rit * 2]) |
                                  static_cast<std::uint16_t>(
                                      static_cast<std::uint16_t>(payload[off + rit * 2 + 1]) << 8);
        const float scale = __half2float(__ushort_as_half(sb));
        const std::uint8_t* packed =
            payload + off + 64 * 2 + static_cast<std::int64_t>(rit) * kBytesPerRowPerGroup;
        for (int lane = 0; lane < kGroupK; ++lane) {
            const std::uint8_t byte = packed[lane >> 1];
            const int u = (lane & 1) ? (byte >> 4) : (byte & 0x0f);
            const int s = (u & 0x08) ? (u - 16) : u;          // sign-extend bit 3
            out[lane] = static_cast<float>(s) * scale;
        }
    }
};

struct Q5Codec {
    static constexpr int kBits = 5;
    static constexpr int kGroupK = 64;
    static constexpr int kBytesPerRowPerGroup = 40;           // ceil(64*5/8)
    static constexpr int kTileBytes = 64 * 2 + 64 * kBytesPerRowPerGroup;  // 2688
    __device__ static void load_group(const std::uint8_t* payload, std::int32_t row,
                                      std::int32_t group, std::int32_t kg, float out[kGroupK]) {
        const std::int32_t tile = row / 64;
        const std::int32_t rit  = row - tile * 64;
        const std::int64_t off  = (static_cast<std::int64_t>(tile) * kg + group) * kTileBytes;
        const std::uint16_t sb  = static_cast<std::uint16_t>(payload[off + rit * 2]) |
                                  static_cast<std::uint16_t>(
                                      static_cast<std::uint16_t>(payload[off + rit * 2 + 1]) << 8);
        const float scale = __half2float(__ushort_as_half(sb));
        const std::uint8_t* packed =
            payload + off + 64 * 2 + static_cast<std::int64_t>(rit) * kBytesPerRowPerGroup;
        for (int lane = 0; lane < kGroupK; ++lane) {
            std::uint32_t u = 0;
            const int bitpos = lane * kBits;
            for (int b = 0; b < kBits; ++b) {
                if ((packed[(bitpos + b) >> 3] & (1u << ((bitpos + b) & 7))) != 0) { u |= 1u << b; }
            }
            const int s = (u & 0x10u) ? (static_cast<int>(u) - 32) : static_cast<int>(u);  // bit 4
            out[lane] = static_cast<float>(s) * scale;
        }
    }
};

struct Q6Codec {
    static constexpr int kBits = 6;
    static constexpr int kGroupK = 64;
    static constexpr int kBytesPerRowPerGroup = 48;           // ceil(64*6/8)
    static constexpr int kTileBytes = 64 * 2 + 64 * kBytesPerRowPerGroup;  // 3200
    __device__ static void load_group(const std::uint8_t* payload, std::int32_t row,
                                      std::int32_t group, std::int32_t kg, float out[kGroupK]) {
        const std::int32_t tile = row / 64;
        const std::int32_t rit  = row - tile * 64;
        const std::int64_t off  = (static_cast<std::int64_t>(tile) * kg + group) * kTileBytes;
        const std::uint16_t sb  = static_cast<std::uint16_t>(payload[off + rit * 2]) |
                                  static_cast<std::uint16_t>(
                                      static_cast<std::uint16_t>(payload[off + rit * 2 + 1]) << 8);
        const float scale = __half2float(__ushort_as_half(sb));
        const std::uint8_t* packed =
            payload + off + 64 * 2 + static_cast<std::int64_t>(rit) * kBytesPerRowPerGroup;
        for (int lane = 0; lane < kGroupK; ++lane) {
            std::uint32_t u = 0;
            const int bitpos = lane * kBits;
            for (int b = 0; b < kBits; ++b) {
                if ((packed[(bitpos + b) >> 3] & (1u << ((bitpos + b) & 7))) != 0) { u |= 1u << b; }
            }
            const int s = (u & 0x20u) ? (static_cast<int>(u) - 64) : static_cast<int>(u);  // bit 5
            out[lane] = static_cast<float>(s) * scale;
        }
    }
};

} // namespace qus::kernels::detail
