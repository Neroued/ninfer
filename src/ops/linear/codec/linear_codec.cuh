#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

__device__ __forceinline__ float codec_load_scale_f16(const std::uint8_t* scales,
                                                      std::int64_t group_index) {
    const auto* p = reinterpret_cast<const std::uint16_t*>(scales + group_index * 2);
    return __half2float(__ushort_as_half(*p));
}

struct Q4Codec {
    static constexpr int kBits                = 4;
    static constexpr int kGroupK              = 64;
    static constexpr int kBytesPerRowPerGroup = 32; // ceil(64*4/8)

    __device__ static void load_group(const std::uint8_t* codes, const std::uint8_t* high,
                                      const std::uint8_t* scales, std::int32_t row,
                                      std::int32_t group, std::int32_t kg, float out[kGroupK]) {
        (void)high;
        const std::int64_t group_index = static_cast<std::int64_t>(row) * kg + group;
        const float scale              = codec_load_scale_f16(scales, group_index);
        const std::uint8_t* packed     = codes + group_index * kBytesPerRowPerGroup;
        for (int lane = 0; lane < kGroupK; ++lane) {
            const std::uint8_t byte = packed[lane >> 1];
            const int u             = (lane & 1) ? (byte >> 4) : (byte & 0x0f);
            const int s             = (u ^ 0x08) - 0x08; // sign-extend bit 3
            out[lane]               = static_cast<float>(s) * scale;
        }
    }

    // Dequantize the two group values a warp lane owns (values 2*lane, 2*lane+1)
    // for the group at group_index. Identical math to load_group; used by the
    // warp-per-row multi-step GEMM so weight dequant is shared across T columns.
    __device__ static void load_pair(const std::uint8_t* codes, const std::uint8_t* /*high*/,
                                     const std::uint8_t* scales, std::int64_t group_index, int lane,
                                     float& w0, float& w1) {
        const float scale       = codec_load_scale_f16(scales, group_index);
        const std::uint8_t byte = codes[group_index * kBytesPerRowPerGroup + lane];
        const int u0            = byte & 0x0f;
        const int u1            = byte >> 4;
        const int s0            = (u0 ^ 0x08) - 0x08;
        const int s1            = (u1 ^ 0x08) - 0x08;
        w0                      = static_cast<float>(s0) * scale;
        w1                      = static_cast<float>(s1) * scale;
    }

    static __device__ __forceinline__ __nv_bfloat162 load_pair_bf162(const std::uint8_t* codes,
                                                                     const std::uint8_t* high,
                                                                     const std::uint8_t* scales,
                                                                     std::int64_t group_index,
                                                                     int lane) {
        return load_pair_bf162_scale_ptr(codes, high, scales + group_index * 2, group_index, lane);
    }

    static __device__ __forceinline__ __nv_bfloat162
    load_pair_bf162_scale_ptr(const std::uint8_t* codes, const std::uint8_t* /*high*/,
                              const std::uint8_t* scale_ptr, std::int64_t group_index, int lane) {
        const float scale =
            __half2float(__ushort_as_half(*reinterpret_cast<const std::uint16_t*>(scale_ptr)));
        const std::uint8_t byte = codes[group_index * kBytesPerRowPerGroup + lane];
        const int u0            = byte & 0x0f;
        const int u1            = byte >> 4;
        const int s0            = (u0 ^ 0x08) - 0x08;
        const int s1            = (u1 ^ 0x08) - 0x08;
        return __floats2bfloat162_rn(static_cast<float>(s0) * scale,
                                     static_cast<float>(s1) * scale);
    }
};

} // namespace ninfer::ops::detail
