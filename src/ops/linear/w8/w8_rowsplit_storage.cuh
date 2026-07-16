#pragma once

#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

struct W8RowSplitStorage {
    static constexpr int kGroupK             = 32;
    static constexpr int kCodeBytesPerGroup  = 32;
    static constexpr int kHighBytesPerGroup  = 0;
    static constexpr int kScaleBytesPerGroup = 2;
};

struct W8ScalarDecodeAtom {
    static constexpr int kGroupK = W8RowSplitStorage::kGroupK;

    __device__ static __forceinline__ void
    load_pair(const std::uint8_t* codes, const std::uint8_t* /*high*/, const std::uint8_t* scales,
              std::int64_t group_index, int lane, float& w0, float& w1) {
        if (lane >= kGroupK / 2) {
            w0 = 0.0f;
            w1 = 0.0f;
            return;
        }
        const float scale = __half2float(
            __ushort_as_half(*reinterpret_cast<const std::uint16_t*>(scales + group_index * 2)));
        const std::uint8_t* packed = codes + group_index * W8RowSplitStorage::kCodeBytesPerGroup +
                                     static_cast<std::int64_t>(lane) * 2;
        w0 = static_cast<float>(static_cast<std::int8_t>(packed[0])) * scale;
        w1 = static_cast<float>(static_cast<std::int8_t>(packed[1])) * scale;
    }
};

} // namespace ninfer::ops::detail
