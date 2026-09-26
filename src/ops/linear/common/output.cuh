#pragma once

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Coordinates are logical output rows and tokens of the complete call, including
// when a launcher splits the CUDA grid into several launches.
struct LinearBf16Output {
    __nv_bfloat16* data;
    std::int32_t rows;

    __device__ __forceinline__ void store(int row, int token, float value) const {
        data[static_cast<std::int64_t>(token) * rows + row] = __float2bfloat16_rn(value);
    }
};

struct LinearBf16StridedOutput {
    __nv_bfloat16* data;
    std::int64_t leading_dim;
    std::int32_t row_begin;

    __device__ __forceinline__ void store(int row, int token, float value) const {
        data[static_cast<std::int64_t>(token) * leading_dim + row_begin + row] =
            __float2bfloat16_rn(value);
    }
};

struct LinearBf16InputView {
    const __nv_bfloat16* data;
    std::int64_t leading_dim;
    std::int32_t row_begin = 0;

    __device__ __forceinline__ float load(int row, int token) const {
        return __bfloat162float(
            data[static_cast<std::int64_t>(token) * leading_dim + row_begin + row]);
    }
};

template <int SplitRow>
struct LinearBf16SplitOutput2 {
    static_assert(SplitRow > 0);
    LinearBf16StridedOutput first;
    LinearBf16StridedOutput second;

    __device__ __forceinline__ void store(int row, int token, float value) const {
        if (row < SplitRow)
            first.store(row, token, value);
        else
            second.store(row - SplitRow, token, value);
    }
};

} // namespace ninfer::ops::detail
