#pragma once

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Coordinates are logical output rows and tokens of the complete call, including
// when a launcher splits the CUDA grid into several launches.
struct LinearBf16Output {
    __nv_bfloat16* data;
    std::int32_t rows;

    __device__ __forceinline__ __nv_bfloat16* at(int row, int token) const {
        return data + static_cast<std::int64_t>(token) * rows + row;
    }

    __device__ __forceinline__ void store(int row, int token, float value) const {
        *at(row, token) = __float2bfloat16_rn(value);
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

// A bound segment still receives coordinates of the complete contraction.
struct LinearBf16SegmentOutput {
    __nv_bfloat16* data;
    int rows;
    int parent_row_begin;

    __device__ __forceinline__ void store(int row, int token, float value) const {
        data[static_cast<std::int64_t>(token) * rows + row - parent_row_begin] =
            __float2bfloat16_rn(value);
    }
};

template <int... SegmentRows>
struct LinearBf16SegmentedOutput {
    static_assert(sizeof...(SegmentRows) > 0 && ((SegmentRows > 0) && ...));
    __nv_bfloat16* data[sizeof...(SegmentRows)];

    // Constant indices keep segment pointers in registers. A runtime array index
    // would spill the pointer array into local memory in CUDA decode kernels.
    template <int Index = 0, int Begin = 0>
    __device__ __forceinline__ LinearBf16SegmentOutput segment(int row) const {
        constexpr int sizes[]{SegmentRows...};
        constexpr int rows = sizes[Index];
        if constexpr (Index + 1 == sizeof...(SegmentRows)) {
            return {data[Index], rows, Begin};
        } else {
            if (row < Begin + rows) return {data[Index], rows, Begin};
            return segment<Index + 1, Begin + rows>(row);
        }
    }

    __device__ __forceinline__ void store(int row, int token, float value) const {
        segment(row).store(row, token, value);
    }

    template <int TileRows>
    __device__ __forceinline__ auto bind_tile(int row_begin) const {
        if constexpr (((SegmentRows % TileRows == 0) && ...))
            return segment(row_begin);
        else
            return *this;
    }
};

template <int TileRows, class Output>
__device__ __forceinline__ auto linear_output_tile(Output output, int row_begin) {
    if constexpr (requires { output.template bind_tile<TileRows>(row_begin); })
        return output.template bind_tile<TileRows>(row_begin);
    else
        return output;
}

} // namespace ninfer::ops::detail
