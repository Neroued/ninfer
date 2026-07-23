#pragma once

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

struct W8OutputTile {
    __nv_bfloat16* data;
    std::int32_t leading_dim;
    std::int32_t parent_row_begin;

    __device__ __forceinline__ __nv_bfloat16* at(std::int32_t parent_row, std::int32_t col) const {
        return data + static_cast<std::int64_t>(col) * leading_dim + parent_row - parent_row_begin;
    }

    __device__ __forceinline__ bool valid(std::int32_t parent_row, std::int32_t total_rows) const {
        return parent_row < total_rows;
    }
};

// Split-output callers must align CTA row tiles to every segment boundary so one tile never
// straddles two final allocations.

struct W8ContiguousOutput {
    __nv_bfloat16* data;
    std::int32_t leading_dim;

    __device__ __forceinline__ std::int32_t row_begin(std::int32_t block,
                                                      std::int32_t tile_rows) const {
        return block * tile_rows;
    }

    __device__ __forceinline__ W8OutputTile tile(std::int32_t /*parent_row_begin*/) const {
        return {data, leading_dim, 0};
    }
};

template <std::int32_t Rows0, std::int32_t Rows1>
struct W8SplitOutput2 {
    static_assert(Rows0 > 0 && Rows1 > 0);

    __nv_bfloat16* out0;
    __nv_bfloat16* out1;

    __device__ __forceinline__ std::int32_t row_begin(std::int32_t block,
                                                      std::int32_t tile_rows) const {
        return block * tile_rows;
    }

    __device__ __forceinline__ W8OutputTile tile(std::int32_t parent_row_begin) const {
        if (parent_row_begin < Rows0) { return {out0, Rows0, 0}; }
        return {out1, Rows1, Rows0};
    }
};

template <std::int32_t Rows0, std::int32_t Rows1, std::int32_t Rows2>
struct W8SplitOutput3 {
    static_assert(Rows0 > 0 && Rows1 > 0 && Rows2 > 0);

    __nv_bfloat16* out0;
    __nv_bfloat16* out1;
    __nv_bfloat16* out2;

    __device__ __forceinline__ std::int32_t row_begin(std::int32_t block,
                                                      std::int32_t tile_rows) const {
        return block * tile_rows;
    }

    __device__ __forceinline__ W8OutputTile tile(std::int32_t parent_row_begin) const {
        constexpr std::int32_t split1 = Rows0;
        constexpr std::int32_t split2 = Rows0 + Rows1;
        if (parent_row_begin < split1) { return {out0, Rows0, 0}; }
        if (parent_row_begin < split2) { return {out1, Rows1, split1}; }
        return {out2, Rows2, split2};
    }
};

template <std::int32_t Rows0, std::int32_t Rows1, std::int32_t Rows2, std::int32_t Rows3>
struct W8SplitOutput4 {
    static_assert(Rows0 > 0 && Rows1 > 0 && Rows2 > 0 && Rows3 > 0);

    __nv_bfloat16* out0;
    __nv_bfloat16* out1;
    __nv_bfloat16* out2;
    __nv_bfloat16* out3;

    __device__ __forceinline__ std::int32_t row_begin(std::int32_t block,
                                                      std::int32_t tile_rows) const {
        return block * tile_rows;
    }

    __device__ __forceinline__ W8OutputTile tile(std::int32_t parent_row_begin) const {
        constexpr std::int32_t split1 = Rows0;
        constexpr std::int32_t split2 = Rows0 + Rows1;
        constexpr std::int32_t split3 = Rows0 + Rows1 + Rows2;
        if (parent_row_begin < split1) { return {out0, Rows0, 0}; }
        if (parent_row_begin < split2) { return {out1, Rows1, split1}; }
        if (parent_row_begin < split3) { return {out2, Rows2, split2}; }
        return {out3, Rows3, split3};
    }
};

} // namespace ninfer::ops::detail
