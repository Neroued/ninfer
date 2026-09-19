#pragma once

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Store epilogue for the K-split MMA when the output is not a packed [rows x cols] tensor.
//
// The fused input projections write two tensors with their own row strides: the GDN path writes qkv
// rows 4096..10239 with the qkv stride (10240) plus all of z, and the attention path writes q with
// the q stride plus k. The shared K-split epilogue assumes `out_ld == rows`, so those callers need
// this one instead: it carries its own strides, and it splits at a compile-time seam exactly like
// the row-split SIMT epilogues do.
//
// The kernel hands over the tile-relative column, so the live-column mask is applied here (the tile
// is rounded up to a multiple of 8 while the problem may have any column count).
template <bool SplitOutput = false, std::int32_t SplitRow = 0>
struct Q4KSplitStridedStore {
    __nv_bfloat16* out      = nullptr; // rows [0, SplitRow) when SplitOutput, else all rows
    std::int32_t out_ld     = 0;
    __nv_bfloat16* out_tail = nullptr; // rows [SplitRow, ...)
    std::int32_t tail_ld    = 0;
    std::int32_t columns    = 0; // live columns

    __device__ __forceinline__ void put(std::int32_t col, std::int32_t row, float value) const {
        if (col >= columns) { return; }
        const std::int64_t c = static_cast<std::int64_t>(col);
        if constexpr (SplitOutput) {
            if (row < SplitRow) {
                out[c * out_ld + row] = __float2bfloat16_rn(value);
            } else {
                out_tail[c * tail_ld + row - SplitRow] = __float2bfloat16_rn(value);
            }
        } else {
            out[c * out_ld + row] = __float2bfloat16_rn(value);
        }
    }

    template <int /*Capacity*/>
    __device__ __forceinline__ void store(std::int32_t row, std::int32_t col, float4 value) const {
        put(col, row, value.x);
        put(col, row + 8, value.z);
        put(col + 1, row, value.y);
        put(col + 1, row + 8, value.w);
    }
};

} // namespace ninfer::ops::detail
