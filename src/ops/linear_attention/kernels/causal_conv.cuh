#pragma once

#include "ops/common/numeric/math.h"
#include "ops/common/numeric/math.cuh"

#include <cuda_bf16.h>
#include <cmath>
#include <cstdint>

namespace ninfer::ops::detail {

__device__ __forceinline__ void causal_conv1d_acc_pair(__nv_bfloat162 w, __nv_bfloat162 x,
                                                       float& acc0, float& acc1) {
    acc0 += __low2float(w) * __low2float(x);
    acc1 += __high2float(w) * __high2float(x);
}

// One destination column resolved for a fixed row: the address of element (row,0) and the row
// count of the destination that owns it.
template <class Element>
struct CausalConvOutputColumn {
    Element* base;
    std::int32_t rows;

    __device__ __forceinline__ Element* at(std::int32_t t) const {
        return base + static_cast<std::int64_t>(t) * rows;
    }
};

// Contiguous [rows,T] output; the channel count is supplied by the calling Op.
template <class Element>
struct CausalConvContiguousOutput {
    Element* out;
    std::int32_t rows;

    __device__ __forceinline__ CausalConvOutputColumn<Element> column(std::int64_t row) const {
        return {out + row, rows};
    }
};

// Three contiguous destinations partitioned by row. Pair stores use halved row counts;
// every split boundary must therefore be even.
template <class Element, std::int32_t Rows0, std::int32_t Rows1, std::int32_t Rows2>
struct CausalConvSplitOutput3 {
    static_assert(Rows0 > 0 && Rows1 > 0 && Rows2 > 0);

    Element* __restrict__ out0;
    Element* __restrict__ out1;
    Element* __restrict__ out2;

    __device__ __forceinline__ CausalConvOutputColumn<Element> column(std::int64_t row) const {
        constexpr std::int32_t split1 = Rows0;
        constexpr std::int32_t split2 = Rows0 + Rows1;
        if (row < split1) { return {out0 + row, Rows0}; }
        if (row < split2) { return {out1 + (row - split1), Rows1}; }
        return {out2 + (row - split2), Rows2};
    }
};

struct SnapshotHistoryPublish {
    __nv_bfloat16* state_write;
    const std::int64_t* byte_offsets;
    const std::int32_t* snapshot_bases;
    std::int64_t snapshot_stride;
    std::int32_t channels;

    __device__ __forceinline__ void publish(std::int32_t token, std::int32_t batch,
                                            std::int32_t row, float s1, float s2, float p) const {
        const std::int64_t base =
            byte_offsets != nullptr
                ? byte_offsets[batch] / 2
                : static_cast<std::int64_t>(snapshot_bases[batch] + token) * snapshot_stride;
        state_write[base + row]                  = __float2bfloat16_rn(s1);
        state_write[base + channels + row]       = __float2bfloat16_rn(s2);
        state_write[base + 2LL * channels + row] = __float2bfloat16_rn(p);
    }
};

struct RecordColumnPublish {
    __nv_bfloat16* record;
    std::int32_t channels;
    std::int32_t width;

    __device__ __forceinline__ void publish(std::int32_t token, std::int32_t batch,
                                            std::int32_t row, float, float, float p) const {
        const std::int64_t column       = static_cast<std::int64_t>(batch) * width + token;
        record[column * channels + row] = __float2bfloat16_rn(p);
    }
};

struct NoHistoryPublish {
    __device__ __forceinline__ void publish(std::int32_t, std::int32_t, std::int32_t, float, float,
                                            float) const {}
};

__device__ __forceinline__ __nv_bfloat16 causal_conv4_silu(float w0, float w1, float w2, float w3,
                                                           float s0, float s1, float s2, float p) {
    float conv = fmaf(w0, s0, 0.0F);
    conv       = fmaf(w1, s1, conv);
    conv       = fmaf(w2, s2, conv);
    conv       = fmaf(w3, p, conv);
    return __float2bfloat16_rn(common::silu(conv));
}

} // namespace ninfer::ops::detail
