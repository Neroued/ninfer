#pragma once
#include <cuda_bf16.h>
#include <cstdint>
#include "ops/moe/routed_swiglu/cuda/contribution.h"

namespace ninfer::ops::detail::swiglu_ffn {
template <int Rows>
struct SwiGluRoutedOutput {
    __nv_bfloat16* destination;
    RoutedSwiGLUContribution routed;

    __device__ __forceinline__ void store(int row, int token, float value) const {
        const auto index = std::int64_t(token) * Rows + row;
        if (!routed.valid_tokens || routed.valid_tokens[token]) value += routed.values[index];
        destination[index] = __float2bfloat16_rn(value);
    }
};

struct SwiGluFloatOutput {
    using Fp32 = void;
    float* data;
    int rows;

    __device__ __forceinline__ void store(int row, int token, float value) const {
        data[std::int64_t(token) * rows + row] = value;
    }
};

// Private compensated operands for a whole BF16 FFN; no public BF16 product seam.
struct SwiGluCompensatedOutput {
    __nv_bfloat16* planes;
    int rows, tokens;

    __device__ __forceinline__ void store(int row, int token, float value) const {
        const auto index = std::int64_t(token) * rows + row;
        const auto hi    = __float2bfloat16_rn(value);
        planes[index]    = hi;
        planes[std::int64_t(rows) * tokens + index] =
            __float2bfloat16_rn(value - __bfloat162float(hi));
    }
};

struct SwiGluPaddedCompensatedOutput {
    __nv_bfloat16* planes;
    int rows, tokens, plane_tokens;

    __device__ __forceinline__ void store(int row, int token, float value) const {
        SwiGluCompensatedOutput{planes, rows, plane_tokens}.store(row, token, value);
        // The last live column owns initialization of this row's padding in both planes.
        // The subsequent down contraction reads a full TMA box; no unwritten cell is consumed.
        if (token == tokens - 1)
            for (int column = tokens; column < plane_tokens; ++column) {
                planes[std::int64_t(column) * rows + row]                = __float2bfloat16_rn(0);
                planes[std::int64_t(plane_tokens + column) * rows + row] = __float2bfloat16_rn(0);
            }
    }
};
} // namespace ninfer::ops::detail::swiglu_ffn
