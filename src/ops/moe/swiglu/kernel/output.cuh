#pragma once
#include <cuda_bf16.h>
#include <cstdint>

namespace ninfer::ops::detail::swiglu_ffn {
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
} // namespace ninfer::ops::detail::swiglu_ffn
