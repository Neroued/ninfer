#pragma once
#include "ops/mhc/implementation.h"
#include <cuda_bf16.h>
#include "ops/common/numeric/convert.cuh"

namespace ninfer::ops::detail::mhc {
// All old streams are held before any output is stored, including exact in-place updates.
__device__ __forceinline__ void mix_streams(const float (&x)[4], float y, const float* q,
                                            const float* c, float (&out)[4]) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float value = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) value = fmaf(c[i * 4 + j], x[i], value);
        out[j] = fmaf(q[j], y, value);
    }
}

template <int Values>
__device__ __forceinline__ void mixed_vectors(Operands o, std::int64_t m, int d, const float* q,
                                              const float* c,
                                              common::Bf16Vector<Values> (&out)[4]) {
    common::Bf16Vector<Values> old[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
        old[i] = common::Bf16Vector<Values>::load(static_cast<const __nv_bfloat16*>(o.residual) +
                                                  m * kWidth + i * kHidden + d);
    const auto y = common::Bf16Vector<Values>::load(static_cast<const __nv_bfloat16*>(o.block) +
                                                    m * kHidden + d);
#pragma unroll
    for (int v = 0; v < Values; ++v) {
        float x[4], values[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) x[i] = __bfloat162float(old[i].word[v]);
        mix_streams(x, __bfloat162float(y.word[v]), q, c, values);
#pragma unroll
        for (int j = 0; j < 4; ++j) out[j].word[v] = __float2bfloat16_rn(values[j]);
    }
}

template <Form F>
__device__ __forceinline__ void residual_values(Operands o, std::int64_t m, int d,
                                                __nv_bfloat16 (&out)[4]) {
    const auto* input = static_cast<const __nv_bfloat16*>(o.residual);
    if constexpr (F == Form::ExpandPre) {
#pragma unroll
        for (int i = 0; i < 4; ++i) out[i] = input[m * kHidden + d];
    } else if constexpr (F == Form::Pre) {
#pragma unroll
        for (int i = 0; i < 4; ++i) out[i] = input[m * kWidth + i * kHidden + d];
    } else {
        float x[4], value[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) x[i] = __bfloat162float(input[m * kWidth + i * kHidden + d]);
        mix_streams(x,
                    __bfloat162float(static_cast<const __nv_bfloat16*>(o.block)[m * kHidden + d]),
                    o.post + m * 4, o.mix + m * 16, value);
#pragma unroll
        for (int i = 0; i < 4; ++i) out[i] = __float2bfloat16_rn(value[i]);
    }
    if constexpr (F == Form::ExpandPre || F == Form::PostPre || F == Form::Post) {
#pragma unroll
        for (int i = 0; i < 4; ++i)
            static_cast<__nv_bfloat16*>(o.updated)[m * kWidth + i * kHidden + d] = out[i];
    }
}
} // namespace ninfer::ops::detail::mhc
