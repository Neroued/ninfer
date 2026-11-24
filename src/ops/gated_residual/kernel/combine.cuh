#pragma once
#include "ops/gated_residual/operands.h"
#include "ops/common/numeric/convert.cuh"

namespace ninfer::ops::detail::hc {
// BF16 conversion of FP32 FMA can double-round at a BF16 midpoint. A warp-level
// branch skips the entire FP64 correction body when no lane needs it; a simple
// scalar predicate otherwise issues predicated-off FP64 operations on every value.
template <Form Kind>
__device__ __forceinline__ __nv_bfloat162 update(__nv_bfloat162 h, __nv_bfloat162 y, float j) {
    const float2 a = __bfloat1622float2(h), b = __bfloat1622float2(y);
    if constexpr (Kind == Form::AddNorm)
        return __floats2bfloat162_rn(__fadd_rn(a.x, b.x), __fadd_rn(a.y, b.y));
    else {
        const float2 value{__fmaf_rn(j, b.x, a.x), __fmaf_rn(j, b.y, a.y)};
        __nv_bfloat16 x = __float2bfloat16_rn(value.x), y_out = __float2bfloat16_rn(value.y);
        const bool mx = (__float_as_uint(value.x) & 0xffffu) == 0x8000u;
        const bool my = (__float_as_uint(value.y) & 0xffffu) == 0x8000u;
        // A power-of-two coefficient preserves the BF16 significand. Addition near
        // a BF16 midpoint is then exact in FP32, so no double-rounding repair is needed.
        const bool needs_repair = (__float_as_uint(j) & 0x7fffffu) != 0;
        if (__any_sync(__activemask(), (mx || my) && needs_repair)) {
            if (mx)
                x = __double2bfloat16(__dadd_rn(double(a.x), __dmul_rn(double(j), double(b.x))));
            if (my)
                y_out =
                    __double2bfloat16(__dadd_rn(double(a.y), __dmul_rn(double(j), double(b.y))));
        }
        return __halves2bfloat162(x, y_out);
    }
}

template <Form Kind, int Block>
__launch_bounds__(Block) __global__
    void combine_vector_kernel(Operands o, std::int64_t first, std::int64_t vectors) {
    const std::int64_t i = first + std::int64_t(blockIdx.x) * Block + threadIdx.x;
    if (i >= first + vectors) return;
    constexpr int per_row = kWidth / 8;
    const auto row = i / per_row, m = row / kStreams;

    const auto h_bits = static_cast<const uint4*>(o.input)[i];
    const auto y_bits = static_cast<const uint4*>(o.block)[m * per_row + i % per_row];
    const auto h      = common::load_vec<common::Bf16x8Pack>(&h_bits);
    const auto y      = common::load_vec<common::Bf16x8Pack>(&y_bits);
    common::Bf16x8Pack result;
    const float j = Kind == Form::AddNorm ? 1.0f : o.injection[row];
#pragma unroll
    for (int p = 0; p < 4; ++p) result.pair[p] = update<Kind>(h.pair[p], y.pair[p], j);
    static_cast<uint4*>(o.residual)[i] = common::load_vec<uint4>(&result);
}
} // namespace ninfer::ops::detail::hc
