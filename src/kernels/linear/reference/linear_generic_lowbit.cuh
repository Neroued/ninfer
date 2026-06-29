#pragma once

#include "kernels/linear/codec/linear_codec.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace qus::kernels::detail {

template <class Codec>
__global__ void linear_generic_lowbit_gemv_kernel(const __nv_bfloat16* x,
                                                  const std::uint8_t* codes,
                                                  const std::uint8_t* scales,
                                                  __nv_bfloat16* out,
                                                  std::int32_t n, std::int32_t k,
                                                  std::int32_t padded_k) {
    const std::int32_t kg     = padded_k / Codec::kGroupK;
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t row64 = start; row64 < n; row64 += stride) {
        const std::int32_t row = static_cast<std::int32_t>(row64);
        float acc              = 0.0f;
        float wbuf[Codec::kGroupK];
        for (std::int32_t group = 0; group < kg; ++group) {
            Codec::load_group(codes, scales, row, group, kg, wbuf);
            for (int lane = 0; lane < Codec::kGroupK; ++lane) {
                const std::int32_t kk = group * Codec::kGroupK + lane;
                if (kk >= k) { break; }
                acc = fmaf(wbuf[lane], __bfloat162float(x[kk]), acc);
            }
        }
        out[row] = __float2bfloat16(acc);
    }
}

template <class Codec>
__global__ void linear_generic_lowbit_gemm_kernel(const __nv_bfloat16* x,
                                                  const std::uint8_t* codes,
                                                  const std::uint8_t* scales,
                                                  __nv_bfloat16* out,
                                                  std::int32_t n, std::int32_t k, std::int32_t t,
                                                  std::int32_t padded_k) {
    const std::int32_t kg       = padded_k / Codec::kGroupK;
    const std::int64_t elements = static_cast<std::int64_t>(n) * t;
    const std::int64_t start    = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride   = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < elements; i += stride) {
        const std::int32_t col     = static_cast<std::int32_t>(i / n);
        const std::int32_t row     = static_cast<std::int32_t>(i - static_cast<std::int64_t>(col) * n);
        const __nv_bfloat16* x_col = x + static_cast<std::int64_t>(col) * k;
        float acc                  = 0.0f;
        float wbuf[Codec::kGroupK];
        for (std::int32_t group = 0; group < kg; ++group) {
            Codec::load_group(codes, scales, row, group, kg, wbuf);
            for (int lane = 0; lane < Codec::kGroupK; ++lane) {
                const std::int32_t kk = group * Codec::kGroupK + lane;
                if (kk >= k) { break; }
                acc = fmaf(wbuf[lane], __bfloat162float(x_col[kk]), acc);
            }
        }
        out[i] = __float2bfloat16(acc);
    }
}

} // namespace qus::kernels::detail
