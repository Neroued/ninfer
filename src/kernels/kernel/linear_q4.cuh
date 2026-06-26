#pragma once

// qus::kernels - Q4G64 TILE_N64_K64 linear kernels. Correctness-baseline
// GEMV/GEMM with direct q5090 decode, fp32 accumulation, and BF16 output.
//
// PERF: decode GEMV is the linear roofline op; this scalar decode path is the
// correctness baseline and should be replaced by a coalesced/vectorized kernel
// in a later tuning task.

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace qus::kernels {

inline constexpr std::int32_t kLinearQ4Group = 64;
inline constexpr std::int32_t kLinearQ4Bpr   = 32;
inline constexpr std::int32_t kLinearQ4Tilew = 64 * 2 + 64 * kLinearQ4Bpr;

__device__ __forceinline__ int unpack_q4_code(const std::uint8_t* packed, int lane) {
    const std::uint8_t byte = packed[lane >> 1];
    const int u             = (lane & 1) ? (byte >> 4) : (byte & 0x0f);
    return (u & 0x08) ? (u - 16) : u;
}

__device__ __forceinline__ float load_q4_weight(const std::uint8_t* payload, std::int32_t row,
                                                std::int32_t group, std::int32_t lane,
                                                std::int32_t kg) {
    const std::int32_t tile        = row / 64;
    const std::int32_t row_in_tile = row - tile * 64;
    const std::int64_t off =
        (static_cast<std::int64_t>(tile) * kg + group) * static_cast<std::int64_t>(kLinearQ4Tilew);
    const std::uint16_t scale_bits =
        static_cast<std::uint16_t>(payload[off + row_in_tile * 2]) |
        static_cast<std::uint16_t>(static_cast<std::uint16_t>(payload[off + row_in_tile * 2 + 1])
                                   << 8);
    const float scale = __half2float(__ushort_as_half(scale_bits));
    const std::uint8_t* packed =
        payload + off + 64 * 2 + static_cast<std::int64_t>(row_in_tile) * kLinearQ4Bpr;
    return static_cast<float>(unpack_q4_code(packed, lane)) * scale;
}

__global__ void linear_q4_gemv_kernel(const __nv_bfloat16* x, const std::uint8_t* payload,
                                      __nv_bfloat16* out, std::int32_t n, std::int32_t k,
                                      std::int32_t padded_k) {
    const std::int32_t kg     = padded_k / kLinearQ4Group;
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;

    for (std::int64_t row64 = start; row64 < n; row64 += stride) {
        const std::int32_t row = static_cast<std::int32_t>(row64);
        float acc              = 0.0f;
        for (std::int32_t group = 0; group < kg; ++group) {
            for (std::int32_t lane = 0; lane < kLinearQ4Group; ++lane) {
                const std::int32_t kk = group * kLinearQ4Group + lane;
                if (kk >= k) { break; }
                const float wv = load_q4_weight(payload, row, group, lane, kg);
                const float xv = __bfloat162float(x[kk]);
                acc            = fmaf(wv, xv, acc);
            }
        }
        out[row] = __float2bfloat16(acc);
    }
}

__global__ void linear_q4_gemm_kernel(const __nv_bfloat16* x, const std::uint8_t* payload,
                                      __nv_bfloat16* out, std::int32_t n, std::int32_t k,
                                      std::int32_t t, std::int32_t padded_k) {
    const std::int32_t kg       = padded_k / kLinearQ4Group;
    const std::int64_t elements = static_cast<std::int64_t>(n) * t;
    const std::int64_t start    = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride   = static_cast<std::int64_t>(gridDim.x) * blockDim.x;

    for (std::int64_t i = start; i < elements; i += stride) {
        const std::int32_t col = static_cast<std::int32_t>(i / n);
        const std::int32_t row = static_cast<std::int32_t>(i - static_cast<std::int64_t>(col) * n);
        float acc              = 0.0f;
        const __nv_bfloat16* x_col = x + static_cast<std::int64_t>(col) * k;
        for (std::int32_t group = 0; group < kg; ++group) {
            for (std::int32_t lane = 0; lane < kLinearQ4Group; ++lane) {
                const std::int32_t kk = group * kLinearQ4Group + lane;
                if (kk >= k) { break; }
                const float wv = load_q4_weight(payload, row, group, lane, kg);
                const float xv = __bfloat162float(x_col[kk]);
                acc            = fmaf(wv, xv, acc);
            }
        }
        out[i] = __float2bfloat16(acc);
    }
}

} // namespace qus::kernels
