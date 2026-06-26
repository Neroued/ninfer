#pragma once

// qus::kernels - partial NeoX RoPE kernel over q and k, in place.
// One block handles one token. It computes the token's rotary coefficients once
// per pair, then reuses them across q and k heads. The final q/k values are
// copied back in vector chunks so the kernel's HBM traffic matches the in-place
// q/k read+write contract used by the benchmark.

#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>

namespace qus::kernels {

inline constexpr int kRopeHeadDim = 256;

__device__ __forceinline__ void rope_apply_one(__nv_bfloat16* data, std::int64_t base, int pair,
                                               int half, float c, float s) {
    const std::int64_t idx1 = base + static_cast<std::int64_t>(pair);
    const std::int64_t idx2 = base + static_cast<std::int64_t>(pair + half);
    const float x1 = __bfloat162float(data[idx1]);
    const float x2 = __bfloat162float(data[idx2]);
    data[idx1] = __float2bfloat16_rn(x1 * c - x2 * s);
    data[idx2] = __float2bfloat16_rn(x2 * c + x1 * s);
}

__device__ __forceinline__ void rope_copy_vec8(__nv_bfloat16* data, std::int64_t idx) {
    void* ptr = data + idx;
    unsigned int x, y, z, w;
    asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
                 : "l"(ptr)
                 : "memory");
    asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};"
                 :
                 : "l"(ptr), "r"(x), "r"(y), "r"(z), "r"(w)
                 : "memory");
}

__global__ void rope_pair_kernel(const std::int32_t* positions, __nv_bfloat16* q,
                                 __nv_bfloat16* k, int rotary_dim, float theta,
                                 std::int32_t q_heads, std::int32_t k_heads, std::int32_t T,
                                 std::int64_t total_pairs) {
    const std::int64_t start =
        blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    const std::int64_t half = static_cast<std::int64_t>(rotary_dim / 2);
    const std::int64_t q_pairs = static_cast<std::int64_t>(q_heads) *
                                 static_cast<std::int64_t>(T) * half;

    for (std::int64_t linear = start; linear < total_pairs; linear += stride) {
        const bool is_q = linear < q_pairs;
        const std::int64_t local = is_q ? linear : linear - q_pairs;
        const std::int32_t heads = is_q ? q_heads : k_heads;
        const int pair = static_cast<int>(local % half);
        const std::int64_t row = local / half;
        const std::int32_t h = static_cast<std::int32_t>(row % heads);
        const std::int32_t t = static_cast<std::int32_t>(row / heads);

        const float freq =
            powf(theta, -2.0f * static_cast<float>(pair) / static_cast<float>(rotary_dim));
        const float angle = static_cast<float>(positions[t]) * freq;
        const float c = cosf(angle);
        const float s = sinf(angle);
        const std::int64_t base =
            (static_cast<std::int64_t>(t) * heads + h) * static_cast<std::int64_t>(kRopeHeadDim);
        rope_apply_one(is_q ? q : k, base, pair, static_cast<int>(half), c, s);
    }
}

__launch_bounds__(256) __global__ void
rope_kernel(const std::int32_t* positions, __nv_bfloat16* q, __nv_bfloat16* k, int rotary_dim,
            float theta, std::int32_t q_heads, std::int32_t k_heads, std::int32_t T,
            std::int64_t /*total_pairs*/) {
    const std::int32_t t = static_cast<std::int32_t>(blockIdx.x);
    if (t >= T) { return; }

    const int half = rotary_dim / 2;
    __shared__ float cos_cache[kRopeHeadDim / 2];
    __shared__ float sin_cache[kRopeHeadDim / 2];

    if (threadIdx.x < static_cast<unsigned>(half)) {
        const int pair = static_cast<int>(threadIdx.x);
        const float freq =
            powf(theta, -2.0f * static_cast<float>(pair) / static_cast<float>(rotary_dim));
        const float angle = static_cast<float>(positions[t]) * freq;
        cos_cache[pair] = cosf(angle);
        sin_cache[pair] = sinf(angle);
    }
    __syncthreads();

    const int q_pairs = q_heads * half;
    const int total_pairs = q_pairs + k_heads * half;
    for (int local = static_cast<int>(threadIdx.x); local < total_pairs; local += blockDim.x) {
        const bool is_q = local < q_pairs;
        const int tensor_local = is_q ? local : local - q_pairs;
        const std::int32_t heads = is_q ? q_heads : k_heads;
        const int pair = tensor_local % half;
        const std::int32_t h = static_cast<std::int32_t>(tensor_local / half);
        const std::int64_t base =
            (static_cast<std::int64_t>(t) * heads + h) * static_cast<std::int64_t>(kRopeHeadDim);
        rope_apply_one(is_q ? q : k, base, pair, half, cos_cache[pair], sin_cache[pair]);
    }
    __syncthreads();

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int warps_per_block = static_cast<int>(blockDim.x) >> 5;
    const int total_heads = q_heads + k_heads;
    for (int head = warp; head < total_heads; head += warps_per_block) {
        const bool is_q = head < q_heads;
        const std::int32_t heads = is_q ? q_heads : k_heads;
        const std::int32_t h =
            is_q ? static_cast<std::int32_t>(head) : static_cast<std::int32_t>(head - q_heads);
        const std::int64_t base =
            (static_cast<std::int64_t>(t) * heads + h) * static_cast<std::int64_t>(kRopeHeadDim);
        rope_copy_vec8(is_q ? q : k, base + static_cast<std::int64_t>(lane) * 8);
    }
}

} // namespace qus::kernels
