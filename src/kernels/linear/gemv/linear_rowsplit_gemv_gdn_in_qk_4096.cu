#include "kernels/linear/gemv/linear_rowsplit_gemv_gdn_in_qk_4096.cuh"

#include "qus/core/device.h" // CUDA_CHECK

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>

namespace qus::kernels::detail {
namespace {

constexpr int kN = 4096;
constexpr int kK = 5120;
constexpr int kGroupK = 64;
constexpr int kGroups = kK / kGroupK;
constexpr int kBytesPerGroup = 32;
constexpr int kWarpsPerRow = 4;
constexpr int kGroupsPerWarp = kGroups / kWarpsPerRow;
constexpr int kBlockThreads = kWarpsPerRow * 32;

__device__ __forceinline__ int sign_extend_q4(int v) {
    return (v & 0x08) ? (v - 16) : v;
}

__device__ __forceinline__ std::uint16_t load_scale_bits(const std::uint8_t* scale_row,
                                                         int group) {
    const std::uint8_t* sp = scale_row + group * 2;
    return static_cast<std::uint16_t>(sp[0]) |
           static_cast<std::uint16_t>(static_cast<std::uint16_t>(sp[1]) << 8);
}

__device__ __forceinline__ float warp_reduce_sum(float acc) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    return acc;
}

__global__ void linear_rowsplit_gemv_gdn_in_qk_4096_q4_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, __nv_bfloat16* __restrict__ out) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int row = static_cast<int>(blockIdx.x);
    if (row >= kN) { return; }

    __shared__ float partials[kWarpsPerRow];
    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(row) * kGroups * kBytesPerGroup;
    const std::uint8_t* scale_row = scales + static_cast<std::int64_t>(row) * kGroups * 2;
    const auto* x2 = reinterpret_cast<const __nv_bfloat162*>(x);

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const int group_begin = warp * kGroupsPerWarp;
    std::uint16_t lane_scale_bits = 0;
    if (lane < kGroupsPerWarp) {
        lane_scale_bits = load_scale_bits(scale_row, group_begin + lane);
    }

#pragma unroll
    for (int tile_group = 0; tile_group < kGroupsPerWarp; tile_group += 2) {
        const int group0 = group_begin + tile_group;
        const auto scale_bits0 =
            static_cast<std::uint16_t>(__shfl_sync(0xffffffffu, lane_scale_bits, tile_group));
        const float scale0 = __half2float(__ushort_as_half(scale_bits0));

        const std::uint8_t packed0 = code_row[group0 * kBytesPerGroup + lane];
        const int q00 = sign_extend_q4(packed0 & 0x0f);
        const int q01 = sign_extend_q4(packed0 >> 4);
        const int k0 = group0 * kGroupK + lane * 2;
        const float2 xv0 = __bfloat1622float2(x2[k0 >> 1]);
        acc0 = fmaf(static_cast<float>(q00) * scale0, xv0.x, acc0);
        acc0 = fmaf(static_cast<float>(q01) * scale0, xv0.y, acc0);

        const int group1 = group0 + 1;
        const auto scale_bits1 =
            static_cast<std::uint16_t>(__shfl_sync(0xffffffffu, lane_scale_bits, tile_group + 1));
        const float scale1 = __half2float(__ushort_as_half(scale_bits1));

        const std::uint8_t packed1 = code_row[group1 * kBytesPerGroup + lane];
        const int q10 = sign_extend_q4(packed1 & 0x0f);
        const int q11 = sign_extend_q4(packed1 >> 4);
        const int k1 = group1 * kGroupK + lane * 2;
        const float2 xv1 = __bfloat1622float2(x2[k1 >> 1]);
        acc1 = fmaf(static_cast<float>(q10) * scale1, xv1.x, acc1);
        acc1 = fmaf(static_cast<float>(q11) * scale1, xv1.y, acc1);
    }

    float acc = acc0 + acc1;
    acc = warp_reduce_sum(acc);
    if (lane == 0) { partials[warp] = acc; }
    __syncthreads();

    if (threadIdx.x == 0) {
        out[row] = __float2bfloat16(partials[0] + partials[1] + partials[2] + partials[3]);
    }
}

} // namespace

void linear_rowsplit_gemv_gdn_in_qk_4096_q4_launch(const Tensor& x, const Weight& w, Tensor& out,
                                                   WorkspaceArena& ws, cudaStream_t stream) {
    (void)ws;
    if (w.n != kN || w.k != kK || w.padded_shape[1] != kK) {
        throw std::invalid_argument("linear: GDN in q/k Q4 fused GEMV requires 4096x5120");
    }
    linear_rowsplit_gemv_gdn_in_qk_4096_q4_kernel<<<kN, kBlockThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace qus::kernels::detail
