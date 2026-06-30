#include "kernels/linear/gemv/linear_rowsplit_gemv_out_6144.cuh"

#include "qus/core/device.h" // CUDA_CHECK

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>

namespace qus::kernels::detail {
namespace {

constexpr int kN             = 5120;
constexpr int kK             = 6144;
constexpr int kGroupK        = 64;
constexpr int kGroups        = kK / kGroupK;
constexpr int kBytesPerGroup = 40;
constexpr int kWarpsPerRow   = 8;
constexpr int kGroupsPerWarp = kGroups / kWarpsPerRow;
constexpr int kBlockThreads  = kWarpsPerRow * 32;

__device__ __forceinline__ int sign_extend_q5(int v) { return (v & 0x10) ? (v - 32) : v; }

__device__ __forceinline__ std::uint16_t load_scale_bits(const std::uint8_t* scale_row, int group) {
    const std::uint8_t* sp = scale_row + group * 2;
    return static_cast<std::uint16_t>(sp[0]) |
           static_cast<std::uint16_t>(static_cast<std::uint16_t>(sp[1]) << 8);
}

__device__ __forceinline__ std::uint32_t
load_q5_pair_bits(const std::uint8_t* __restrict__ code_group, int lane) {
    const int bitpos       = lane * 10;
    const int wi           = bitpos >> 5;
    const int shift        = bitpos & 31;
    const auto* words      = reinterpret_cast<const std::uint32_t*>(code_group);
    const std::uint32_t w0 = words[wi];
    const std::uint32_t w1 = wi < 9 ? words[wi + 1] : 0u;
    return __funnelshift_r(w0, w1, shift);
}

__device__ __forceinline__ float accumulate_group(const __nv_bfloat162* __restrict__ x2,
                                                  const std::uint8_t* __restrict__ code_group,
                                                  std::uint16_t scale_bits, int lane, int group,
                                                  float acc) {
    const float scale        = __half2float(__ushort_as_half(scale_bits));
    const std::uint32_t bits = load_q5_pair_bits(code_group, lane);

    const int q0    = sign_extend_q5(static_cast<int>(bits & 0x1fu));
    const int q1    = sign_extend_q5(static_cast<int>((bits >> 5) & 0x1fu));
    const int k0    = group * kGroupK + lane * 2;
    const float2 xv = __bfloat1622float2(x2[k0 >> 1]);
    acc             = fmaf(static_cast<float>(q0) * scale, xv.x, acc);
    acc             = fmaf(static_cast<float>(q1) * scale, xv.y, acc);
    return acc;
}

__device__ __forceinline__ float warp_reduce_sum(float acc) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    return acc;
}

__global__ void linear_rowsplit_gemv_out_6144_q5_kernel(const __nv_bfloat16* __restrict__ x,
                                                        const std::uint8_t* __restrict__ codes,
                                                        const std::uint8_t* __restrict__ scales,
                                                        __nv_bfloat16* __restrict__ out) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int row  = static_cast<int>(blockIdx.x);
    if (row >= kN) { return; }

    __shared__ float partials[kWarpsPerRow];
    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(row) * kGroups * kBytesPerGroup;
    const std::uint8_t* scale_row = scales + static_cast<std::int64_t>(row) * kGroups * 2;
    const auto* x2                = reinterpret_cast<const __nv_bfloat162*>(x);
    const int group_begin         = warp * kGroupsPerWarp;
    const int group_end           = group_begin + kGroupsPerWarp;

    float acc = 0.0f;
#pragma unroll 1
    for (int group = group_begin; group < group_end; group += 3) {
        const std::uint8_t* code_group0 = code_row + group * kBytesPerGroup;
        const std::uint8_t* code_group1 = code_group0 + kBytesPerGroup;
        const std::uint8_t* code_group2 = code_group1 + kBytesPerGroup;

        std::uint16_t scale0_bits = 0;
        std::uint16_t scale1_bits = 0;
        std::uint16_t scale2_bits = 0;
        if (lane == 0) {
            scale0_bits = load_scale_bits(scale_row, group);
            scale1_bits = load_scale_bits(scale_row, group + 1);
            scale2_bits = load_scale_bits(scale_row, group + 2);
        }
        scale0_bits = static_cast<std::uint16_t>(__shfl_sync(0xffffffffu, scale0_bits, 0));
        scale1_bits = static_cast<std::uint16_t>(__shfl_sync(0xffffffffu, scale1_bits, 0));
        scale2_bits = static_cast<std::uint16_t>(__shfl_sync(0xffffffffu, scale2_bits, 0));

        acc = accumulate_group(x2, code_group0, scale0_bits, lane, group, acc);
        acc = accumulate_group(x2, code_group1, scale1_bits, lane, group + 1, acc);
        acc = accumulate_group(x2, code_group2, scale2_bits, lane, group + 2, acc);
    }

    acc = warp_reduce_sum(acc);
    if (lane == 0) { partials[warp] = acc; }
    __syncthreads();

    if (threadIdx.x == 0) {
        float row_acc = 0.0f;
#pragma unroll
        for (int i = 0; i < kWarpsPerRow; ++i) {
            row_acc += partials[i];
        }
        out[row] = __float2bfloat16(row_acc);
    }
}

} // namespace

void linear_rowsplit_gemv_out_6144_q5_launch(const Tensor& x, const Weight& w, Tensor& out,
                                             WorkspaceArena& ws, cudaStream_t stream) {
    (void)ws;
    if (w.n != kN || w.k != kK || w.padded_shape[1] != kK) {
        throw std::invalid_argument("linear: Out6144 Q5 tuned GEMV requires 5120x6144");
    }
    linear_rowsplit_gemv_out_6144_q5_kernel<<<kN, kBlockThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace qus::kernels::detail
