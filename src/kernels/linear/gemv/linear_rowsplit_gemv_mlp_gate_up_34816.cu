#include "kernels/linear/gemv/linear_rowsplit_gemv_mlp_gate_up_34816.cuh"

#include "qus/core/device.h" // CUDA_CHECK

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>

namespace qus::kernels::detail {
namespace {

constexpr int kN = 34816;
constexpr int kK = 5120;
constexpr int kGroupK = 64;
constexpr int kGroups = kK / kGroupK;
constexpr int kBytesPerGroup = 32;
constexpr int kWarpsPerBlock = 8;
constexpr int kBlockThreads = kWarpsPerBlock * 32;

__device__ __forceinline__ int sign_extend_q4(int v) {
    return (v & 0x08) ? (v - 16) : v;
}

__global__ void linear_rowsplit_gemv_mlp_gate_up_34816_q4_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, __nv_bfloat16* __restrict__ out) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int row = static_cast<int>(blockIdx.x) * kWarpsPerBlock + warp;
    if (row >= kN) { return; }

    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(row) * kGroups * kBytesPerGroup;
    const std::uint8_t* scale_row = scales + static_cast<std::int64_t>(row) * kGroups * 2;
    const auto* x2 = reinterpret_cast<const __nv_bfloat162*>(x);

    float acc = 0.0f;
    for (int tile = 0; tile < kGroups; tile += 32) {
        const int tile_count = (kGroups - tile) < 32 ? (kGroups - tile) : 32;
        std::uint16_t lane_scale_bits = 0;
        if (lane < tile_count) {
            const std::uint8_t* sp = scale_row + (tile + lane) * 2;
            lane_scale_bits = static_cast<std::uint16_t>(sp[0]) |
                              static_cast<std::uint16_t>(static_cast<std::uint16_t>(sp[1]) << 8);
        }

#pragma unroll
        for (int tile_group = 0; tile_group < 32; ++tile_group) {
            if (tile_group >= tile_count) { break; }
            const int group = tile + tile_group;
            const auto scale_bits = static_cast<std::uint16_t>(
                __shfl_sync(0xffffffffu, lane_scale_bits, tile_group));
            const float scale = __half2float(__ushort_as_half(scale_bits));

            const std::uint8_t packed = code_row[group * kBytesPerGroup + lane];
            const int q0 = sign_extend_q4(packed & 0x0f);
            const int q1 = sign_extend_q4(packed >> 4);
            const int k0 = group * kGroupK + lane * 2;
            const float2 xv = __bfloat1622float2(x2[k0 >> 1]);
            acc = fmaf(static_cast<float>(q0) * scale, xv.x, acc);
            acc = fmaf(static_cast<float>(q1) * scale, xv.y, acc);
        }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0) { out[row] = __float2bfloat16(acc); }
}

} // namespace

void linear_rowsplit_gemv_mlp_gate_up_34816_q4_launch(const Tensor& x, const Weight& w,
                                                      Tensor& out, WorkspaceArena& ws,
                                                      cudaStream_t stream) {
    (void)ws;
    if (w.n != kN || w.k != kK || w.padded_shape[1] != kK) {
        throw std::invalid_argument("linear: MLP gate/up Q4 fused GEMV requires 34816x5120");
    }
    const int grid = (kN + kWarpsPerBlock - 1) / kWarpsPerBlock;
    linear_rowsplit_gemv_mlp_gate_up_34816_q4_kernel<<<grid, kBlockThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace qus::kernels::detail
