#include "kernels/linear/gemv/linear_rowsplit_gemv_attn_in_7168.cuh"

#include "kernels/common/math.h"
#include "kernels/common/memory.cuh"
#include "kernels/common/warp.cuh"
#include "kernels/linear/gemv/linear_rowsplit_gemv_q5_core.cuh"
#include "ninfer/core/device.h" // CUDA_CHECK

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::kernels::detail {
namespace {

constexpr int kN              = 7168;
constexpr int kK              = 5120;
constexpr int kProjRows       = 6144;
constexpr int kGroupK         = 64;
constexpr int kGroups         = kK / kGroupK;
constexpr int kQ4WarpsPerRow  = 4;
constexpr int kQ4GroupsPerWarp = kGroups / kQ4WarpsPerRow;
constexpr int kQ4BlockThreads = kQ4WarpsPerRow * 32;
constexpr int kQ4ProjBlocks   = kProjRows / kQ4WarpsPerRow;
constexpr int kQ4BytesPerGroup = 32;

// Q4 attention-in GEMV: 6144 "proj" rows use one warp per row (8 warps/block), the
// remaining 1024 rows use a 4-way split-K block. Byte-load path retained as-is.
__global__ void linear_rowsplit_gemv_attn_in_7168_q4_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, __nv_bfloat16* __restrict__ out) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const auto* x2 = reinterpret_cast<const __nv_bfloat162*>(x);

    if (static_cast<int>(blockIdx.x) < kQ4ProjBlocks) {
        const int row = static_cast<int>(blockIdx.x) * kQ4WarpsPerRow + warp;
        const std::uint8_t* code_row =
            codes + static_cast<std::int64_t>(row) * kGroups * kQ4BytesPerGroup;
        const std::uint8_t* scale_row = scales + static_cast<std::int64_t>(row) * kGroups * 2;

        float acc = 0.0f;
#pragma unroll 1
        for (int tile = 0; tile < kGroups; tile += 32) {
            const int tile_count = (kGroups - tile) < 32 ? (kGroups - tile) : 32;
            std::uint16_t lane_scale_bits = 0;
            if (lane < tile_count) {
                lane_scale_bits = load_vec<std::uint16_t>(scale_row + (tile + lane) * 2);
            }

#pragma unroll
            for (int tile_group = 0; tile_group < 32; ++tile_group) {
                if (tile_group >= tile_count) { break; }
                const int group = tile + tile_group;
                const auto scale_bits = static_cast<std::uint16_t>(
                    __shfl_sync(0xffffffffu, lane_scale_bits, tile_group));
                const float scale = __half2float(__ushort_as_half(scale_bits));

                const std::uint8_t packed = code_row[group * kQ4BytesPerGroup + lane];
                const int q0 = sign_extend<4>(packed & 0x0f);
                const int q1 = sign_extend<4>(packed >> 4);
                const int k0 = group * kGroupK + lane * 2;
                const float2 xv = __bfloat1622float2(x2[k0 >> 1]);
                acc = fmaf(static_cast<float>(q0) * scale, xv.x, acc);
                acc = fmaf(static_cast<float>(q1) * scale, xv.y, acc);
            }
        }

        acc = warp_reduce_sum(acc);
        if (lane == 0) { out[row] = __float2bfloat16(acc); }
        return;
    }

    const int row = kProjRows + (static_cast<int>(blockIdx.x) - kQ4ProjBlocks);
    if (row >= kN) { return; }

    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(row) * kGroups * kQ4BytesPerGroup;
    const std::uint8_t* scale_row = scales + static_cast<std::int64_t>(row) * kGroups * 2;
    __shared__ float partials[kQ4WarpsPerRow];
    const int group_begin = warp * kQ4GroupsPerWarp;
    float acc = 0.0f;

    std::uint16_t lane_scale_bits = 0;
    if (lane < kQ4GroupsPerWarp) {
        lane_scale_bits = load_vec<std::uint16_t>(scale_row + (group_begin + lane) * 2);
    }

#pragma unroll
    for (int tile_group = 0; tile_group < kQ4GroupsPerWarp; ++tile_group) {
        const int group = group_begin + tile_group;
        const auto scale_bits =
            static_cast<std::uint16_t>(__shfl_sync(0xffffffffu, lane_scale_bits, tile_group));
        const float scale = __half2float(__ushort_as_half(scale_bits));

        const std::uint8_t packed = code_row[group * kQ4BytesPerGroup + lane];
        const int q0 = sign_extend<4>(packed & 0x0f);
        const int q1 = sign_extend<4>(packed >> 4);
        const int k0 = group * kGroupK + lane * 2;
        const float2 xv = __bfloat1622float2(x2[k0 >> 1]);
        acc = fmaf(static_cast<float>(q0) * scale, xv.x, acc);
        acc = fmaf(static_cast<float>(q1) * scale, xv.y, acc);
    }

    acc = warp_reduce_sum(acc);
    if (lane == 0) { partials[warp] = acc; }
    __syncthreads();

    if (threadIdx.x == 0) {
        out[row] = __float2bfloat16(partials[0] + partials[1] + partials[2] + partials[3]);
    }
}

} // namespace

void linear_rowsplit_gemv_attn_in_7168_q4_launch(const Tensor& x, const Weight& w, Tensor& out,
                                                 WorkspaceArena& ws, cudaStream_t stream) {
    (void)ws;
    if (w.n != kN || w.k != kK || w.padded_shape[1] != kK) {
        throw std::invalid_argument("linear: Attention q/k Q4 fused GEMV requires 7168x5120");
    }
    constexpr int kBlocks = kQ4ProjBlocks + (kN - kProjRows);
    linear_rowsplit_gemv_attn_in_7168_q4_kernel<<<kBlocks, kQ4BlockThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

// Q5 attention-in: every one of the 7168 rows is identical work (K=5120, 5 tiles),
// so a single uniform one-row-per-warp pass over the shared cp.async-staged core.
void linear_rowsplit_gemv_attn_in_7168_q5_launch(const Tensor& x, const Weight& w, Tensor& out,
                                                 WorkspaceArena& ws, cudaStream_t stream) {
    (void)ws;
    if (w.n != kN || w.k != kK || w.padded_shape[1] != kK) {
        throw std::invalid_argument("linear: Attention gate/v Q5 fused GEMV requires 7168x5120");
    }

    launch_q5_rowsplit_gemv<kN, kK, /*kRowsPerBlock=*/16, /*kStages=*/2, /*kStageX=*/true>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.qhigh), static_cast<const std::uint8_t*>(w.scales),
        static_cast<__nv_bfloat16*>(out.data), stream);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::kernels::detail
