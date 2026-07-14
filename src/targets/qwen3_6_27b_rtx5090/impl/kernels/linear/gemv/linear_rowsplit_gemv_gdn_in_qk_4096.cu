#include "kernels/linear/gemv/linear_rowsplit_gemv_gdn_in_qk_4096.cuh"

#include "kernels/common/math.h"
#include "kernels/common/memory.cuh"
#include "kernels/common/warp.cuh"
#include "core/device.h" // CUDA_CHECK

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::kernels::detail {
namespace {

constexpr int kN = 4096;
constexpr int kK = 5120;
constexpr int kGroupK = 64;
constexpr int kGroups = kK / kGroupK;
constexpr int kBytesPerGroup = 32;
constexpr int kVecBytes = 16;
constexpr int kWarpsPerRow = 8;
constexpr int kGroupsPerWarp = kGroups / kWarpsPerRow;
constexpr int kGroupsPerWarpTile = 16;
constexpr int kVecsPerWarpTile = kGroupsPerWarpTile * kBytesPerGroup / kVecBytes;
constexpr int kBlockThreads = kWarpsPerRow * 32;
static_assert(kBytesPerGroup == 2 * kVecBytes);
static_assert(kVecsPerWarpTile == 32);

__global__ void linear_rowsplit_gemv_gdn_in_qk_4096_q4_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, __nv_bfloat16* __restrict__ out) {
    __shared__ uint4 code_tile[kWarpsPerRow][kVecsPerWarpTile];
    __shared__ float partials[kWarpsPerRow];

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int row = static_cast<int>(blockIdx.x);
    if (row >= kN) { return; }

    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(row) * kGroups * kBytesPerGroup;
    const std::uint8_t* scale_row = scales + static_cast<std::int64_t>(row) * kGroups * 2;
    const auto* x2 = reinterpret_cast<const __nv_bfloat162*>(x);

    float acc = 0.0f;
    const int group_begin = warp * kGroupsPerWarp;

    for (int tile = 0; tile < kGroupsPerWarp; tile += kGroupsPerWarpTile) {
        const int tile_count =
            (kGroupsPerWarp - tile) < kGroupsPerWarpTile ? (kGroupsPerWarp - tile)
                                                         : kGroupsPerWarpTile;
        const int tile_vecs = tile_count * kBytesPerGroup / kVecBytes;
        const auto* global_vecs =
            reinterpret_cast<const uint4*>(code_row + (group_begin + tile) * kBytesPerGroup);
        if (lane < tile_vecs) { code_tile[warp][lane] = global_vecs[lane]; }
        __syncwarp();

        std::uint16_t lane_scale_bits = 0;
        if (lane < tile_count) {
            lane_scale_bits =
                load_vec<std::uint16_t>(scale_row + (group_begin + tile + lane) * 2);
        }
        const auto* tile_codes = reinterpret_cast<const std::uint8_t*>(code_tile[warp]);
#pragma unroll
        for (int tile_group = 0; tile_group < kGroupsPerWarpTile; ++tile_group) {
            if (tile_group >= tile_count) { break; }
            const auto scale_bits = static_cast<std::uint16_t>(
                __shfl_sync(0xffffffffu, lane_scale_bits, tile_group));
            const float scale = __half2float(__ushort_as_half(scale_bits));

            const int packed = static_cast<int>(tile_codes[tile_group * kBytesPerGroup + lane]);
            const int q0 = sign_extend<4>(packed & 0x0f);
            const int q1 = sign_extend<4>(packed >> 4);
            const int k0 = (group_begin + tile + tile_group) * kGroupK + lane * 2;
            const float2 xv = __bfloat1622float2(x2[k0 >> 1]);
            acc = fmaf(static_cast<float>(q0) * scale, xv.x, acc);
            acc = fmaf(static_cast<float>(q1) * scale, xv.y, acc);
        }
        __syncwarp();
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

} // namespace ninfer::kernels::detail
