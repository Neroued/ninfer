#include "kernels/linear/gemv/linear_rowsplit_gemv_lm_head.cuh"

#include "qus/core/device.h" // CUDA_CHECK

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>

namespace qus::kernels::detail {
namespace {

constexpr int kN = 248320;
constexpr int kK = 5120;
constexpr int kGroupK = 64;
constexpr int kGroups = kK / kGroupK;
constexpr int kNibbleBytesPerGroup = 32;
constexpr int kHighBytesPerGroup = 16;
constexpr int kVecBytes = 16;
constexpr int kGroupsPerWarpTile = 16;
constexpr int kNibbleVecsPerWarpTile = kGroupsPerWarpTile * kNibbleBytesPerGroup / kVecBytes;
constexpr int kHighVecsPerWarpTile = kGroupsPerWarpTile * kHighBytesPerGroup / kVecBytes;
constexpr int kWarpsPerBlock = 4;
constexpr int kBlockThreads = kWarpsPerBlock * 32;
static_assert(kGroups % kGroupsPerWarpTile == 0);
static_assert(kNibbleVecsPerWarpTile == 32);
static_assert(kHighVecsPerWarpTile == 16);

__device__ __forceinline__ int sign_extend_q6(int v) {
    return (v & 0x20) ? (v - 64) : v;
}

__device__ __forceinline__ std::uint16_t load_scale_bits(const std::uint8_t* scale_row,
                                                         int group) {
    const std::uint8_t* sp = scale_row + group * 2;
    return static_cast<std::uint16_t>(sp[0]) |
           static_cast<std::uint16_t>(static_cast<std::uint16_t>(sp[1]) << 8);
}

__device__ __forceinline__ float accumulate_group(const __nv_bfloat162* __restrict__ x2,
                                                  const std::uint8_t* __restrict__ nibble_group,
                                                  const std::uint8_t* __restrict__ high_group,
                                                  std::uint16_t scale_bits,
                                                  int lane, int group, float acc) {
    const float scale = __half2float(__ushort_as_half(scale_bits));

    const std::uint8_t low = nibble_group[lane];
    const std::uint8_t high = high_group[lane >> 1] >> ((lane & 1) * 4);

    const int q0 = sign_extend_q6(static_cast<int>((low & 0x0fu) | ((high & 0x03u) << 4)));
    const int q1 = sign_extend_q6(static_cast<int>((low >> 4) | ((high & 0x0cu) << 2)));
    const int k0 = group * kGroupK + lane * 2;
    const float2 xv = __bfloat1622float2(x2[k0 >> 1]);
    acc = fmaf(static_cast<float>(q0) * scale, xv.x, acc);
    acc = fmaf(static_cast<float>(q1) * scale, xv.y, acc);
    return acc;
}

__global__ void linear_rowsplit_gemv_lm_head_q6_kernel(const __nv_bfloat16* __restrict__ x,
                                                       const std::uint8_t* __restrict__ codes,
                                                       const std::uint8_t* __restrict__ high_bits,
                                                       const std::uint8_t* __restrict__ scales,
                                                       __nv_bfloat16* __restrict__ out) {
    __shared__ uint4 nibble_tile[kWarpsPerBlock][kNibbleVecsPerWarpTile];
    __shared__ uint4 high_tile[kWarpsPerBlock][kHighVecsPerWarpTile];

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int row = static_cast<int>(blockIdx.x) * kWarpsPerBlock + warp;
    if (row >= kN) { return; }

    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(row) * kGroups * kNibbleBytesPerGroup;
    const std::uint8_t* high_row =
        high_bits + static_cast<std::int64_t>(row) * kGroups * kHighBytesPerGroup;
    const std::uint8_t* scale_row = scales + static_cast<std::int64_t>(row) * kGroups * 2;
    const auto* x2 = reinterpret_cast<const __nv_bfloat162*>(x);

    float acc = 0.0f;
    for (int tile = 0; tile < kGroups; tile += kGroupsPerWarpTile) {
        const auto* nibble_vecs =
            reinterpret_cast<const uint4*>(code_row + tile * kNibbleBytesPerGroup);
        const auto* high_vecs =
            reinterpret_cast<const uint4*>(high_row + tile * kHighBytesPerGroup);
        nibble_tile[warp][lane] = nibble_vecs[lane];
        if (lane < kHighVecsPerWarpTile) { high_tile[warp][lane] = high_vecs[lane]; }
        __syncwarp();

        std::uint16_t lane_scale_bits = 0;
        if (lane < kGroupsPerWarpTile) {
            lane_scale_bits = load_scale_bits(scale_row, tile + lane);
        }
        const auto* tile_nibbles = reinterpret_cast<const std::uint8_t*>(nibble_tile[warp]);
        const auto* tile_highs = reinterpret_cast<const std::uint8_t*>(high_tile[warp]);

#pragma unroll
        for (int tile_group = 0; tile_group < kGroupsPerWarpTile; ++tile_group) {
            const auto scale_bits = static_cast<std::uint16_t>(
                __shfl_sync(0xffffffffu, lane_scale_bits, tile_group));
            acc = accumulate_group(
                x2, tile_nibbles + tile_group * kNibbleBytesPerGroup,
                tile_highs + tile_group * kHighBytesPerGroup, scale_bits, lane, tile + tile_group,
                acc);
        }
        __syncwarp();
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0) { out[row] = __float2bfloat16(acc); }
}

} // namespace

void linear_rowsplit_gemv_lm_head_q6_launch(const Tensor& x, const Weight& w, Tensor& out,
                                            WorkspaceArena& ws, cudaStream_t stream) {
    (void)ws;
    if (w.n != kN || w.k != kK || w.padded_shape[1] != kK) {
        throw std::invalid_argument("linear: lm_head Q6 tuned GEMV requires 248320x5120");
    }
    const int grid = (kN + kWarpsPerBlock - 1) / kWarpsPerBlock;
    linear_rowsplit_gemv_lm_head_q6_kernel<<<grid, kBlockThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.qhigh),
        static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace qus::kernels::detail
