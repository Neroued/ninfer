#include "kernels/linear/gemv/linear_rowsplit_gemv_lm_head.cuh"

#include "kernels/common/math.h"
#include "kernels/common/memory.cuh"
#include "kernels/common/warp.cuh"
#include "qus/core/device.h" // CUDA_CHECK

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>

namespace qus::kernels::detail {
namespace {

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

__device__ __forceinline__ float accumulate_group(const __nv_bfloat162* __restrict__ x2,
                                                  const std::uint8_t* __restrict__ nibble_group,
                                                  const std::uint8_t* __restrict__ high_group,
                                                  std::uint16_t scale_bits,
                                                  int lane, int group, float acc) {
    const float scale = __half2float(__ushort_as_half(scale_bits));

    const std::uint8_t low = nibble_group[lane];
    const std::uint8_t high = high_group[lane >> 1] >> ((lane & 1) * 4);

    const int q0 = sign_extend<6>(static_cast<int>((low & 0x0fu) | ((high & 0x03u) << 4)));
    const int q1 = sign_extend<6>(static_cast<int>((low >> 4) | ((high & 0x0cu) << 2)));
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
                                                       int n,
                                                       __nv_bfloat16* __restrict__ out) {
    __shared__ uint4 nibble_tile[kWarpsPerBlock][kNibbleVecsPerWarpTile];
    __shared__ uint4 high_tile[kWarpsPerBlock][kHighVecsPerWarpTile];

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int row = static_cast<int>(blockIdx.x) * kWarpsPerBlock + warp;
    if (row >= n) { return; }

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
            lane_scale_bits = load_vec<std::uint16_t>(scale_row + (tile + lane) * 2);
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

    acc = warp_reduce_sum(acc);
    if (lane == 0) { out[row] = __float2bfloat16(acc); }
}

// Q4 variant of the tuned warp-per-row lm_head GEMV. Identical geometry to the Q6
// kernel minus the high-bit plane: each nibble byte holds two signed 4-bit codes,
// so there is no high tile to stage and each code dequants by sign-extending its nibble.
__global__ void linear_rowsplit_gemv_lm_head_q4_kernel(const __nv_bfloat16* __restrict__ x,
                                                       const std::uint8_t* __restrict__ codes,
                                                       const std::uint8_t* __restrict__ scales,
                                                       int n,
                                                       __nv_bfloat16* __restrict__ out) {
    __shared__ uint4 nibble_tile[kWarpsPerBlock][kNibbleVecsPerWarpTile];

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int row = static_cast<int>(blockIdx.x) * kWarpsPerBlock + warp;
    if (row >= n) { return; }

    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(row) * kGroups * kNibbleBytesPerGroup;
    const std::uint8_t* scale_row = scales + static_cast<std::int64_t>(row) * kGroups * 2;
    const auto* x2 = reinterpret_cast<const __nv_bfloat162*>(x);

    float acc = 0.0f;
    for (int tile = 0; tile < kGroups; tile += kGroupsPerWarpTile) {
        const auto* nibble_vecs =
            reinterpret_cast<const uint4*>(code_row + tile * kNibbleBytesPerGroup);
        nibble_tile[warp][lane] = nibble_vecs[lane];
        __syncwarp();

        std::uint16_t lane_scale_bits = 0;
        if (lane < kGroupsPerWarpTile) {
            lane_scale_bits = load_vec<std::uint16_t>(scale_row + (tile + lane) * 2);
        }
        const auto* tile_nibbles = reinterpret_cast<const std::uint8_t*>(nibble_tile[warp]);

#pragma unroll
        for (int tile_group = 0; tile_group < kGroupsPerWarpTile; ++tile_group) {
            const auto scale_bits = static_cast<std::uint16_t>(
                __shfl_sync(0xffffffffu, lane_scale_bits, tile_group));
            const float scale = __half2float(__ushort_as_half(scale_bits));

            const std::uint8_t low = tile_nibbles[tile_group * kNibbleBytesPerGroup + lane];
            const int q0 = sign_extend<4>(static_cast<int>(low & 0x0fu));
            const int q1 = sign_extend<4>(static_cast<int>(low >> 4));
            const int k0 = (tile + tile_group) * kGroupK + lane * 2;
            const float2 xv = __bfloat1622float2(x2[k0 >> 1]);
            acc = fmaf(static_cast<float>(q0) * scale, xv.x, acc);
            acc = fmaf(static_cast<float>(q1) * scale, xv.y, acc);
        }
        __syncwarp();
    }

    acc = warp_reduce_sum(acc);
    if (lane == 0) { out[row] = __float2bfloat16(acc); }
}

} // namespace

void linear_rowsplit_gemv_lm_head_q6_launch(const Tensor& x, const Weight& w, Tensor& out,
                                            WorkspaceArena& ws, cudaStream_t stream) {
    (void)ws;
    // K is fixed (drives the constexpr group/tile geometry and the unrolled inner
    // loop); N is a runtime parameter so the same tuned warp-per-row kernel serves
    // the full vocab head and any draft head (65536/98304/131072) unchanged.
    if (w.k != kK || w.padded_shape[1] != kK) {
        throw std::invalid_argument("linear: lm_head Q6 tuned GEMV requires k=5120");
    }
    if (w.n <= 0) { throw std::invalid_argument("linear: lm_head Q6 tuned GEMV requires n>0"); }
    const int n    = static_cast<int>(w.n);
    const int grid = div_up(n, kWarpsPerBlock);
    linear_rowsplit_gemv_lm_head_q6_kernel<<<grid, kBlockThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.qhigh),
        static_cast<const std::uint8_t*>(w.scales), n, static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

void linear_rowsplit_gemv_lm_head_q4_launch(const Tensor& x, const Weight& w, Tensor& out,
                                            WorkspaceArena& ws, cudaStream_t stream) {
    (void)ws;
    // Same contract as the Q6 launch: K fixed at 5120, N runtime so the one tuned
    // kernel serves any Q4 draft head (65536/98304/131072). No high plane is read.
    if (w.k != kK || w.padded_shape[1] != kK) {
        throw std::invalid_argument("linear: lm_head Q4 tuned GEMV requires k=5120");
    }
    if (w.n <= 0) { throw std::invalid_argument("linear: lm_head Q4 tuned GEMV requires n>0"); }
    const int n    = static_cast<int>(w.n);
    const int grid = div_up(n, kWarpsPerBlock);
    linear_rowsplit_gemv_lm_head_q4_kernel<<<grid, kBlockThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), n, static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace qus::kernels::detail
