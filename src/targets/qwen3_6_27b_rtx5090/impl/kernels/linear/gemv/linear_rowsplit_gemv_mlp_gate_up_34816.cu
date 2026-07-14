#include "kernels/linear/gemv/linear_rowsplit_gemv_mlp_gate_up_34816.cuh"

#include "kernels/common/math.cuh"
#include "kernels/common/memory.cuh"
#include "kernels/common/warp.cuh"
#include "core/device.h" // CUDA_CHECK

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::kernels::detail {
namespace {

constexpr int kN = 34816;
constexpr int kK = 5120;
constexpr int kIntermediate = kN / 2;
constexpr int kGroupK = 64;
constexpr int kGroups = kK / kGroupK;
constexpr int kBytesPerGroup = 32;
constexpr int kVecBytes = 16;
constexpr int kGroupsPerWarpTile = 16;
constexpr int kVecsPerWarpTile = kGroupsPerWarpTile * kBytesPerGroup / kVecBytes;
constexpr int kWarpsPerBlock = 4;
constexpr int kBlockThreads = kWarpsPerBlock * 32;
constexpr int kPairsPerBlock = kWarpsPerBlock;
constexpr int kXVecs = kK / 8; // x as uint4 (8 bf16 each)
constexpr int kTiles = kGroups / kGroupsPerWarpTile;
static_assert(kIntermediate % kPairsPerBlock == 0);
static_assert(kBytesPerGroup == 2 * kVecBytes);
static_assert(kGroups % kGroupsPerWarpTile == 0);
static_assert(kVecsPerWarpTile == 32);

__global__ void linear_rowsplit_gemv_mlp_gate_up_34816_q4_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, __nv_bfloat16* __restrict__ out) {
    __shared__ __align__(16) __nv_bfloat16 x_sh[kK];
    __shared__ uint4 code_tile[kWarpsPerBlock][kVecsPerWarpTile];

    auto*       x_sh_v = reinterpret_cast<uint4*>(x_sh);
    const auto* x_g    = reinterpret_cast<const uint4*>(x);
    for (int i = static_cast<int>(threadIdx.x); i < kXVecs;
         i += static_cast<int>(blockDim.x)) {
        x_sh_v[i] = x_g[i];
    }
    __syncthreads();

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int row = static_cast<int>(blockIdx.x) * kWarpsPerBlock + warp;
    if (row >= kN) { return; }

    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(row) * kGroups * kBytesPerGroup;
    const std::uint8_t* scale_row = scales + static_cast<std::int64_t>(row) * kGroups * 2;
    const auto* x2 = reinterpret_cast<const __nv_bfloat162*>(x_sh);

    float acc = 0.0f;
    for (int tile = 0; tile < kGroups; tile += kGroupsPerWarpTile) {
        const auto* global_vecs = reinterpret_cast<const uint4*>(
            code_row + tile * kBytesPerGroup);
        code_tile[warp][lane] = global_vecs[lane];
        __syncwarp();

        const auto* tile_codes =
            reinterpret_cast<const std::uint8_t*>(code_tile[warp]);
        std::uint16_t lane_scale_bits = 0;
        if (lane < kGroupsPerWarpTile) {
            const std::uint8_t* sp = scale_row + (tile + lane) * 2;
            lane_scale_bits = static_cast<std::uint16_t>(sp[0]) |
                              static_cast<std::uint16_t>(static_cast<std::uint16_t>(sp[1]) << 8);
        }

#pragma unroll
        for (int tile_group = 0; tile_group < kGroupsPerWarpTile; ++tile_group) {
            const auto scale_bits = static_cast<std::uint16_t>(
                __shfl_sync(0xffffffffu, lane_scale_bits, tile_group));
            const float scale = __half2float(__ushort_as_half(scale_bits));

            const int packed = static_cast<int>(tile_codes[tile_group * kBytesPerGroup + lane]);
            const int q0 = sign_extend<4>(packed & 0x0f);
            const int q1 = sign_extend<4>(packed >> 4);
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

__device__ __forceinline__ void q4_issue_pair_tile(
    uint4 (*__restrict__ s_code)[kVecsPerWarpTile], uint4 (*__restrict__ s_scale)[2],
    const std::uint8_t* __restrict__ gate_code_row,
    const std::uint8_t* __restrict__ gate_scale_row,
    const std::uint8_t* __restrict__ up_code_row,
    const std::uint8_t* __restrict__ up_scale_row, int tile, int lane) {
    const int g0 = tile * kGroupsPerWarpTile;
    pipe_copy<16>(
        &s_code[0][lane],
        reinterpret_cast<const uint4*>(gate_code_row + g0 * kBytesPerGroup) + lane);
    pipe_copy<16>(
        &s_code[1][lane],
        reinterpret_cast<const uint4*>(up_code_row + g0 * kBytesPerGroup) + lane);
    if (lane < 2) {
        pipe_copy<16>(
            &s_scale[0][lane],
            reinterpret_cast<const uint4*>(gate_scale_row + g0 * 2) + lane);
        pipe_copy<16>(&s_scale[1][lane],
                      reinterpret_cast<const uint4*>(up_scale_row + g0 * 2) + lane);
    }
    pipe_commit();
}

__global__ void linear_rowsplit_gemv_mlp_gate_up_silu_17408_q4_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, __nv_bfloat16* __restrict__ out) {
    constexpr int kStages = 3;
    constexpr int kPrefetch = kStages - 1;
    __shared__ __align__(16) __nv_bfloat16 x_sh[kK];
    __shared__ uint4 code_tile[kWarpsPerBlock][kStages][2][kVecsPerWarpTile];
    __shared__ uint4 scale_tile[kWarpsPerBlock][kStages][2][2];

    auto*       x_sh_v = reinterpret_cast<uint4*>(x_sh);
    const auto* x_g    = reinterpret_cast<const uint4*>(x);
    for (int i = static_cast<int>(threadIdx.x); i < kXVecs;
         i += static_cast<int>(blockDim.x)) {
        x_sh_v[i] = x_g[i];
    }
    __syncthreads();

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int out_row = static_cast<int>(blockIdx.x) * kPairsPerBlock + warp;

    const std::uint8_t* gate_code_row =
        codes + static_cast<std::int64_t>(out_row) * kGroups * kBytesPerGroup;
    const std::uint8_t* gate_scale_row =
        scales + static_cast<std::int64_t>(out_row) * kGroups * 2;
    const std::uint8_t* up_code_row =
        codes + static_cast<std::int64_t>(out_row + kIntermediate) * kGroups * kBytesPerGroup;
    const std::uint8_t* up_scale_row =
        scales + static_cast<std::int64_t>(out_row + kIntermediate) * kGroups * 2;
    const auto* x2 = reinterpret_cast<const __nv_bfloat162*>(x_sh);

    float gate_acc = 0.0f;
    float up_acc = 0.0f;
#pragma unroll
    for (int p = 0; p < kPrefetch; ++p) {
        if (p < kTiles) {
            q4_issue_pair_tile(code_tile[warp][p], scale_tile[warp][p], gate_code_row,
                               gate_scale_row, up_code_row, up_scale_row, p, lane);
        } else {
            pipe_commit();
        }
    }

#pragma unroll 1
    for (int tile = 0; tile < kTiles; ++tile) {
        const int fetch = tile + kPrefetch;
        if (fetch < kTiles) {
            const int buf = fetch % kStages;
            q4_issue_pair_tile(code_tile[warp][buf], scale_tile[warp][buf], gate_code_row,
                               gate_scale_row, up_code_row, up_scale_row, fetch, lane);
        } else {
            pipe_commit();
        }
        pipe_wait<kPrefetch>();
        __syncwarp();

        const int buf = tile % kStages;
        const auto* gate_codes =
            reinterpret_cast<const std::uint8_t*>(code_tile[warp][buf][0]);
        const auto* up_codes =
            reinterpret_cast<const std::uint8_t*>(code_tile[warp][buf][1]);
        const auto* gate_scales =
            reinterpret_cast<const std::uint16_t*>(scale_tile[warp][buf][0]);
        const auto* up_scales =
            reinterpret_cast<const std::uint16_t*>(scale_tile[warp][buf][1]);
#pragma unroll
        for (int tile_group = 0; tile_group < kGroupsPerWarpTile; ++tile_group) {
            const float gate_scale =
                __half2float(__ushort_as_half(static_cast<std::uint16_t>(gate_scales[tile_group])));
            const float up_scale =
                __half2float(__ushort_as_half(static_cast<std::uint16_t>(up_scales[tile_group])));

            const int gate_packed =
                static_cast<int>(gate_codes[tile_group * kBytesPerGroup + lane]);
            const int gate_q0 = sign_extend<4>(gate_packed & 0x0f);
            const int gate_q1 = sign_extend<4>(gate_packed >> 4);
            const int up_packed = static_cast<int>(up_codes[tile_group * kBytesPerGroup + lane]);
            const int up_q0 = sign_extend<4>(up_packed & 0x0f);
            const int up_q1 = sign_extend<4>(up_packed >> 4);
            const int k0 = (tile * kGroupsPerWarpTile + tile_group) * kGroupK + lane * 2;
            const float2 xv = __bfloat1622float2(x2[k0 >> 1]);
            gate_acc = fmaf(static_cast<float>(gate_q0) * gate_scale, xv.x, gate_acc);
            gate_acc = fmaf(static_cast<float>(gate_q1) * gate_scale, xv.y, gate_acc);
            up_acc = fmaf(static_cast<float>(up_q0) * up_scale, xv.x, up_acc);
            up_acc = fmaf(static_cast<float>(up_q1) * up_scale, xv.y, up_acc);
        }
        __syncwarp();
    }

    gate_acc = warp_reduce_sum(gate_acc);
    up_acc   = warp_reduce_sum(up_acc);
    if (lane == 0) {
        const float g = __bfloat162float(__float2bfloat16(gate_acc));
        const float u = __bfloat162float(__float2bfloat16(up_acc));
        out[out_row] = __float2bfloat16(silu(g) * u);
    }
}

} // namespace

void linear_rowsplit_gemv_mlp_gate_up_34816_q4_launch(const Tensor& x, const Weight& w,
                                                      Tensor& out, WorkspaceArena& ws,
                                                      cudaStream_t stream) {
    (void)ws;
    if (w.n != kN || w.k != kK || w.padded_shape[1] != kK) {
        throw std::invalid_argument("linear: MLP gate/up Q4 fused GEMV requires 34816x5120");
    }
    const int grid = div_up(kN, kWarpsPerBlock);
    linear_rowsplit_gemv_mlp_gate_up_34816_q4_kernel<<<grid, kBlockThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

void linear_rowsplit_gemv_mlp_gate_up_silu_17408_q4_launch(const Tensor& x, const Weight& w,
                                                           Tensor& out, cudaStream_t stream) {
    if (w.n != kN || w.k != kK || w.padded_shape[1] != kK) {
        throw std::invalid_argument("mlp_gate_up_silu_decode: Q4 fused weight requires 34816x5120");
    }
    const int grid = kIntermediate / kPairsPerBlock;
    linear_rowsplit_gemv_mlp_gate_up_silu_17408_q4_kernel<<<grid, kBlockThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::kernels::detail
