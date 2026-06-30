#include "kernels/linear/gemv/linear_rowsplit_gemv_mlp_down.cuh"

#include "qus/core/device.h" // CUDA_CHECK

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>

namespace qus::kernels::detail {
namespace {

constexpr int kN = 5120;
constexpr int kK = 17408;
constexpr int kGroupK = 64;
constexpr int kGroups = kK / kGroupK;
constexpr int kBytesPerGroup = 40;
constexpr int kWarpsPerBlock = 4;
constexpr int kBlockThreads = kWarpsPerBlock * 32;
constexpr int kSplitK = 8;

struct ArenaScope {
    WorkspaceArena& ws;
    std::size_t mark;

    explicit ArenaScope(WorkspaceArena& arena) : ws(arena), mark(arena.mark()) {}
    ~ArenaScope() { ws.rewind(mark); }

    ArenaScope(const ArenaScope&) = delete;
    ArenaScope& operator=(const ArenaScope&) = delete;
};

__device__ __forceinline__ int sign_extend_q5(int v) {
    return (v & 0x10) ? (v - 32) : v;
}

__device__ __forceinline__ std::uint32_t load_q5_pair_bits(const std::uint8_t* __restrict__ group,
                                                           int lane) {
    const int bitpos      = lane * 10;
    const int byte_offset = bitpos >> 3;
    const int bit_shift   = bitpos & 7;
    const std::uint8_t* p = group + byte_offset;
    return (static_cast<std::uint32_t>(p[0]) |
            (static_cast<std::uint32_t>(p[1]) << 8)) >>
           bit_shift;
}

__device__ __forceinline__ float accumulate_group(const __nv_bfloat162* __restrict__ x2,
                                                  const std::uint8_t* __restrict__ code_group,
                                                  std::uint16_t scale_bits,
                                                  int lane, int group, float acc) {
    const float scale = __half2float(__ushort_as_half(scale_bits));
    const std::uint32_t bits = load_q5_pair_bits(code_group, lane);

    const int q0 = sign_extend_q5(static_cast<int>(bits & 0x1fu));
    const int q1 = sign_extend_q5(static_cast<int>((bits >> 5) & 0x1fu));
    const int k0 = group * kGroupK + lane * 2;
    const float2 xv = __bfloat1622float2(x2[k0 >> 1]);
    acc = fmaf(static_cast<float>(q0) * scale, xv.x, acc);
    acc = fmaf(static_cast<float>(q1) * scale, xv.y, acc);
    return acc;
}

__global__ void linear_rowsplit_gemv_mlp_down_q5_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, float* __restrict__ partials) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int row = static_cast<int>(blockIdx.x) * kWarpsPerBlock + warp;
    if (row >= kN) { return; }
    const int split = static_cast<int>(blockIdx.y);
    const int group_begin = split * (kGroups / kSplitK);
    const int group_end = group_begin + (kGroups / kSplitK);

    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(row) * kGroups * kBytesPerGroup;
    const std::uint8_t* scale_row = scales + static_cast<std::int64_t>(row) * kGroups * 2;
    const auto* x2 = reinterpret_cast<const __nv_bfloat162*>(x);

    float acc = 0.0f;
    int group = group_begin;
#pragma unroll 1
    for (; group + 2 < group_end; group += 3) {
        const std::uint8_t* code_group0 = code_row + group * kBytesPerGroup;
        const std::uint8_t* code_group1 = code_group0 + kBytesPerGroup;
        const std::uint8_t* code_group2 = code_group1 + kBytesPerGroup;

        std::uint16_t scale0_bits = 0;
        std::uint16_t scale1_bits = 0;
        std::uint16_t scale2_bits = 0;
        if (lane == 0) {
            const std::uint8_t* sp = scale_row + group * 2;
            scale0_bits = static_cast<std::uint16_t>(sp[0]) |
                          static_cast<std::uint16_t>(static_cast<std::uint16_t>(sp[1]) << 8);
        } else if (lane == 1) {
            const std::uint8_t* sp = scale_row + (group + 1) * 2;
            scale1_bits = static_cast<std::uint16_t>(sp[0]) |
                          static_cast<std::uint16_t>(static_cast<std::uint16_t>(sp[1]) << 8);
        } else if (lane == 2) {
            const std::uint8_t* sp = scale_row + (group + 2) * 2;
            scale2_bits = static_cast<std::uint16_t>(sp[0]) |
                          static_cast<std::uint16_t>(static_cast<std::uint16_t>(sp[1]) << 8);
        }
        scale0_bits = static_cast<std::uint16_t>(__shfl_sync(0xffffffffu, scale0_bits, 0));
        scale1_bits = static_cast<std::uint16_t>(__shfl_sync(0xffffffffu, scale1_bits, 1));
        scale2_bits = static_cast<std::uint16_t>(__shfl_sync(0xffffffffu, scale2_bits, 2));

        acc = accumulate_group(x2, code_group0, scale0_bits, lane, group, acc);
        acc = accumulate_group(x2, code_group1, scale1_bits, lane, group + 1, acc);
        acc = accumulate_group(x2, code_group2, scale2_bits, lane, group + 2, acc);
    }

    if (group < group_end) {
        const bool has_group1 = group + 1 < group_end;
        const std::uint8_t* code_group0 = code_row + group * kBytesPerGroup;
        const std::uint8_t* code_group1 = code_group0 + kBytesPerGroup;

        std::uint16_t scale0_bits = 0;
        std::uint16_t scale1_bits = 0;
        if (lane == 0) {
            const std::uint8_t* sp = scale_row + group * 2;
            scale0_bits = static_cast<std::uint16_t>(sp[0]) |
                          static_cast<std::uint16_t>(static_cast<std::uint16_t>(sp[1]) << 8);
        } else if (has_group1 && lane == 1) {
            const std::uint8_t* sp = scale_row + (group + 1) * 2;
            scale1_bits = static_cast<std::uint16_t>(sp[0]) |
                          static_cast<std::uint16_t>(static_cast<std::uint16_t>(sp[1]) << 8);
        }
        scale0_bits = static_cast<std::uint16_t>(__shfl_sync(0xffffffffu, scale0_bits, 0));
        scale1_bits = static_cast<std::uint16_t>(__shfl_sync(0xffffffffu, scale1_bits, 1));

        acc = accumulate_group(x2, code_group0, scale0_bits, lane, group, acc);
        if (has_group1) {
            acc = accumulate_group(x2, code_group1, scale1_bits, lane, group + 1, acc);
        }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0) { partials[static_cast<std::int64_t>(split) * kN + row] = acc; }
}

__global__ void linear_rowsplit_gemv_mlp_down_q5_reduce_kernel(
    const float* __restrict__ partials, __nv_bfloat16* __restrict__ out) {
    const int row = static_cast<int>(blockIdx.x) * blockDim.x + static_cast<int>(threadIdx.x);
    if (row >= kN) { return; }
    float acc = 0.0f;
#pragma unroll
    for (int split = 0; split < kSplitK; ++split) {
        acc += partials[static_cast<std::int64_t>(split) * kN + row];
    }
    out[row] = __float2bfloat16(acc);
}

} // namespace

void linear_rowsplit_gemv_mlp_down_q5_launch(const Tensor& x, const Weight& w, Tensor& out,
                                             WorkspaceArena& ws, cudaStream_t stream) {
    if (w.n != kN || w.k != kK || w.padded_shape[1] != kK) {
        throw std::invalid_argument("linear: MLP down Q5 tuned GEMV requires 5120x17408");
    }
    ArenaScope arena_scope(ws);
    Tensor partials_tensor = ws.alloc(DType::FP32, {kSplitK, kN});
    auto* partials = static_cast<float*>(partials_tensor.data);

    const int grid = (kN + kWarpsPerBlock - 1) / kWarpsPerBlock;
    linear_rowsplit_gemv_mlp_down_q5_kernel<<<dim3(grid, kSplitK, 1), kBlockThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), partials);
    CUDA_CHECK(cudaGetLastError());

    constexpr int kReduceThreads = 256;
    const int reduce_grid = (kN + kReduceThreads - 1) / kReduceThreads;
    linear_rowsplit_gemv_mlp_down_q5_reduce_kernel<<<reduce_grid, kReduceThreads, 0, stream>>>(
        partials, static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace qus::kernels::detail
