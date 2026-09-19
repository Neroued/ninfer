#include "core/weight.h"
#include "ops/sparse_moe/decode/sparse_moe_decode.h"

#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"
#include "ops/linear/q4/q4_rowsplit_storage.cuh"
#include "ops/linear/q5/q5_rowsplit_storage.cuh"
#include "ops/linear/q6/q6_rowsplit_storage.cuh"
#include "ops/linear/q8/q8_rowsplit_storage.cuh"
#include "ops/sparse_moe/sparse_moe_route.cuh"
#include "ops/sparse_moe/small_t/sparse_moe_small_t.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kHidden           = 2048;
constexpr int kExperts          = 256;
constexpr int kRouterRows       = kExperts + 1;
constexpr int kTopK             = 8;
constexpr int kIntermediate     = 512;
constexpr int kD1Warps          = 8;
constexpr int kAdaptiveD3Blocks = 5 * kIntermediate;
constexpr int kAdaptiveD4Blocks = 5 * (kHidden / 4);

__device__ __forceinline__ float dot_bf16_eight(const __nv_bfloat16* a, const __nv_bfloat16* b) {
    const uint4 av  = load_vec<uint4>(a);
    const uint4 bv  = load_vec<uint4>(b);
    const float2 a0 = bf16x2_bits_to_float2(av.x);
    const float2 a1 = bf16x2_bits_to_float2(av.y);
    const float2 a2 = bf16x2_bits_to_float2(av.z);
    const float2 a3 = bf16x2_bits_to_float2(av.w);
    const float2 b0 = bf16x2_bits_to_float2(bv.x);
    const float2 b1 = bf16x2_bits_to_float2(bv.y);
    const float2 b2 = bf16x2_bits_to_float2(bv.z);
    const float2 b3 = bf16x2_bits_to_float2(bv.w);
    float sum       = 0.0f;
    sum             = fmaf(a0.x, b0.x, sum);
    sum             = fmaf(a0.y, b0.y, sum);
    sum             = fmaf(a1.x, b1.x, sum);
    sum             = fmaf(a1.y, b1.y, sum);
    sum             = fmaf(a2.x, b2.x, sum);
    sum             = fmaf(a2.y, b2.y, sum);
    sum             = fmaf(a3.x, b3.x, sum);
    sum             = fmaf(a3.y, b3.y, sum);
    return sum;
}

__device__ __forceinline__ float router_row_dot(const __nv_bfloat16* x, const __nv_bfloat16* row) {
    constexpr int kSlice = kHidden / kD1Warps;
    constexpr int kVecs  = kSlice / (32 * 8);
    const int warp       = static_cast<int>(threadIdx.x) >> 5;
    const int lane       = static_cast<int>(threadIdx.x) & 31;
    float sum            = 0.0f;
#pragma unroll
    for (int vector = 0; vector < kVecs; ++vector) {
        const int k = warp * kSlice + vector * 32 * 8 + lane * 8;
        sum += dot_bf16_eight(row + k, x + k);
    }
    return warp_reduce_sum(sum);
}

__global__ void sparse_moe_d1_kernel(const __nv_bfloat16* __restrict__ x,
                                     const __nv_bfloat16* __restrict__ router,
                                     float* __restrict__ scores,
                                     const char* __restrict__ shared_down_codes,
                                     unsigned long long shared_down_bytes) {
    __shared__ float partial[kD1Warps];
    const int row   = static_cast<int>(blockIdx.x);
    const int warp  = static_cast<int>(threadIdx.x) >> 5;
    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const float dot = router_row_dot(x, router + static_cast<std::int64_t>(row) * kHidden);
    if (lane == 0) { partial[warp] = dot; }
    __syncthreads();
    if (warp == 0) {
        float value = lane < kD1Warps ? partial[lane] : 0.0f;
        value       = warp_reduce_sum<kD1Warps>(value);
        if (lane == 0) { scores[row] = value; }
    }
    if (shared_down_codes != nullptr) {
        // Warm L2 for the shared-expert down codes that D4's shared warp
        // (the block's long pole) will stream at t~14.3us. Pure cache hint.
        const unsigned long long offset =
            (static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x) * 128ull;
        if (offset < shared_down_bytes) {
            asm volatile("prefetch.global.L2 [%0];" ::"l"(shared_down_codes + offset));
        }
    }
}

__global__ void sparse_moe_d2_warp_kernel(const float* __restrict__ scores, int* __restrict__ ids,
                                          float* __restrict__ alpha,
                                          float* __restrict__ shared_scale) {
    __shared__ float selected_logits[kTopK];
    if (threadIdx.x == 0) { pdl::trigger_dependents(); }
    sparse_moe_select_top8_warp(scores, ids, alpha, shared_scale, selected_logits);
}

// The stored planes of one matrix. They travel together because a dot product takes one matrix at
// a time and each codec reads a different subset of them: a call site handing them over one by one
// had to know that subset before it could name the codec.
struct SparseMoePlanes {
    const std::uint8_t* __restrict__ codes  = nullptr;
    const std::uint8_t* __restrict__ high   = nullptr;
    const std::uint8_t* __restrict__ scales = nullptr;
    // One divisor per source matrix the plane was assembled from, each covering `divisor_rows`
    // consecutive rows. Only NVFP4 has them.
    const float* __restrict__ divisors = nullptr;
    int divisor_rows                   = 1;
};

// Codecs come in two lane ownerships. Under `kPackedWord8` a lane owns eight consecutive K values
// and a single decode feeds eight FP32 FMAs, so four adjacent groups of eight lanes each form one
// 128-byte warp transaction. The other ownership is scalar - one value per lane in D3, a pair in
// D4 - which is what a 32-wide group leaves room for. Either way the codec takes the matrix's
// column count as a template argument, because it walks its own row and only it knows how its
// planes are laid out.

struct Q4Codec {
    static constexpr int kGroupK              = 64;
    static constexpr bool kPackedWord8        = true;
    static constexpr bool kSingleValuePerLane = false;

    __device__ static __forceinline__ float row_coefficient(const SparseMoePlanes&, int) {
        return 1.0F;
    }

    template <int K>
    __device__ static __forceinline__ void load_eight(const SparseMoePlanes& planes, int row,
                                                      int group, int lane_in_group, float,
                                                      float (&weights)[8]) {
        const std::int64_t index   = static_cast<std::int64_t>(row) * (K / kGroupK) + group;
        const std::uint32_t packed = *reinterpret_cast<const std::uint32_t*>(
            planes.codes + index * Q4RowSplitStorage::kCodeBytesPerGroup + lane_in_group * 4);
        const auto scale_bits = *reinterpret_cast<const std::uint16_t*>(
            planes.scales + index * Q4RowSplitStorage::kScaleBytesPerGroup);
        Q4SimtDecodeAtom::decode_eight(packed, scale_bits, weights);
    }
};

struct Q5Codec {
    static constexpr int kGroupK              = 64;
    static constexpr bool kPackedWord8        = true;
    static constexpr bool kSingleValuePerLane = false;

    __device__ static __forceinline__ float row_coefficient(const SparseMoePlanes&, int) {
        return 1.0F;
    }

    template <int K>
    __device__ static __forceinline__ void load_eight(const SparseMoePlanes& planes, int row,
                                                      int group, int lane_in_group, float,
                                                      float (&weights)[8]) {
        const std::int64_t index   = static_cast<std::int64_t>(row) * (K / kGroupK) + group;
        const std::uint32_t packed = *reinterpret_cast<const std::uint32_t*>(
            planes.codes + index * Q5RowSplitStorage::kCodeBytesPerGroup + lane_in_group * 4);
        const std::uint8_t high_bits =
            planes.high[index * Q5RowSplitStorage::kHighBytesPerGroup + lane_in_group];
        const auto scale_bits = *reinterpret_cast<const std::uint16_t*>(
            planes.scales + index * Q5RowSplitStorage::kScaleBytesPerGroup);
        Q5SimtDecodeAtom::decode_eight(packed, high_bits, scale_bits, weights);
    }
};

struct Q6Codec {
    static constexpr int kGroupK              = 64;
    static constexpr bool kPackedWord8        = true;
    static constexpr bool kSingleValuePerLane = false;

    __device__ static __forceinline__ float row_coefficient(const SparseMoePlanes&, int) {
        return 1.0F;
    }

    template <int K>
    __device__ static __forceinline__ void load_eight(const SparseMoePlanes& planes, int row,
                                                      int group, int lane_in_group, float,
                                                      float (&weights)[8]) {
        const std::int64_t index   = static_cast<std::int64_t>(row) * (K / kGroupK) + group;
        const std::uint32_t packed = *reinterpret_cast<const std::uint32_t*>(
            planes.codes + index * Q6RowSplitStorage::kCodeBytesPerGroup + lane_in_group * 4);
        const std::uint16_t high_bits = *reinterpret_cast<const std::uint16_t*>(
            planes.high + index * Q6RowSplitStorage::kHighBytesPerGroup + lane_in_group * 2);
        const auto scale_bits = *reinterpret_cast<const std::uint16_t*>(
            planes.scales + index * Q6RowSplitStorage::kScaleBytesPerGroup);
        Q6SimtDecodeAtom::decode_eight(packed, high_bits, scale_bits, weights);
    }
};

struct Q8Codec {
    static constexpr int kGroupK              = 32;
    static constexpr bool kPackedWord8        = false;
    static constexpr bool kSingleValuePerLane = true;

    __device__ static __forceinline__ float row_coefficient(const SparseMoePlanes&, int) {
        return 1.0F;
    }

    template <int K>
    __device__ static __forceinline__ float load_one(const SparseMoePlanes& planes, int row,
                                                     int group, int lane) {
        const std::int64_t index = static_cast<std::int64_t>(row) * (K / kGroupK) + group;
        const float scale = __half2float(__ushort_as_half(*reinterpret_cast<const std::uint16_t*>(
            planes.scales + index * Q8RowSplitStorage::kScaleBytesPerGroup)));
        return static_cast<float>(static_cast<std::int8_t>(
                   planes.codes[index * Q8RowSplitStorage::kCodeBytesPerGroup + lane])) *
               scale;
    }

    template <int K>
    __device__ static __forceinline__ void load_pair(const SparseMoePlanes& planes, int row,
                                                     int group, int lane, float& w0, float& w1) {
        Q8ScalarDecodeAtom::load_pair(planes.codes, planes.high, planes.scales,
                                      static_cast<std::int64_t>(row) * (K / kGroupK) + group, lane,
                                      w0, w1);
    }
};

// NVFP4 packs two e2m1 codes per byte and one e4m3 scale per sixteen values, so a lane's eight
// values are four code bytes and exactly one scale: whichever half of a block the lane owns, its
// eight values lie inside that one block. Only the scale plane is swizzled; the codes are plain
// row-major. The per-tensor divisor is folded into the decoded block scale, once per eight values
// rather than once per value.
struct Nvfp4Codec {
    static constexpr int kGroupK              = 64;
    static constexpr bool kPackedWord8        = true;
    static constexpr bool kSingleValuePerLane = false;

    // Which stored divisor this row was quantised against. Taken once per row, not per group.
    __device__ static __forceinline__ float row_coefficient(const SparseMoePlanes& planes,
                                                            int row) {
        return __frcp_rn(planes.divisors[row / planes.divisor_rows]);
    }

    template <int K>
    __device__ static __forceinline__ void load_eight(const SparseMoePlanes& planes, int row,
                                                      int group, int lane_in_group, float row_scale,
                                                      float (&weights)[8]) {
        const std::uint32_t packed = *reinterpret_cast<const std::uint32_t*>(
            planes.codes + static_cast<std::int64_t>(row) * (K / 2) + group * 32 +
            lane_in_group * 4);
        const std::uint8_t scale =
            planes.scales[nvfp4_scale_byte<K / 64>(row, group * 4 + (lane_in_group >> 1))];
        const float coefficient = decode_nvfp4_e4m3(scale) * row_scale;
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const float2 code =
                decode_nvfp4_e2m1x2(static_cast<std::uint8_t>(packed >> (8 * pair)));
            weights[pair * 2]     = code.x * coefficient;
            weights[pair * 2 + 1] = code.y * coefficient;
        }
    }
};

// A codec declares exactly one lane ownership. The trap below catches a codec that declares
// neither, which would otherwise compile and reduce a zero accumulator.
template <class>
inline constexpr bool kNoLaneOwnership = false;

template <class Codec, int K>
__device__ __forceinline__ void dot_two_rows(const SparseMoePlanes& planes, int row0, int row1,
                                             const __nv_bfloat16* x, int k_begin, int k_end,
                                             float& result0, float& result1) {
    const int lane        = static_cast<int>(threadIdx.x) & 31;
    float acc0            = 0.0f;
    float acc1            = 0.0f;
    const int first_group = k_begin / Codec::kGroupK;
    const int last_group  = k_end / Codec::kGroupK;
    if constexpr (Codec::kPackedWord8) {
        const int lane_group    = lane >> 3;
        const int lane_in_group = lane & 7;
        const float scale0      = Codec::row_coefficient(planes, row0);
        const float scale1      = Codec::row_coefficient(planes, row1);
        for (int group_base = first_group; group_base < last_group; group_base += 4) {
            const int group = group_base + lane_group;
            float weights0[8];
            float weights1[8];
            Codec::template load_eight<K>(planes, row0, group, lane_in_group, scale0, weights0);
            Codec::template load_eight<K>(planes, row1, group, lane_in_group, scale1, weights1);
            const uint4 input     = load_vec<uint4>(x + group * Codec::kGroupK + lane_in_group * 8);
            const float2 x0       = bf16x2_bits_to_float2(input.x);
            const float2 x1       = bf16x2_bits_to_float2(input.y);
            const float2 x2       = bf16x2_bits_to_float2(input.z);
            const float2 x3       = bf16x2_bits_to_float2(input.w);
            const float values[8] = {x0.x, x0.y, x1.x, x1.y, x2.x, x2.y, x3.x, x3.y};
#pragma unroll
            for (int item = 0; item < 8; ++item) {
                acc0 = fmaf(weights0[item], values[item], acc0);
                acc1 = fmaf(weights1[item], values[item], acc1);
            }
        }
    } else if constexpr (Codec::kSingleValuePerLane) {
        for (int group = first_group; group < last_group; ++group) {
            const float w0 = Codec::template load_one<K>(planes, row0, group, lane);
            const float w1 = Codec::template load_one<K>(planes, row1, group, lane);
            const float xv = __bfloat162float(x[group * Codec::kGroupK + lane]);
            acc0           = fmaf(w0, xv, acc0);
            acc1           = fmaf(w1, xv, acc1);
        }
    } else {
        static_assert(kNoLaneOwnership<Codec>, "dot_two_rows: codec declares no lane ownership");
    }
    result0 = warp_reduce_sum(acc0);
    result1 = warp_reduce_sum(acc1);
}

template <class RoutedCodec, class SharedCodec>
__global__ void sparse_moe_d3_nine_warp_kernel(const __nv_bfloat16* __restrict__ x,
                                               const int* __restrict__ ids, SparseMoePlanes routed,
                                               SparseMoePlanes shared, float* __restrict__ act) {
    __shared__ __align__(16) __nv_bfloat16 x_shared[kHidden];
    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    if (tid == 0) { pdl::trigger_dependents(); }
    if (tid < 256) { store_vec(x_shared + tid * 8, load_vec<uint4>(x + tid * 8)); }
    __syncthreads();

    const int j = static_cast<int>(blockIdx.x);
    float gate  = 0.0f;
    float up    = 0.0f;
    if (warp < kTopK) {
        pdl::wait_for_dependencies();
        const int expert   = ids[warp];
        const int row_base = expert * 1024;
        dot_two_rows<RoutedCodec, kHidden>(routed, row_base + j, row_base + kIntermediate + j,
                                           x_shared, 0, kHidden, gate, up);
    } else {
        dot_two_rows<SharedCodec, kHidden>(shared, j, kIntermediate + j, x_shared, 0, kHidden, gate,
                                           up);
    }
    if (lane == 0) { act[static_cast<std::int64_t>(warp) * kIntermediate + j] = silu(gate) * up; }
}

template <class RoutedCodec, class SharedCodec, int PathsPerBlock, bool Adaptive>
__global__ void sparse_moe_d3_path_tiled_kernel(const __nv_bfloat16* __restrict__ x,
                                                const int* __restrict__ token_ids,
                                                SparseMoePlanes routed, SparseMoePlanes shared,
                                                float* __restrict__ token_activations, int tokens,
                                                const int* __restrict__ adaptive_route_jobs) {
    // Three path CTAs per token/output row expose enough blocks for the 170-SM target and keep the
    // heavier shared Q8 path from holding eight completed routed warps resident.
    static_assert(PathsPerBlock > 0 && (kTopK + 1) % PathsPerBlock == 0);
    constexpr int kPathBlocks = (kTopK + 1) / PathsPerBlock;
    __shared__ __align__(16) __nv_bfloat16 x_shared[kHidden];
    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    if constexpr (Adaptive) {
        if (*adaptive_route_jobs >= 0) { return; }
    }
    const int total_work = tokens * kPathBlocks * kIntermediate;
    const int first_work =
        Adaptive ? static_cast<int>(blockIdx.x)
                 : static_cast<int>(blockIdx.y) * kIntermediate + static_cast<int>(blockIdx.x);
    const int work_stride = Adaptive ? static_cast<int>(gridDim.x) : total_work;
    for (int work = first_work; work < total_work; work += work_stride) {
        const int token_path = work / kIntermediate;
        const int j          = work - token_path * kIntermediate;
        const int token      = token_path / kPathBlocks;
        const int path_block = token_path - token * kPathBlocks;
        const int path       = path_block * PathsPerBlock + warp;
        const auto* input    = x + static_cast<std::int64_t>(token) * kHidden;
        for (int vector = tid; vector < kHidden / 8; vector += static_cast<int>(blockDim.x)) {
            store_vec(x_shared + vector * 8, load_vec<uint4>(input + vector * 8));
        }
        __syncthreads();

        constexpr int kRouterPartitions = 4;
        const int activation_begin      = (token * (kTopK + 1) + path) * kIntermediate;
        // Routed paths need S2's ids. A shared path is independent only when its output lies
        // beyond the partial-score prefix that S2 may still read from the lifetime-unioned scratch.
        const bool must_wait_for_s2 =
            path < kTopK || activation_begin < tokens * kRouterRows * kRouterPartitions;
        if constexpr (!Adaptive) {
            if (must_wait_for_s2) { pdl::wait_for_dependencies(); }
        }
        float gate = 0.0f;
        float up   = 0.0f;
        if (path < kTopK) {
            const int expert   = token_ids[token * kTopK + path];
            const int row_base = expert * (2 * kIntermediate);
            dot_two_rows<RoutedCodec, kHidden>(routed, row_base + j, row_base + kIntermediate + j,
                                               x_shared, 0, kHidden, gate, up);
        } else {
            dot_two_rows<SharedCodec, kHidden>(shared, j, kIntermediate + j, x_shared, 0, kHidden,
                                               gate, up);
        }
        if (lane == 0) {
            token_activations[(static_cast<std::int64_t>(token) * (kTopK + 1) + path) *
                                  kIntermediate +
                              j] = silu(gate) * up;
        }
        if constexpr (Adaptive) { __syncthreads(); }
    }
}

template <class Codec, int K, int Rows>
__device__ __forceinline__ void dot_fp32_rows(const SparseMoePlanes& planes, int row_base,
                                              const float* x, int first_group, int last_group,
                                              float (&result)[Rows]) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    float acc[Rows];
#pragma unroll
    for (int row = 0; row < Rows; ++row) { acc[row] = 0.0f; }
    if constexpr (Codec::kPackedWord8) {
        // The same eight-value lane ownership as D3, over the FP32 SwiGLU result rather than the
        // BF16 hidden state. Each row decodes from its own stored planes before the FP32
        // accumulation.
        const int lane_group    = lane >> 3;
        const int lane_in_group = lane & 7;
        float row_scale[Rows];
#pragma unroll
        for (int row = 0; row < Rows; ++row) {
            row_scale[row] = Codec::row_coefficient(planes, row_base + row);
        }
        for (int group_base = first_group; group_base < last_group; group_base += 4) {
            const int group = group_base + lane_group;
            const float4 x0 = load_vec<float4>(x + group * Codec::kGroupK + lane_in_group * 8);
            const float4 x1 = load_vec<float4>(x + group * Codec::kGroupK + lane_in_group * 8 + 4);
            const float values[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
            for (int row = 0; row < Rows; ++row) {
                float weights[8];
                Codec::template load_eight<K>(planes, row_base + row, group, lane_in_group,
                                              row_scale[row], weights);
#pragma unroll
                for (int item = 0; item < 8; ++item) {
                    acc[row] = fmaf(weights[item], values[item], acc[row]);
                }
            }
        }
    } else if constexpr (Codec::kSingleValuePerLane) {
        // Two values a lane here rather than one: the FP32 input is half the width of the BF16
        // hidden state, so the same group needs half the lanes.
        if (lane < Codec::kGroupK / 2) {
            for (int group = first_group; group < last_group; ++group) {
                const int k     = group * Codec::kGroupK + lane * 2;
                const float2 xv = load_vec<float2>(x + k);
#pragma unroll
                for (int row = 0; row < Rows; ++row) {
                    float w0, w1;
                    Codec::template load_pair<K>(planes, row_base + row, group, lane, w0, w1);
                    acc[row] = fmaf(w0, xv.x, acc[row]);
                    acc[row] = fmaf(w1, xv.y, acc[row]);
                }
            }
        }
    } else {
        static_assert(kNoLaneOwnership<Codec>, "dot_fp32_rows: codec declares no lane ownership");
    }
#pragma unroll
    for (int row = 0; row < Rows; ++row) { result[row] = warp_reduce_sum(acc[row]); }
}

template <class RoutedCodec, class SharedCodec, int Rows>
__global__ void sparse_moe_d4_nine_warp_kernel(
    const int* __restrict__ ids, const float* __restrict__ alpha,
    const float* __restrict__ shared_scale, const float* __restrict__ act, SparseMoePlanes routed,
    SparseMoePlanes shared, __nv_bfloat16* __restrict__ destination,
    const char* __restrict__ prefetch_data, unsigned long long prefetch_bytes) {
    __shared__ float paths[kTopK + 1][Rows];
    pdl::wait_for_dependencies();
    const int warp     = static_cast<int>(threadIdx.x) >> 5;
    const int lane     = static_cast<int>(threadIdx.x) & 31;
    const int row_base = static_cast<int>(blockIdx.x) * Rows;
    if (warp < kTopK) {
        const int expert = ids[warp];
        float dot[Rows];
        dot_fp32_rows<RoutedCodec, kIntermediate, Rows>(
            routed, expert * kHidden + row_base,
            act + static_cast<std::int64_t>(warp) * kIntermediate, 0,
            kIntermediate / RoutedCodec::kGroupK, dot);
        if (lane == 0) {
#pragma unroll
            for (int row = 0; row < Rows; ++row) { paths[warp][row] = alpha[warp] * dot[row]; }
        }
    } else {
        float dot[Rows];
        dot_fp32_rows<SharedCodec, kIntermediate, Rows>(
            shared, row_base, act + static_cast<std::int64_t>(kTopK) * kIntermediate, 0,
            kIntermediate / SharedCodec::kGroupK, dot);
        if (lane == 0) {
#pragma unroll
            for (int row = 0; row < Rows; ++row) { paths[kTopK][row] = *shared_scale * dot[row]; }
        }
    }
    if (prefetch_data != nullptr) {
        // Fire-and-forget L2 warmup of the next consumer's weight codes,
        // issued BEFORE the block barrier so it hides behind the slowest warp instead
        // of extending the grid tail. Pure cache hint; no values, no addition order.
        const unsigned long long offset =
            (static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x) * 128ull;
        if (offset < prefetch_bytes) {
            asm volatile("prefetch.global.L2 [%0];" ::"l"(prefetch_data + offset));
        }
    }
    __syncthreads();
    if (warp == 0 && lane < Rows) {
        float value = __bfloat162float(destination[row_base + lane]);
#pragma unroll
        for (int path = 0; path < kTopK + 1; ++path) { value += paths[path][lane]; }
        destination[row_base + lane] = __float2bfloat16_rn(value);
    }
}

template <class RoutedCodec, class SharedCodec, int Rows, bool Adaptive>
__global__ void
sparse_moe_d4_token_kernel(const int* __restrict__ token_ids, const float* __restrict__ token_alpha,
                           const float* __restrict__ shared_scale,
                           const float* __restrict__ token_activations, SparseMoePlanes routed,
                           SparseMoePlanes shared, __nv_bfloat16* __restrict__ destination,
                           int tokens, const int* __restrict__ adaptive_route_jobs) {
    // Token is a grid dimension rather than an in-CTA serial loop. Rows lets one routed-weight
    // stream serve adjacent outputs while retaining the deterministic rank-order FP32 epilogue.
    __shared__ float paths[kTopK + 1][Rows];
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    if constexpr (Adaptive) {
        if (*adaptive_route_jobs >= 0) { return; }
    }
    constexpr int kRowBlocks = kHidden / Rows;
    const int total_work     = tokens * kRowBlocks;
    const int first_work =
        Adaptive ? static_cast<int>(blockIdx.x)
                 : static_cast<int>(blockIdx.y) * kRowBlocks + static_cast<int>(blockIdx.x);
    const int work_stride = Adaptive ? static_cast<int>(gridDim.x) : total_work;
    for (int work = first_work; work < total_work; work += work_stride) {
        const int token     = work / kRowBlocks;
        const int row_block = work - token * kRowBlocks;
        const int row_base  = row_block * Rows;
        const float* act =
            token_activations + static_cast<std::int64_t>(token) * (kTopK + 1) * kIntermediate;
        if (warp < kTopK) {
            const int expert = token_ids[token * kTopK + warp];
            float dot[Rows];
            dot_fp32_rows<RoutedCodec, kIntermediate, Rows>(
                routed, expert * kHidden + row_base,
                act + static_cast<std::int64_t>(warp) * kIntermediate, 0,
                kIntermediate / RoutedCodec::kGroupK, dot);
            if (lane == 0) {
#pragma unroll
                for (int row = 0; row < Rows; ++row) {
                    paths[warp][row] = token_alpha[token * kTopK + warp] * dot[row];
                }
            }
        } else {
            float dot[Rows];
            dot_fp32_rows<SharedCodec, kIntermediate, Rows>(
                shared, row_base, act + static_cast<std::int64_t>(kTopK) * kIntermediate, 0,
                kIntermediate / SharedCodec::kGroupK, dot);
            if (lane == 0) {
#pragma unroll
                for (int row = 0; row < Rows; ++row) {
                    paths[kTopK][row] = shared_scale[token] * dot[row];
                }
            }
        }
        __syncthreads();
        if (warp == 0 && lane < Rows) {
            __nv_bfloat16* output =
                destination + static_cast<std::int64_t>(token) * kHidden + row_base + lane;
            float value = __bfloat162float(*output);
#pragma unroll
            for (int path = 0; path < kTopK + 1; ++path) { value += paths[path][lane]; }
            *output = __float2bfloat16_rn(value);
        }
        if constexpr (Adaptive) { __syncthreads(); }
    }
}

// How many bytes of the shared-expert down codes the D1 prefetch may touch. A four-bit plane is
// half the size of an eight-bit one, and walking the eight-bit length over it addresses memory the
// artifact never allocated.
unsigned long long shared_down_code_bytes(QType qtype) {
    const unsigned long long elements = static_cast<unsigned long long>(kHidden) * kIntermediate;
    return qtype == QType::NVFP4 ? elements / 2 : elements;
}

void launch_d1(const Tensor& x, const SparseMoeWeights& weights,
               const SparseMoeDecodeWorkspace& workspace, cudaStream_t stream) {
    sparse_moe_d1_kernel<<<kRouterRows, kD1Warps * 32, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const __nv_bfloat16*>(weights.router_shared_gate.qdata),
        static_cast<float*>(workspace.scratch.data),
        static_cast<const char*>(weights.shared_down.qdata),
        shared_down_code_bytes(weights.shared_down.qtype));
    CUDA_CHECK(cudaGetLastError());
}

SparseMoePlanes matrix_planes(const Weight& weight) {
    // Only NVFP4 stores its block scale as a quotient of a divisor, and only it has a plane of
    // them; for the row-split codecs the field is not part of the representation at all. The row
    // stride is one here so that a row's divisor index is well formed whatever the codec, and the
    // codecs without divisors never read it.
    return {static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.qhigh),
            static_cast<const std::uint8_t*>(weight.scales),
            static_cast<const float*>(weight.weight_divisors),
            weight.weight_divisor_rows > 0 ? weight.weight_divisor_rows : 1};
}

template <class RoutedCodec, class SharedCodec>
void launch_d3_dependent_codec(const Tensor& x, const SparseMoeWeights& weights,
                               const SparseMoeDecodeWorkspace& workspace, cudaStream_t stream) {
    CUDA_CHECK(pdl::launch_dependent(
        {dim3(kIntermediate), dim3(9 * 32), 0, stream},
        sparse_moe_d3_nine_warp_kernel<RoutedCodec, SharedCodec>,
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const int*>(workspace.ids.data),
        matrix_planes(weights.routed_gate_up), matrix_planes(weights.shared_gate_up),
        static_cast<float*>(workspace.scratch.data)));
}

void launch_d2_d3(const Tensor& x, const SparseMoeWeights& weights,
                  const SparseMoeDecodeWorkspace& workspace, cudaStream_t stream) {
    const auto* scores = static_cast<const float*>(workspace.scratch.data);
    auto* ids          = static_cast<int*>(workspace.ids.data);
    auto* alpha        = static_cast<float*>(workspace.alpha.data);
    auto* shared_scale = static_cast<float*>(workspace.shared_scale.data);
    sparse_moe_d2_warp_kernel<<<1, 32, 0, stream>>>(scores, ids, alpha, shared_scale);
    CUDA_CHECK(cudaGetLastError());

    switch (weights.routed_gate_up.qtype) {
    case QType::Q4_G64_FP16:
        launch_d3_dependent_codec<Q4Codec, Q8Codec>(x, weights, workspace, stream);
        return;
    case QType::Q8_G32_FP16:
        launch_d3_dependent_codec<Q8Codec, Q8Codec>(x, weights, workspace, stream);
        return;
    case QType::NVFP4:
        launch_d3_dependent_codec<Nvfp4Codec, Nvfp4Codec>(x, weights, workspace, stream);
        return;
    default:
        throw std::invalid_argument("sparse_moe: unsupported D3 codec");
    }
}

// Hidden rows one D4 block takes. NVFP4 interleaves rows inside a 512-byte scale tile so that rows
// r and r+1 share one 32-byte sector: taking both halves the sectors the kernel reads from L2, 398k
// to 326k, and the kernel from 10.2 us to 8.0. A third and fourth row start the next sector and buy
// nothing. The row-split planes store a scale per row, so a second row only costs them grid.
template <class Codec>
inline constexpr int kDecodeDownRows = 1;
template <>
inline constexpr int kDecodeDownRows<Nvfp4Codec> = 2;

template <class RoutedCodec, class SharedCodec>
void launch_d4_dependent_codec(const SparseMoeWeights& weights, Tensor& destination,
                               const SparseMoeDecodeWorkspace& workspace, cudaStream_t stream,
                               const void* prefetch_data, std::size_t prefetch_bytes) {
    constexpr int kRows = kDecodeDownRows<RoutedCodec>;
    CUDA_CHECK(pdl::launch_dependent(
        {dim3(kHidden / kRows), dim3(9 * 32), 0, stream},
        sparse_moe_d4_nine_warp_kernel<RoutedCodec, SharedCodec, kRows>,
        static_cast<const int*>(workspace.ids.data),
        static_cast<const float*>(workspace.alpha.data),
        static_cast<const float*>(workspace.shared_scale.data),
        static_cast<const float*>(workspace.scratch.data), matrix_planes(weights.routed_down),
        matrix_planes(weights.shared_down), static_cast<__nv_bfloat16*>(destination.data),
        static_cast<const char*>(prefetch_data), static_cast<unsigned long long>(prefetch_bytes)));
}

void launch_d4_dependent(const SparseMoeWeights& weights, Tensor& destination,
                         const SparseMoeDecodeWorkspace& workspace, cudaStream_t stream,
                         const void* prefetch_data, std::size_t prefetch_bytes) {
    switch (weights.routed_down.qtype) {
    case QType::Q5_G64_FP16:
        launch_d4_dependent_codec<Q5Codec, Q8Codec>(weights, destination, workspace, stream,
                                                    prefetch_data, prefetch_bytes);
        return;
    case QType::Q6_G64_FP16:
        launch_d4_dependent_codec<Q6Codec, Q8Codec>(weights, destination, workspace, stream,
                                                    prefetch_data, prefetch_bytes);
        return;
    case QType::Q8_G32_FP16:
        launch_d4_dependent_codec<Q8Codec, Q8Codec>(weights, destination, workspace, stream,
                                                    prefetch_data, prefetch_bytes);
        return;
    case QType::NVFP4:
        launch_d4_dependent_codec<Nvfp4Codec, Nvfp4Codec>(weights, destination, workspace, stream,
                                                          prefetch_data, prefetch_bytes);
        return;
    default:
        throw std::invalid_argument("sparse_moe: unsupported D4 codec");
    }
}

template <class RoutedCodec, class SharedCodec, int PathsPerBlock, bool Adaptive>
void launch_d3_small_t_paths(const Tensor& x, const SparseMoeWeights& weights, const int* token_ids,
                             float* token_activations, std::int32_t tokens, cudaStream_t stream,
                             const int* adaptive_route_jobs) {
    constexpr int kPathBlocks    = (kTopK + 1) / PathsPerBlock;
    const auto* input            = static_cast<const __nv_bfloat16*>(x.data);
    const SparseMoePlanes routed = matrix_planes(weights.routed_gate_up);
    const SparseMoePlanes shared = matrix_planes(weights.shared_gate_up);
    if constexpr (Adaptive) {
        sparse_moe_d3_path_tiled_kernel<RoutedCodec, SharedCodec, PathsPerBlock, true>
            <<<kAdaptiveD3Blocks, PathsPerBlock * 32, 0, stream>>>(
                input, token_ids, routed, shared, token_activations, tokens, adaptive_route_jobs);
        CUDA_CHECK(cudaGetLastError());
    } else {
        CUDA_CHECK(pdl::launch_dependent(
            {dim3(kIntermediate, tokens * kPathBlocks), dim3(PathsPerBlock * 32), 0, stream},
            sparse_moe_d3_path_tiled_kernel<RoutedCodec, SharedCodec, PathsPerBlock, false>, input,
            token_ids, routed, shared, token_activations, tokens, nullptr));
    }
}

template <class RoutedCodec, class SharedCodec, bool Adaptive>
void launch_d3_small_t_codec(const Tensor& x, const SparseMoeWeights& weights, const int* token_ids,
                             float* token_activations, std::int32_t tokens,
                             SparseMoeSmallTD3Schedule schedule, cudaStream_t stream,
                             const int* adaptive_route_jobs) {
    switch (schedule) {
    case SparseMoeSmallTD3Schedule::Paths1:
        launch_d3_small_t_paths<RoutedCodec, SharedCodec, 1, Adaptive>(
            x, weights, token_ids, token_activations, tokens, stream, adaptive_route_jobs);
        return;
    case SparseMoeSmallTD3Schedule::Paths3:
        launch_d3_small_t_paths<RoutedCodec, SharedCodec, 3, Adaptive>(
            x, weights, token_ids, token_activations, tokens, stream, adaptive_route_jobs);
        return;
    case SparseMoeSmallTD3Schedule::Paths9:
        launch_d3_small_t_paths<RoutedCodec, SharedCodec, 9, Adaptive>(
            x, weights, token_ids, token_activations, tokens, stream, adaptive_route_jobs);
        return;
    }
    throw std::logic_error("sparse_moe: unknown small-T D3 schedule");
}

template <class RoutedCodec, class SharedCodec, int Rows, bool Adaptive>
void launch_d4_small_t_rows(const SparseMoeWeights& weights, Tensor& destination,
                            const int* token_ids, const float* token_alpha,
                            const float* shared_scale, const float* token_activations,
                            std::int32_t tokens, cudaStream_t stream,
                            const int* adaptive_route_jobs) {
    const dim3 grid = Adaptive ? dim3(kAdaptiveD4Blocks) : dim3(kHidden / Rows, tokens);
    sparse_moe_d4_token_kernel<RoutedCodec, SharedCodec, Rows, Adaptive>
        <<<grid, 9 * 32, 0, stream>>>(
            token_ids, token_alpha, shared_scale, token_activations,
            matrix_planes(weights.routed_down), matrix_planes(weights.shared_down),
            static_cast<__nv_bfloat16*>(destination.data), tokens, adaptive_route_jobs);
    CUDA_CHECK(cudaGetLastError());
}

template <class RoutedCodec, class SharedCodec, bool Adaptive>
void launch_d4_small_t_codec(const SparseMoeWeights& weights, Tensor& destination,
                             const int* token_ids, const float* token_alpha,
                             const float* shared_scale, const float* token_activations,
                             std::int32_t tokens, SparseMoeSmallTD4Schedule schedule,
                             cudaStream_t stream, const int* adaptive_route_jobs) {
    switch (schedule) {
    case SparseMoeSmallTD4Schedule::Rows1:
        launch_d4_small_t_rows<RoutedCodec, SharedCodec, 1, Adaptive>(
            weights, destination, token_ids, token_alpha, shared_scale, token_activations, tokens,
            stream, adaptive_route_jobs);
        return;
    case SparseMoeSmallTD4Schedule::Rows2:
        launch_d4_small_t_rows<RoutedCodec, SharedCodec, 2, Adaptive>(
            weights, destination, token_ids, token_alpha, shared_scale, token_activations, tokens,
            stream, adaptive_route_jobs);
        return;
    case SparseMoeSmallTD4Schedule::Rows4:
        launch_d4_small_t_rows<RoutedCodec, SharedCodec, 4, Adaptive>(
            weights, destination, token_ids, token_alpha, shared_scale, token_activations, tokens,
            stream, adaptive_route_jobs);
        return;
    }
    throw std::logic_error("sparse_moe: unknown small-T D4 schedule");
}

} // namespace

void sparse_moe_decode_launch_d3_small_t(const Tensor& x, const SparseMoeWeights& weights,
                                         const int* token_ids, float* token_activations,
                                         std::int32_t tokens, SparseMoeSmallTD3Schedule schedule,
                                         cudaStream_t stream, const int* adaptive_route_jobs) {
    switch (weights.routed_gate_up.qtype) {
    case QType::Q4_G64_FP16:
        if (adaptive_route_jobs == nullptr) {
            launch_d3_small_t_codec<Q4Codec, Q8Codec, false>(
                x, weights, token_ids, token_activations, tokens, schedule, stream, nullptr);
        } else {
            launch_d3_small_t_codec<Q4Codec, Q8Codec, true>(x, weights, token_ids,
                                                            token_activations, tokens, schedule,
                                                            stream, adaptive_route_jobs);
        }
        return;
    case QType::Q8_G32_FP16:
        launch_d3_small_t_codec<Q8Codec, Q8Codec, false>(x, weights, token_ids, token_activations,
                                                         tokens, schedule, stream, nullptr);
        return;
    case QType::NVFP4:
        launch_d3_small_t_codec<Nvfp4Codec, Nvfp4Codec, false>(
            x, weights, token_ids, token_activations, tokens, schedule, stream, nullptr);
        return;
    default:
        throw std::invalid_argument("sparse_moe: unsupported small-T D3 codec");
    }
}

void sparse_moe_decode_launch_d4_small_t(const SparseMoeWeights& weights, Tensor& destination,
                                         const int* token_ids, const float* token_alpha,
                                         const float* shared_scale, const float* token_activations,
                                         std::int32_t tokens, SparseMoeSmallTD4Schedule schedule,
                                         cudaStream_t stream, const int* adaptive_route_jobs) {
    switch (weights.routed_down.qtype) {
    case QType::Q5_G64_FP16:
        if (adaptive_route_jobs == nullptr) {
            launch_d4_small_t_codec<Q5Codec, Q8Codec, false>(
                weights, destination, token_ids, token_alpha, shared_scale, token_activations,
                tokens, schedule, stream, nullptr);
        } else {
            launch_d4_small_t_codec<Q5Codec, Q8Codec, true>(
                weights, destination, token_ids, token_alpha, shared_scale, token_activations,
                tokens, schedule, stream, adaptive_route_jobs);
        }
        return;
    case QType::Q6_G64_FP16:
        if (adaptive_route_jobs == nullptr) {
            launch_d4_small_t_codec<Q6Codec, Q8Codec, false>(
                weights, destination, token_ids, token_alpha, shared_scale, token_activations,
                tokens, schedule, stream, nullptr);
        } else {
            launch_d4_small_t_codec<Q6Codec, Q8Codec, true>(
                weights, destination, token_ids, token_alpha, shared_scale, token_activations,
                tokens, schedule, stream, adaptive_route_jobs);
        }
        return;
    case QType::Q8_G32_FP16:
        launch_d4_small_t_codec<Q8Codec, Q8Codec, false>(
            weights, destination, token_ids, token_alpha, shared_scale, token_activations, tokens,
            schedule, stream, nullptr);
        return;
    case QType::NVFP4:
        launch_d4_small_t_codec<Nvfp4Codec, Nvfp4Codec, false>(
            weights, destination, token_ids, token_alpha, shared_scale, token_activations, tokens,
            schedule, stream, nullptr);
        return;
    default:
        throw std::invalid_argument("sparse_moe: unsupported small-T D4 codec");
    }
}

void sparse_moe_decode_launch(const Tensor& x, const SparseMoeWeights& weights, Tensor& destination,
                              const SparseMoeDecodePlan& plan,
                              const SparseMoeDecodeWorkspace& workspace, cudaStream_t stream) {
    launch_d1(x, weights, workspace, stream);
    launch_d2_d3(x, weights, workspace, stream);
    launch_d4_dependent(weights, destination, workspace, stream, plan.next_weight_prefetch,
                        plan.next_weight_prefetch_bytes);
}

} // namespace ninfer::ops::detail
