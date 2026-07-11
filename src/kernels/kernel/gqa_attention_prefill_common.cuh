#pragma once

// Fixed Qwen3.6 GQA geometry and leaf PTX helpers shared by the independently tuned
// BF16 and INT8 prompt kernels. This file deliberately owns no staging policy,
// shared-memory arena, warp schedule, or kernel body.

#include "kernels/common/math.cuh"
#include "kernels/common/mma.cuh"
#include "kernels/common/warp.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace qus::kernels {

inline constexpr int kGqaPrefillHeadDim   = 256;
inline constexpr int kGqaPrefillQHeads    = 24;
inline constexpr int kGqaPrefillKVHeads   = 4;
inline constexpr int kGqaPrefillGroupSize = 6;

inline constexpr int kGqaPrefillBr        = 64;
inline constexpr int kGqaPrefillBc        = 64;
inline constexpr int kGqaPrefillThreads   = 128;
inline constexpr int kGqaPrefillSmemBytes = (kGqaPrefillBr + 2 * kGqaPrefillBc) *
                                            kGqaPrefillHeadDim *
                                            static_cast<int>(sizeof(__nv_bfloat16));

__device__ __forceinline__ std::int64_t gqa_prefill_cache_index(int kv_head, int d, int position,
                                                                int padded_context) {
    return static_cast<std::int64_t>(d) + static_cast<std::int64_t>(kGqaPrefillHeadDim) *
                                              (static_cast<std::int64_t>(position) +
                                               static_cast<std::int64_t>(padded_context) * kv_head);
}

__device__ __forceinline__ std::int64_t gqa_prefill_q_index(int q_head, int d, int token) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kGqaPrefillHeadDim) *
               (static_cast<std::int64_t>(q_head) +
                static_cast<std::int64_t>(kGqaPrefillQHeads) * token);
}

// XOR-swizzled b16 element address. INT8 operands use the same layout by packing
// two consecutive signed bytes into each b16 lane before ldmatrix.
__device__ __forceinline__ int gqa_prefill_swz(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

__device__ __forceinline__ unsigned gqa_prefill_swz_addr(unsigned lane_base, unsigned ck,
                                                         unsigned as, unsigned r) {
    return lane_base + ((ck | as) ^ r);
}

} // namespace qus::kernels
