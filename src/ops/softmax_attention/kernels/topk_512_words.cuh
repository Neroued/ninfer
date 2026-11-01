#pragma once

#include "ops/softmax_attention/kernels/topk_512.cuh"

namespace ninfer::ops::detail {

struct Top512DenseIds {
    static constexpr bool dense = true;
    int first;

    __device__ unsigned word(int item) const {
        return 0xffffffffu - unsigned(first + item * 256 + threadIdx.x);
    }
};

template <int Items>
struct Top512GatherIds {
    static constexpr bool dense = false;
    unsigned values[Items];

    __device__ unsigned word(int item) const { return values[item]; }
};

struct Top512WordStorage {
    union {
        int histogram[2][256];
        unsigned differences[16];
        Top512Scan::TempStorage scan;
    } temp;

    unsigned high_prefix, low_prefix, high_mask, low_mask;
    std::uint64_t anchor;
    int high_shift, low_shift, remaining, selected_count;
    int warp_sums[8];
};

__device__ __forceinline__ void top512_word_pivot(Top512WordStorage& s, int shift, bool high) {
    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    // Every warp must read the old rank before the pivot's writer updates it.
    const int remaining = s.remaining;
    const int histogram = s.temp.histogram[0][tid] + s.temp.histogram[1][tid];
    int inclusive       = histogram;
#pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
        const int value = __shfl_down_sync(0xffffffffu, inclusive, offset);
        if (lane + offset < 32) inclusive += value;
    }
    if (lane == 0) s.warp_sums[warp] = inclusive;
    __syncthreads();
    int above = inclusive - histogram;
#pragma unroll
    for (int w = 1; w < 8; ++w)
        if (w > warp) above += s.warp_sums[w];
    if (above < remaining && above + histogram >= remaining) {
        if (high) {
            s.high_prefix |= unsigned(tid) << shift;
            s.high_mask |= 255u << shift;
        } else {
            s.low_prefix |= unsigned(tid) << shift;
            s.low_mask |= 255u << shift;
        }
        s.remaining      = remaining - above;
        s.selected_count = histogram;
    }
    __syncthreads();
}

template <bool High, int Items, class Ids>
__device__ __forceinline__ bool top512_word_refine(unsigned (&scores)[Items], const Ids& ids,
                                                   Top512WordStorage& s) {
    const int tid = threadIdx.x;
    for (int shift = High ? s.high_shift : s.low_shift; shift >= 0; shift -= 8) {
        s.temp.histogram[0][tid] = 0;
        s.temp.histogram[1][tid] = 0;
        __syncthreads();
        const unsigned prefix = High ? s.high_prefix : s.low_prefix;
        const unsigned mask   = High ? s.high_mask : s.low_mask;
#pragma unroll
        for (int j = 0; j < Items; ++j) {
            const unsigned value = High ? scores[j] : ids.word(j);
            if (scores[j] && (value & mask) == prefix && (High || scores[j] == s.high_prefix))
                atomicAdd(&s.temp.histogram[tid / 32 % 2][(value >> shift) & 255u], 1);
        }
        __syncthreads();
        top512_word_pivot(s, shift, High);
        if (s.remaining == s.selected_count) return true;
    }
    return false;
}

// Exact score/id selection, refining 32-bit score words before tied IDs. Dense
// partitions derive IDs on demand instead of keeping another Items registers.
// The first count entries are valid; padded scores are zero, below any valid
// ordered_score_bits value. The 64-bit total order and output encoding are unchanged.
template <int Items, class Ids, class Store>
__device__ void select_top512_words(unsigned (&scores)[Items], const Ids& ids, int count,
                                    Top512WordStorage& s, Store store) {
    const int tid = threadIdx.x;
    if (count <= 512) {
#pragma unroll
        for (int j = 0; j < Items; ++j)
            if (j * 256 + tid < count)
                store(j * 256 + tid, (std::uint64_t(scores[j]) << 32) | ids.word(j));
        return;
    }
    if (tid == 0) s.anchor = (std::uint64_t(scores[0]) << 32) | ids.word(0);
    __syncthreads();
    const auto anchor        = s.anchor;
    unsigned high_difference = 0, low_difference = 0;
#pragma unroll
    for (int j = 0; j < Items; ++j)
        if (scores[j]) {
            high_difference |= scores[j] ^ unsigned(anchor >> 32);
            if constexpr (!Ids::dense) low_difference |= ids.word(j) ^ unsigned(anchor);
        }
    unsigned low  = __reduce_or_sync(0xffffffffu, low_difference);
    unsigned high = __reduce_or_sync(0xffffffffu, high_difference);
    if (tid % 32 == 0) {
        s.temp.differences[tid / 32]     = low;
        s.temp.differences[8 + tid / 32] = high;
    }
    __syncthreads();
    if (tid < 32) {
        if constexpr (Ids::dense)
            low = unsigned(ids.first) ^ unsigned(ids.first + count - 1);
        else
            low = __reduce_or_sync(0xffffffffu, tid < 8 ? s.temp.differences[tid] : 0u);
        high = __reduce_or_sync(0xffffffffu, tid < 8 ? s.temp.differences[8 + tid] : 0u);
        if (tid == 0) {
            s.high_shift     = high ? (31 - int(__clz(high))) / 8 * 8 : -1;
            s.low_shift      = low ? (31 - int(__clz(low))) / 8 * 8 : -1;
            s.high_mask      = high ? (s.high_shift == 24 ? 0u : ~0u << (s.high_shift + 8)) : ~0u;
            s.low_mask       = low ? (s.low_shift == 24 ? 0u : ~0u << (s.low_shift + 8)) : ~0u;
            s.high_prefix    = unsigned(anchor >> 32) & s.high_mask;
            s.low_prefix     = unsigned(anchor) & s.low_mask;
            s.remaining      = 512;
            s.selected_count = count;
        }
    }
    __syncthreads();
    const bool done = top512_word_refine<true>(scores, ids, s);
    if (!done) top512_word_refine<false>(scores, ids, s);
    const auto threshold = (std::uint64_t(s.high_prefix) << 32) | (done ? 0u : s.low_prefix);
    int kept             = 0;
#pragma unroll
    for (int j = 0; j < Items; ++j)
        kept += ((std::uint64_t(scores[j]) << 32) | ids.word(j)) >= threshold;
    int rank;
    Top512Scan(s.temp.scan).ExclusiveSum(kept, rank);
#pragma unroll
    for (int j = 0; j < Items; ++j)
        if (((std::uint64_t(scores[j]) << 32) | ids.word(j)) >= threshold)
            store(rank++, (std::uint64_t(scores[j]) << 32) | ids.word(j));
}
} // namespace ninfer::ops::detail
