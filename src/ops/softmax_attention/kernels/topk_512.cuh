#pragma once

#include "ops/common/cuda/reduce.cuh"
#include <cub/block/block_scan.cuh>

namespace ninfer::ops::detail {

using Top512Scan = cub::BlockScan<int, 256>;

template <int Bits, int Copies>
struct Top512Storage {
    static constexpr int bins = 1 << Bits;

    union {
        int histogram[Copies][bins];
        unsigned differences[16];
        Top512Scan::TempStorage scan;
    } temp;

    std::uint64_t prefix, anchor, mask;
    int shift, remaining, selected_count;
};

struct Top512StoreKeys {
    std::uint64_t* out;

    __device__ void operator()(int rank, std::uint64_t key) const { out[rank] = key; }
};

struct Top512StoreIds {
    std::int32_t* out;

    __device__ void operator()(int rank, std::uint64_t key) const {
        out[rank] = common::id_from_order_key(key);
    }
};

// Full radix selection over a unique score/id total order. Prefix refinement stops only
// when exactly 512 keys are >= the prefix. No approximate local candidate truncation.
// The first count keys are valid; remaining entries are zero padding.
template <int Items, int Bits, int Copies, class Store>
__device__ void select_top512(std::uint64_t (&keys)[Items], int count,
                              Top512Storage<Bits, Copies>& s, Store store) {
    constexpr int Bins = 1 << Bits;
    const int tid      = threadIdx.x;
    if (count <= 512) {
#pragma unroll
        for (int j = 0; j < Items; ++j)
            if (j * 256 + tid < count) store(j * 256 + tid, keys[j]);
        return;
    }
    if (tid == 0) s.anchor = keys[0];
    __syncthreads();
    const auto anchor        = s.anchor;
    std::uint64_t difference = 0;
#pragma unroll
    for (int j = 0; j < Items; ++j)
        if (keys[j]) difference |= keys[j] ^ anchor;
    // OR of differences from any valid anchor gives the exact common prefix.
    const unsigned low  = __reduce_or_sync(0xffffffffu, unsigned(difference));
    const unsigned high = __reduce_or_sync(0xffffffffu, unsigned(difference >> 32));
    if (tid % 32 == 0) {
        s.temp.differences[tid / 32]     = low;
        s.temp.differences[8 + tid / 32] = high;
    }
    __syncthreads();
    if (tid < 32) {
        const unsigned lo = __reduce_or_sync(0xffffffffu, tid < 8 ? s.temp.differences[tid] : 0u);
        const unsigned hi =
            __reduce_or_sync(0xffffffffu, tid < 8 ? s.temp.differences[8 + tid] : 0u);
        if (tid == 0) {
            s.shift     = ((hi ? 63 - __clz(hi) : 31 - __clz(lo)) / Bits) * Bits;
            s.mask      = s.shift >= 64 - Bits ? 0 : (~std::uint64_t(0) << (s.shift + Bits));
            s.prefix    = anchor & s.mask;
            s.remaining = 512;
        }
    }
    __syncthreads();
    for (int shift = s.shift; shift >= 0; shift -= Bits) {
        for (int i = tid; i < Copies * Bins; i += 256) (&s.temp.histogram[0][0])[i] = 0;
        __syncthreads();
#pragma unroll
        for (int j = 0; j < Items; ++j) {
            const auto key = keys[j];
            const int bin =
                key && (key & s.mask) == s.prefix ? int((key >> shift) & (Bins - 1)) : Bins;
            // Unit increments compile to the hardware's combined shared-memory count.
            if (bin < Bins) atomicAdd(&s.temp.histogram[tid / 32 % Copies][bin], 1);
        }
        __syncthreads();
        int histogram = 0;
#pragma unroll
        for (int copy = 0; copy < Copies; ++copy) {
            if constexpr (Bins == 256)
                histogram += s.temp.histogram[copy][tid];
            else if (tid < Bins)
                histogram += s.temp.histogram[copy][tid];
        }
        __syncthreads();
        int below, total;
        Top512Scan(s.temp.scan).ExclusiveSum(histogram, below, total);
        const int above     = total - below - histogram;
        const int remaining = s.remaining;
        __syncthreads();
        if (above < remaining && above + histogram >= remaining) {
            s.prefix |= std::uint64_t(tid) << shift;
            s.mask |= std::uint64_t(Bins - 1) << shift;
            s.remaining      = remaining - above;
            s.selected_count = histogram;
        }
        __syncthreads();
        if (s.selected_count == s.remaining) break;
    }
    const auto threshold = s.prefix;
    int kept             = 0;
#pragma unroll
    for (int j = 0; j < Items; ++j) kept += keys[j] >= threshold;
    int rank;
    Top512Scan(s.temp.scan).ExclusiveSum(kept, rank);
#pragma unroll
    for (int j = 0; j < Items; ++j)
        if (keys[j] >= threshold) store(rank++, keys[j]);
}

} // namespace ninfer::ops::detail
