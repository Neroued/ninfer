#pragma once

#include "ops/softmax_attention/kpool_mla/prepare/arguments.h"
#include "ops/common/numeric/math.cuh"

namespace ninfer::ops::detail::kpool {

inline void set_window(PrepareArguments& a, std::int64_t begin, int columns) {
    a.first_row         = begin / a.tokens;
    a.first_column      = begin - std::int64_t(a.first_row) * a.tokens;
    a.first_span        = a.tokens - a.first_column;
    a.last_row          = (begin + columns - 1) / a.tokens;
    a.last_stop         = begin + columns - std::int64_t(a.last_row) * a.tokens;
    const auto divisor  = common::FastDivisorU31::make(a.tokens);
    a.token_multiplier  = divisor.multiplier;
    a.token_shift       = divisor.shift;
    a.stage_columns     = columns;
    a.first_pool_jobs   = (std::min(a.first_span, columns) + 6) / 4;
    a.pool_jobs_per_row = (std::int64_t(a.tokens) + 6) / 4;
    a.middle_pool_jobs  = std::max(0, a.last_row - a.first_row - 1) * a.pool_jobs_per_row;
    a.pool_jobs         = a.first_pool_jobs + a.middle_pool_jobs +
                  (a.last_row != a.first_row ? (a.last_stop + 6LL) / 4 : 0);
    const auto pools  = common::FastDivisorU31::make(a.pool_jobs_per_row);
    a.pool_multiplier = pools.multiplier;
    a.pool_shift      = pools.shift;
}

// Only the bounded window remainder needs division. The full flat position can exceed I32;
// it is never narrowed, and no Device 64-bit division is needed for batch addressing.
template <class Coordinates>
__device__ __forceinline__ void coordinate(const Coordinates& a, int column, int& b, int& t) {
    if (column < a.first_span) {
        b = a.first_row;
        t = a.first_column + column;
    } else {
        const auto delta = std::uint32_t(column - a.first_span);
        const auto row   = (__umulhi(delta, a.token_multiplier) + delta) >> a.token_shift;
        b                = a.first_row + 1 + row;
        t                = delta - row * a.tokens;
    }
}

__device__ __forceinline__ void pool_coordinate(const PrepareArguments& a, int& job, int& b,
                                                int& start, int& stop) {
    if (job < a.first_pool_jobs) {
        b     = a.first_row;
        start = a.first_column;
        stop  = b == a.last_row ? a.last_stop : a.tokens;
        return;
    }
    job -= a.first_pool_jobs;
    if (job < a.middle_pool_jobs) {
        const auto local =
            (__umulhi(unsigned(job), a.pool_multiplier) + unsigned(job)) >> a.pool_shift;
        b = a.first_row + 1 + local;
        job -= local * a.pool_jobs_per_row;
        start = 0;
        stop  = a.tokens;
        return;
    }
    job -= a.middle_pool_jobs;
    b     = a.last_row;
    start = 0;
    stop  = a.last_stop;
}

// A window intersecting at most two segments encodes them directly. Larger
// windows use packed anchors so each CTA still owns exactly one segment's pool.
__device__ __forceinline__ int packed_segment(const PackedPrepareArguments& a, int column) {
    if (a.first_row >= 0) return column < a.first_span ? a.first_row : a.last_row;
    return a.token_to_seq[a.window_begin + column];
}

__device__ __forceinline__ int packed_segment_begin(const PackedPrepareArguments& a, int b) {
    if (a.first_row >= 0)
        return b == a.first_row ? a.window_begin - a.first_column : a.window_begin + a.first_span;
    return a.cu_seqlens[b];
}

__device__ __forceinline__ void pool_coordinate(const PackedPrepareArguments& a, int& job, int& b,
                                                int& start, int& stop) {
    if (a.first_row >= 0) {
        if (job < a.first_pool_jobs) {
            b = a.first_row;
            start = a.first_column;
            stop = start + a.first_span;
        } else {
            b = a.last_row;
            start = 0;
            stop = a.last_stop;
            job -= a.first_pool_jobs;
        }
        return;
    }
    const int local = a.pool_columns[job];
    const int m     = a.window_begin + local;
    b               = a.token_to_seq[m];
    const int base  = a.cu_seqlens[b];
    start           = max(0, a.window_begin - base);
    stop            = min(a.window_begin + a.stage_columns, a.cu_seqlens[b + 1]) - base;
    job             = (m - base - start) / 4;
}
} // namespace ninfer::ops::detail::kpool
