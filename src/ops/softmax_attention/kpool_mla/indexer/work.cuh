#pragma once

#include "ops/softmax_attention/kpool_mla/indexer/arguments.h"

namespace ninfer::ops::detail::kpool {

struct QuerySpan {
    int first, live, position, visible;

    template <int Queries>
    __device__ int groups() const {
        return visible > kIndexSelected ? (live - first + Queries - 1) / Queries : 0;
    }
};

__device__ __forceinline__ QuerySpan query_span(const IndexerArguments& a, int b) {
    const int live = min(a.width, max(0, a.valid[b] - a.begin));
    if (!live) return {};
    // Live positions are consecutive. Derive the span from its first ordinal;
    // an inert request must not read positions or its page-table selector.
    const int position = a.positions[std::int64_t(b) * a.tokens + a.begin];
    return {max(0, 4 * (kIndexSelected + 1) - 1 - position), live, position, (position + live) / 4};
}

struct ScoreWork {
    int batch = -1, first = 0, start = 0, step = 0, live = 0, position = 0;
};

template <int Queries, int Keys>
__device__ __forceinline__ ScoreWork score_work(const IndexerArguments& a) {
    if (a.batch == 1) {
        const auto span  = query_span(a, 0);
        const int groups = span.groups<Queries>();
        if (!groups) return {};
        const int group = int(blockIdx.x) % groups;
        int workers     = gridDim.x / groups;
        if constexpr (Queries == 8) {
            // Unequal strides use every resident CTA when K fits L2. Long
            // streams retain equal strides so query groups reuse each key tile.
            if (a.columns < 32768)
                workers += group < int(gridDim.x) % groups;
            else if (int(blockIdx.x) >= workers * groups)
                return {};
        } else if (int(blockIdx.x) >= workers * groups)
            return {};
        return {0,
                span.first + group * Queries,
                int(blockIdx.x) / groups * Keys,
                workers * Keys,
                span.live,
                span.position};
    }
    int groups        = 0;
    std::int64_t work = 0;
    for (int b = 0; b < a.batch; ++b) {
        const auto span = query_span(a, b);
        const int n     = span.groups<Queries>();
        groups += n;
        work += std::int64_t(n) * span.visible;
    }
    if (!groups) return {};
    const int available = max(0, int(gridDim.x) - groups);
    int remaining       = blockIdx.x;
    for (int b = 0; b < a.batch; ++b) {
        const auto span = query_span(a, b);
        const int n     = span.groups<Queries>();
        if (!n) continue;
        // Reserve one worker per scored group, then distribute the remainder by
        // history length. Rounding down keeps the total within the captured grid.
        const int workers =
            1 + (n == groups ? available / n : int(available * std::int64_t(span.visible) / work));
        const int jobs = workers * n;
        if (remaining < jobs)
            return {b,
                    span.first + remaining % n * Queries,
                    remaining / n * Keys,
                    workers * Keys,
                    span.live,
                    span.position};
        remaining -= jobs;
    }
    return {};
}

// A packed window has a launch-time tile map, but eligibility remains a
// device decision. Count scored tiles before distributing the resident workers
// so a deep segment still gets the full wave alongside all-select segments.
template <int Queries, int Keys>
__device__ __forceinline__ ScoreWork score_work(const PackedIndexerArguments& a) {
    const int lane    = threadIdx.x % 32;
    const int chunks  = (a.tile_count + 31) / 32;
    const auto scored = [&](int chunk) {
        const int i = chunk * 32 + lane;
        bool active = false;
        if (i < a.tile_count) {
            const unsigned tile = a.tiles[i];
            const int last      = (tile & 255) + (tile >> 8) - 1;
            active              = a.end(last) / 4 > kIndexSelected;
        }
        return __ballot_sync(0xffffffffu, active);
    };
    int groups = 0;
    for (int chunk = 0; chunk < chunks; ++chunk) groups += __popc(scored(chunk));
    if (!groups) return {};
    int rank          = int(blockIdx.x) % groups;
    const int worker  = int(blockIdx.x) / groups;
    const int workers = (int(gridDim.x) - 1 - rank) / groups + 1;
    for (int chunk = 0; chunk < chunks; ++chunk) {
        const unsigned mask = scored(chunk);
        const int count     = __popc(mask);
        if (rank >= count) {
            rank -= count;
            continue;
        }
        const int i         = __fns(mask, 0, rank + 1);
        const unsigned tile = a.tiles[chunk * 32 + i];
        const int first     = tile & 255;
        const int column    = a.begin + first;
        return {a.token_to_seq[column], first,
                worker * Keys,          workers * Keys,
                first + int(tile >> 8), a.positions[column] - first};
    }
    return {};
}

struct PartitionSpan {
    int first, live, parts, low_rows, jobs;
};

__device__ __forceinline__ PartitionSpan partition_span(const IndexerArguments& a, int b) {
    const auto span = query_span(a, b);
    if (!span.groups<1>()) return {};
    // An FP8 query window is shorter than one partition's original-token span,
    // so its scored rows have at most two adjacent partition counts.
    constexpr int tokens_per_partition = 4 * kIndexPartition;
    const int parts =
        (span.position + span.first + tokens_per_partition - 3) / tokens_per_partition;
    const int boundary =
        (parts + 1) * tokens_per_partition - span.position - (tokens_per_partition - 3);
    const int low = min(span.live - span.first, max(0, boundary - span.first));
    return {span.first, span.live, parts, low,
            low * parts + (span.live - span.first - low) * (parts + 1)};
}

struct PartitionJob {
    int slot, part;
};

__device__ __forceinline__ PartitionJob partition_job(const IndexerArguments& a, int job) {
    for (int b = 0; b < a.batch; ++b) {
        const auto span = partition_span(a, b);
        if (job >= span.jobs) {
            job -= span.jobs;
            continue;
        }
        const int low_jobs = span.low_rows * span.parts;
        if (job < low_jobs) return {b * a.width + span.first + job / span.parts, job % span.parts};
        job -= low_jobs;
        return {b * a.width + span.first + span.low_rows + job / (span.parts + 1),
                job % (span.parts + 1)};
    }
    return {-1, 0};
}
} // namespace ninfer::ops::detail::kpool
