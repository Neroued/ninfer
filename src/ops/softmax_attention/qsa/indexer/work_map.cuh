#pragma once

#include "ops/softmax_attention/qsa/indexer/arguments.h"

namespace ninfer::ops::detail::qsa {

// Large batches keep one kernel node per stage. Split the CTA-local work maps while
// retaining window-wide score and candidate storage.
template <int Capacity, bool BatchedLoop, class Arguments, class Run>
__device__ __forceinline__ void for_each_indexer_job_group(const Arguments& a, Run&& run) {
    if constexpr (!BatchedLoop) {
        run(a, int(blockIdx.x), int(gridDim.x));
    } else {
        const int group_rows = Capacity / a.width;
        const int groups     = 1 + (a.batch - 1) / group_rows;
        const int lanes      = min(int(gridDim.x), groups);
        const int lane       = int(blockIdx.x) % lanes;
        const int worker     = int(blockIdx.x) / lanes;
        const int stride     = 1 + (int(gridDim.x) - 1 - lane) / lanes;
        for (std::int64_t group = lane; group < groups; group += lanes) {
            const int first = int(group * group_rows);
            run(a.rows(first, min(group_rows, a.batch - first)), worker, stride);
            if (group + lanes < groups) __syncthreads();
        }
    }
}

// Per-query selection and SIMT scoring use slot indices directly.
template <int Capacity>
struct RaggedIndexerSlots {
    int visible[Capacity], prefix[Capacity + 1], groups;

    template <int QueryTile, int KeyTile>
    __device__ bool build(const RaggedIndexerArguments& a) {
        static_assert(QueryTile == 1);
        const int tid   = threadIdx.x;
        const int count = tid < a.width ? a.visible(tid) : 0;
        if (tid < a.width) visible[tid] = count;
        if (!__syncthreads_or(count > kIndexerSelected)) return false;
        if (tid < 32) {
            int carry = 0;
#pragma unroll
            for (int part = 0; part < Capacity / 32; ++part) {
                const int slot = part * 32 + tid;
                const int n    = part == 0 ? count : (slot < a.width ? visible[slot] : 0);
                int sum        = n > kIndexerSelected ? (n + KeyTile - 1) / KeyTile : 0;
#pragma unroll
                for (int shift = 1; shift < 32; shift *= 2) {
                    const int previous = __shfl_up_sync(0xffffffffu, sum, shift);
                    if (tid >= shift) sum += previous;
                }
                if (slot < a.width) prefix[slot + 1] = carry + sum;
                carry += __shfl_sync(0xffffffffu, sum, 31);
            }
            if (tid == 0) {
                groups    = a.width;
                prefix[0] = 0;
            }
        }
        __syncthreads();
        return true;
    }

    __device__ int group_for(int job) const {
        int lo = 0, hi = groups;
        while (lo + 1 < hi) {
            const int mid = (lo + hi) / 2;
            if (prefix[mid] <= job)
                lo = mid;
            else
                hi = mid;
        }
        return lo;
    }
};

struct RaggedScoreTile {
    unsigned bits;

    // Seven bits each for the start slot and length minus one. The remaining bits
    // hold the last query's frontier beyond the 2048-token all-select threshold.
    __device__ int start() const { return bits & 127; }

    __device__ int end() const { return start() + ((bits >> 7) & 127) + 1; }

    __device__ int visible(int slot) const {
        return (int(bits >> 14) + 4 * kIndexerSelected - (end() - 1 - slot)) / 4;
    }

    __device__ int visible() const { return visible(end() - 1); }
};

template <int Capacity>
struct RaggedScoreJobs {
    static_assert(Capacity <= kIndexerMaxWindowQueries);
    RaggedScoreTile tiles[Capacity];
    // K256 permits up to 128 * 1024 jobs. With separate high bytes the complete
    // map fits below the 1024-byte dynamic-shared alignment on Spark.
    std::uint16_t end_low[Capacity];
    std::uint8_t end_high[Capacity];
    int groups;

    template <int QueryTile, int KeyTile = kIndexerKeyTile>
    __device__ bool build(const RaggedIndexerArguments& a) {
        const int tid       = threadIdx.x;
        unsigned descriptor = 0;
        if (tid < a.width) {
            const int col = a.begin + tid, seg = a.token_to_seq[col];
            const int start = max(a.begin, a.cu_seqlens[seg]);
            if ((col - start) % QueryTile == 0) {
                const int len = min(QueryTile, min(a.width - tid, a.cu_seqlens[seg + 1] - col));
                const int frontier = a.positions[col + len - 1] + 1;
                if (frontier / 4 > kIndexerSelected)
                    descriptor = unsigned(tid) | (unsigned(len - 1) << 7) |
                                 (unsigned(frontier - 4 * kIndexerSelected) << 14);
            }
        }
        if (tid < Capacity) tiles[tid] = {descriptor};
        if (!__syncthreads_or(descriptor != 0)) return false;

        if (tid < 32) {
            int groups = 0, jobs = 0;
#pragma unroll
            for (int part = 0; part < Capacity / 32; ++part) {
                const auto tile = part == 0 ? RaggedScoreTile{descriptor} : tiles[part * 32 + tid];
                const unsigned active = __ballot_sync(0xffffffffu, tile.bits != 0);
                if (!active) continue;
                int sum = tile.bits ? (tile.visible() + KeyTile - 1) / KeyTile : 0;
#pragma unroll
                for (int shift = 1; shift < 32; shift *= 2) {
                    const int previous = __shfl_up_sync(0xffffffffu, sum, shift);
                    if (tid >= shift) sum += previous;
                }
                if (tile.bits) {
                    const int g   = groups + __popc(active & ((1u << tid) - 1));
                    tiles[g]      = tile;
                    const int end = jobs + sum;
                    end_low[g]    = end;
                    end_high[g]   = end >> 16;
                }
                groups += __popc(active);
                jobs += __shfl_sync(0xffffffffu, sum, 31);
            }
            if (tid == 0) this->groups = groups;
        }
        __syncthreads();
        return true;
    }

    __device__ int job_end(int group) const {
        return int(end_low[group]) | (int(end_high[group]) << 16);
    }

    __device__ int job_count() const { return job_end(groups - 1); }

    __device__ int job_begin(int group) const { return group ? job_end(group - 1) : 0; }

    __device__ int group_for(int job) const {
        int lo = 0, hi = groups - 1;
        while (lo < hi) {
            const int mid = (lo + hi) / 2;
            if (job_end(mid) <= job)
                lo = mid + 1;
            else
                hi = mid;
        }
        return lo;
    }
};

template <class Jobs>
__device__ __forceinline__ int indexer_score_jobs(const Jobs& jobs) {
    if constexpr (requires { jobs.job_count(); })
        return jobs.job_count();
    else
        return jobs.prefix[jobs.groups];
}

template <class Jobs>
__device__ __forceinline__ int indexer_score_job_begin(const Jobs& jobs, int group) {
    if constexpr (requires { jobs.job_begin(group); })
        return jobs.job_begin(group);
    else
        return jobs.prefix[group];
}

template <class Jobs>
__device__ __forceinline__ int indexer_score_start(const Jobs& jobs, int group, int rectangular) {
    if constexpr (requires { jobs.tiles; })
        return jobs.tiles[group].start();
    else
        return rectangular;
}

template <class Jobs>
__device__ __forceinline__ int indexer_score_end(const Jobs& jobs, int group, int rectangular) {
    if constexpr (requires { jobs.tiles; })
        return jobs.tiles[group].end();
    else
        return rectangular;
}

template <class Jobs>
__device__ __forceinline__ int indexer_score_length(const Jobs& jobs, int group) {
    if constexpr (requires { jobs.tiles; })
        return jobs.tiles[group].visible();
    else
        return jobs.length[group];
}

template <int Capacity = 64>
struct IndexerJobs {
    int visible[Capacity];
    int prefix[Capacity + 1];
    int groups;

    __device__ int tile_visible(const IndexerArguments& a, int b, int first, int tile) const {
        // Live queries are a prefix with consecutive positions, so its last query has the
        // largest causal history. No position from an inert suffix is inspected.
        const int end = min(min(a.width, a.valid[b] - a.begin), first + tile);
        return end > first ? visible[b * a.width + end - 1] : 0;
    }

    // A query tile never crosses batch rows. Every CTA rebuilds this small immutable map,
    // so actual context lengths distribute work without a Host read or a global queue.
    template <int QueryTile, int KeyTile>
    __device__ bool build(const IndexerArguments& a) {
        const int tid    = threadIdx.x;
        const int length = tid < a.width * a.batch ? a.visible(tid) : 0;
        if (tid < a.width * a.batch) visible[tid] = length;
        if (!__syncthreads_or(length > kIndexerSelected)) return false;
        if (tid < 32) {
            const int per_row = (a.width + QueryTile - 1) / QueryTile;
            const int count   = per_row * a.batch;
            int carry         = 0;
#pragma unroll
            for (int half = 0; half < Capacity / 32; ++half) {
                const int g = tid + half * 32;
                const int n =
                    g < count ? tile_visible(a, g / per_row, (g % per_row) * QueryTile, QueryTile)
                              : 0;
                int sum = n > kIndexerSelected ? (n + KeyTile - 1) / KeyTile : 0;
#pragma unroll
                for (int offset = 1; offset < 32; offset *= 2) {
                    const int previous = __shfl_up_sync(0xffffffffu, sum, offset);
                    if (tid >= offset) sum += previous;
                }
                prefix[g + 1] = carry + sum;
                carry += __shfl_sync(0xffffffffu, sum, 31);
            }
            if (tid == 0) {
                groups    = count;
                prefix[0] = 0;
            }
        }
        __syncthreads();
        return true;
    }

    __device__ int group_for(int job) const {
        int lo = 0, hi = groups;
        while (lo + 1 < hi) {
            const int mid = (lo + hi) / 2;
            if (prefix[mid] <= job)
                lo = mid;
            else
                hi = mid;
        }
        return lo;
    }
};

} // namespace ninfer::ops::detail::qsa
