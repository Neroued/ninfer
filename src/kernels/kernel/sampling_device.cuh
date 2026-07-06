#pragma once

// qus::kernels - shared device-side sampling primitives (no __global__ symbols,
// so this header can be included by multiple translation units). Holds the
// counter-based RNG, the candidate ordering, and the block-collaborative
// truncated-distribution builder used by both sample_column and the MTP
// rejection-sampling accept kernel.

#include "qus/kernels/sampling.h"

#include <cub/block/block_merge_sort.cuh>
#include <cub/block/block_radix_sort.cuh>

#include <cuda_bf16.h>
#include <climits>
#include <cstdint>
#include <math_constants.h>

namespace qus::kernels {

inline constexpr int kSamplerBlock         = 256;
inline constexpr int kSamplerMaxCandidates = 256;
inline constexpr int kSamplerTileItems     = 256;
inline constexpr int kSamplerItemsPerThread = 3;
inline constexpr int kSamplerPartialTileItems = kSamplerBlock * kSamplerItemsPerThread;
inline constexpr int kSamplerFusedItemsPerThread = 2;
inline constexpr int kSamplerFusedPartialTileItems =
    kSamplerBlock * kSamplerFusedItemsPerThread;
inline constexpr int kSamplerFinalizeItemsPerThread = 10;
inline constexpr int kSamplerFinalizeTileItems = kSamplerBlock * kSamplerFinalizeItemsPerThread;
inline constexpr int kSamplerGroupItemsPerThread = 2;
inline constexpr int kSamplerGroupTileItems = kSamplerBlock * kSamplerGroupItemsPerThread;
inline constexpr int kSamplerPartialsPerGroup = 21;
inline constexpr int kSamplerFastCandidates = 20;
inline constexpr int kSamplerScratchColumns = 8;
inline constexpr int kSamplerScratchPartialBlocks = 1024;

static __device__ float sampling_partial_val[kSamplerScratchColumns *
                                             kSamplerScratchPartialBlocks *
                                             kSamplerMaxCandidates];
static __device__ int sampling_partial_idx[kSamplerScratchColumns *
                                           kSamplerScratchPartialBlocks *
                                           kSamplerMaxCandidates];
static __device__ int sampling_dist_idx[kSamplerScratchColumns * kSamplerMaxCandidates];
static __device__ float sampling_dist_prob[kSamplerScratchColumns * kSamplerMaxCandidates];
static __device__ int sampling_dist_support[kSamplerScratchColumns];
static __device__ int sampling_mtp_finalize_init;
static __device__ int sampling_mtp_finalize_count;
static __device__ int sampling_partial_done[kSamplerScratchColumns];
static __device__ int sampling_group_init[kSamplerScratchColumns];
static __device__ int sampling_group_done[kSamplerScratchColumns];

using SamplingPartialBlockSort =
    cub::BlockRadixSort<unsigned long long, kSamplerBlock, kSamplerItemsPerThread, int>;
using SamplingFinalizeBlockSort =
    cub::BlockRadixSort<unsigned long long, kSamplerBlock, kSamplerFinalizeItemsPerThread, int>;
using SamplingGroupBlockSort =
    cub::BlockRadixSort<unsigned long long, kSamplerBlock, kSamplerGroupItemsPerThread, int>;
using SamplingPartialMergeSort =
    cub::BlockMergeSort<unsigned long long, kSamplerBlock, kSamplerItemsPerThread, int>;
using SamplingFusedPartialMergeSort =
    cub::BlockMergeSort<unsigned long long, kSamplerBlock, kSamplerFusedItemsPerThread, int>;
using SamplingGroupMergeSort =
    cub::BlockMergeSort<unsigned long long, kSamplerBlock, kSamplerGroupItemsPerThread, int>;

struct SamplingFusedGroupShared {
    typename SamplingGroupMergeSort::TempStorage sort_storage;
    float cand_val[kSamplerMaxCandidates];
    int cand_idx[kSamplerMaxCandidates];
    float prob[kSamplerMaxCandidates];
    int n_support;
    int is_last;
};

union SamplingFusedShared {
    typename SamplingFusedPartialMergeSort::TempStorage partial_sort_storage;
    SamplingFusedGroupShared group;
};

struct SamplingKeyGreater {
    __device__ __forceinline__ bool operator()(unsigned long long a, unsigned long long b) const {
        return a > b;
    }
};

__device__ __forceinline__ unsigned long long sampling_splitmix64(unsigned long long x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// Uniform float in [0,1) from (seed, position, purpose, sub). Pure function of
// its inputs so it is safe under CUDA-graph replay (no mutable RNG state).
__device__ __forceinline__ float sampling_uniform(unsigned long long seed, int position,
                                                   int purpose, unsigned int sub) {
    unsigned long long key = seed;
    key = sampling_splitmix64(key ^ (static_cast<unsigned long long>(static_cast<unsigned int>(position)) *
                                     0xD1B54A32D192ED03ull));
    key = sampling_splitmix64(key ^ (static_cast<unsigned long long>(static_cast<unsigned int>(purpose)) << 21) ^
                              (static_cast<unsigned long long>(sub) * 0x2545F4914F6CDD1Dull));
    const unsigned int bits = static_cast<unsigned int>(key >> 40);  // 24 bits
    return static_cast<float>(bits) * (1.0f / 16777216.0f);
}

// Candidate ordering: higher value wins, ties broken by lower vocab index.
__device__ __forceinline__ bool sampling_better(float v, int i, float bv, int bi) {
    return v > bv || (v == bv && i < bi);
}

// True when (v,i) ranks strictly below pivot (pv,pi) in the ordering above.
__device__ __forceinline__ bool sampling_worse_than(float v, int i, float pv, int pi) {
    return pv > v || (pv == v && pi < i);
}

__device__ __forceinline__ unsigned int sampling_ordered_float(float v) {
    const unsigned int bits = __float_as_uint(v);
    return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

__device__ __forceinline__ unsigned long long sampling_sort_key(float v, int idx) {
    if (idx == INT_MAX) { return 0ull; }
    return (static_cast<unsigned long long>(sampling_ordered_float(v)) << 32) |
           static_cast<unsigned int>(0xffffffffu - static_cast<unsigned int>(idx));
}

__device__ __forceinline__ float sampling_key_float(unsigned long long key) {
    const unsigned int ordered = static_cast<unsigned int>(key >> 32);
    const unsigned int bits = (ordered & 0x80000000u) ? (ordered ^ 0x80000000u) : ~ordered;
    return __uint_as_float(bits);
}

__device__ __forceinline__ int sampling_candidate_cap(const SamplingConfig& cfg,
                                                      std::int32_t vocab) {
    int cap = kSamplerMaxCandidates;
    if (cfg.top_k > 0 && cfg.top_k < cap) { cap = cfg.top_k; }
    if (vocab < cap) { cap = vocab; }
    return cap;
}

__device__ __forceinline__ int sampling_partial_offset(int col, int partial, int j) {
    return ((col * kSamplerScratchPartialBlocks + partial) * kSamplerMaxCandidates) + j;
}

__device__ __forceinline__ int sampling_dist_offset(int col, int j) {
    return col * kSamplerMaxCandidates + j;
}

__device__ __forceinline__ float sampling_adjusted_logit(float raw, int v,
                                                         const SamplingConfig& c) {
    float x = raw;
    if (c.token_counts != nullptr) {
        const int cnt = c.token_counts[v];
        if (cnt > 0) { x -= c.presence_penalty; }
        x -= c.frequency_penalty * static_cast<float>(cnt);
    }
    return x / c.temperature;
}

__device__ __forceinline__ void sampling_insert_candidate(float* vals, int* idxs, int cap,
                                                          float v, int idx) {
    if (cap <= 0 || !sampling_better(v, idx, vals[cap - 1], idxs[cap - 1])) { return; }
    int pos = cap - 1;
    while (pos > 0 && sampling_better(v, idx, vals[pos - 1], idxs[pos - 1])) {
        vals[pos] = vals[pos - 1];
        idxs[pos] = idxs[pos - 1];
        --pos;
    }
    vals[pos] = v;
    idxs[pos] = idx;
}

__device__ inline void sampling_sort_tile_desc(float* vals, int* idxs) {
    const int tid = threadIdx.x;
    for (int size = 2; size <= kSamplerTileItems; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            const int other = tid ^ stride;
            __syncthreads();
            if (other > tid) {
                const float ov = vals[other];
                const int oi = idxs[other];
                const bool descending = ((tid & size) == 0);
                const bool swap =
                    descending ? sampling_better(ov, oi, vals[tid], idxs[tid])
                               : sampling_better(vals[tid], idxs[tid], ov, oi);
                if (swap) {
                    const float tv = vals[tid];
                    const int ti = idxs[tid];
                    vals[tid] = ov;
                    idxs[tid] = oi;
                    vals[other] = tv;
                    idxs[other] = ti;
                }
            }
            __syncthreads();
        }
    }
}

__device__ inline void sampling_normalize_support(const SamplingConfig& cfg, float* cand_val,
                                                  int* cand_idx, float* prob, int* n_support,
                                                  int n) {
    const int tid = threadIdx.x;
    if (tid == 0) {
        const float m = cand_val[0];
        float sum     = 0.0f;
        for (int j = 0; j < n; ++j) {
            const float e = __expf(cand_val[j] - m);
            prob[j]       = e;
            sum += e;
        }
        const float e0           = prob[0];
        const float min_p_thresh = (cfg.min_p > 0.0f) ? cfg.min_p * e0 : -1.0f;
        const bool top_p_active  = (cfg.top_p < 1.0f);
        const float top_p_target = cfg.top_p * sum;
        float cum                = 0.0f;
        int support              = 0;
        for (int j = 0; j < n; ++j) {
            if (min_p_thresh >= 0.0f && prob[j] < min_p_thresh) { break; }
            cum += prob[j];
            support = j + 1;
            if (top_p_active && cum >= top_p_target) { break; }
        }
        if (support < 1) { support = 1; }
        float ssum = 0.0f;
        for (int j = 0; j < support; ++j) { ssum += prob[j]; }
        const float inv = 1.0f / ssum;
        for (int j = 0; j < support; ++j) { prob[j] *= inv; }
        *n_support = support;
    }
    __syncthreads();
}


// All threads of the block must call. Small-column exact truncated support
// builder used by unit tests and any <=256-token column. It sorts one shared
// tile and then applies the same top-p/min-p renormalization as the large path.
__device__ inline void sampling_build_truncated_small(const __nv_bfloat16* logits,
                                                      std::int64_t base, std::int32_t vocab,
                                                      const SamplingConfig& cfg, float* tile_val,
                                                      int* tile_idx, float* cand_val,
                                                      int* cand_idx, float* prob,
                                                      int* n_support) {
    const int tid = threadIdx.x;
    const int cap = sampling_candidate_cap(cfg, vocab);
    if (tid < kSamplerTileItems) {
        if (tid < vocab) {
            const float x = sampling_adjusted_logit(__bfloat162float(logits[base + tid]), tid, cfg);
            tile_val[tid] = x;
            tile_idx[tid] = tid;
        } else {
            tile_val[tid] = -CUDART_INF_F;
            tile_idx[tid] = INT_MAX;
        }
    }
    __syncthreads();
    sampling_sort_tile_desc(tile_val, tile_idx);
    if (tid < cap) {
        cand_val[tid] = tile_val[tid];
        cand_idx[tid] = tile_idx[tid];
    }
    __syncthreads();
    sampling_normalize_support(cfg, cand_val, cand_idx, prob, n_support, cap);
}

// Single-block fallback for large columns when the fixed sampler scratch cannot
// represent the launch. It still reads each vocab entry once and keeps a bounded
// per-thread top-20, so the old top_k*vocab global reread pattern is gone.
__device__ inline void sampling_build_truncated_block_fast(const __nv_bfloat16* logits,
                                                           std::int64_t base,
                                                           std::int32_t vocab,
                                                           const SamplingConfig& cfg,
                                                           float* merge_val, int* merge_idx,
                                                           float* cand_val, int* cand_idx,
                                                           float* prob, int* n_support) {
    const int tid = threadIdx.x;
    const int cap = sampling_candidate_cap(cfg, vocab);
    if (cap > kSamplerFastCandidates) {
        if (tid == 0) {
            for (int j = 0; j < cap; ++j) {
                cand_val[j] = -CUDART_INF_F;
                cand_idx[j] = INT_MAX;
            }
            for (int v = 0; v < vocab; ++v) {
                const float x =
                    sampling_adjusted_logit(__bfloat162float(logits[base + v]), v, cfg);
                sampling_insert_candidate(cand_val, cand_idx, cap, x, v);
            }
        }
        __syncthreads();
        sampling_normalize_support(cfg, cand_val, cand_idx, prob, n_support, cap);
        return;
    }

    float local_val[kSamplerFastCandidates];
    int local_idx[kSamplerFastCandidates];
#pragma unroll
    for (int j = 0; j < kSamplerFastCandidates; ++j) {
        local_val[j] = -CUDART_INF_F;
        local_idx[j] = INT_MAX;
    }

    const int fast_cap = cap <= kSamplerFastCandidates ? cap : kSamplerFastCandidates;
    for (int v = tid; v < vocab; v += blockDim.x) {
        const float x = sampling_adjusted_logit(__bfloat162float(logits[base + v]), v, cfg);
        sampling_insert_candidate(local_val, local_idx, fast_cap, x, v);
    }

    for (int j = 0; j < kSamplerFastCandidates; ++j) {
        const int off = tid * kSamplerFastCandidates + j;
        merge_val[off] = local_val[j];
        merge_idx[off] = local_idx[j];
    }
    __syncthreads();

    if (tid == 0) {
        for (int j = 0; j < cap; ++j) {
            cand_val[j] = -CUDART_INF_F;
            cand_idx[j] = INT_MAX;
        }
        const int merge_n = blockDim.x * kSamplerFastCandidates;
        for (int p = 0; p < merge_n; ++p) {
            const int idx = merge_idx[p];
            if (idx == INT_MAX) { continue; }
            sampling_insert_candidate(cand_val, cand_idx, fast_cap, merge_val[p], idx);
        }
    }
    __syncthreads();
    sampling_normalize_support(cfg, cand_val, cand_idx, prob, n_support, fast_cap);
}

__device__ inline void sampling_merge_partials_to_support(
    int col, int partial_blocks, const SamplingConfig& cfg,
    typename SamplingFinalizeBlockSort::TempStorage& sort_storage, float* merge_val,
    int* merge_idx, float* cand_val, int* cand_idx, float* prob, int* n_support,
    std::int32_t vocab, int cap, bool normalize) {
    const int tid = threadIdx.x;
    const int n = partial_blocks * cap;
    if (n <= kSamplerFinalizeTileItems) {
        unsigned long long keys[kSamplerFinalizeItemsPerThread];
        int items[kSamplerFinalizeItemsPerThread];
#pragma unroll
        for (int item = 0; item < kSamplerFinalizeItemsPerThread; ++item) {
            const int p = item * blockDim.x + tid;
            if (p < n) {
                const int partial = p / cap;
                const int j = p - partial * cap;
                const int off = sampling_partial_offset(col, partial, j);
                const int idx = sampling_partial_idx[off];
                const float v = sampling_partial_val[off];
                keys[item] = sampling_sort_key(v, idx);
                items[item] = idx;
            } else {
                keys[item] = 0ull;
                items[item] = INT_MAX;
            }
        }
        SamplingFinalizeBlockSort(sort_storage).SortDescending(keys, items);
#pragma unroll
        for (int item = 0; item < kSamplerFinalizeItemsPerThread; ++item) {
            const int rank = tid * kSamplerFinalizeItemsPerThread + item;
            if (rank < cap) {
                cand_val[rank] = sampling_key_float(keys[item]);
                cand_idx[rank] = items[item];
            }
        }
        __syncthreads();
        if (normalize) {
            sampling_normalize_support(cfg, cand_val, cand_idx, prob, n_support, cap);
        } else {
            if (tid == 0) { *n_support = cap; }
            __syncthreads();
        }
        return;
    }

    if (cap <= kSamplerFastCandidates) {
        const int lane = tid & 31;
        const int warp = tid >> 5;
        constexpr unsigned int kMask = 0xffffffffu;
        float pivot_val = 0.0f;
        int pivot_idx = -1;
        for (int rank = 0; rank < kSamplerFastCandidates; ++rank) {
            float best_val = -CUDART_INF_F;
            int best_idx = INT_MAX;
            if (rank < cap) {
                for (int p = tid; p < n; p += blockDim.x) {
                    const int partial = p / cap;
                    const int j = p - partial * cap;
                    const int off = sampling_partial_offset(col, partial, j);
                    const int idx = sampling_partial_idx[off];
                    const float v = sampling_partial_val[off];
                    if (idx != INT_MAX &&
                        (rank == 0 || sampling_worse_than(v, idx, pivot_val, pivot_idx)) &&
                        sampling_better(v, idx, best_val, best_idx)) {
                        best_val = v;
                        best_idx = idx;
                    }
                }
            }
            for (int offset = 16; offset > 0; offset >>= 1) {
                const float other_val = __shfl_down_sync(kMask, best_val, offset);
                const int other_idx = __shfl_down_sync(kMask, best_idx, offset);
                if (sampling_better(other_val, other_idx, best_val, best_idx)) {
                    best_val = other_val;
                    best_idx = other_idx;
                }
            }
            best_val = __shfl_sync(kMask, best_val, 0);
            best_idx = __shfl_sync(kMask, best_idx, 0);
            if (lane == 0) {
                const int off = warp * kSamplerFastCandidates + rank;
                merge_val[off] = best_val;
                merge_idx[off] = best_idx;
            }
            pivot_val = best_val;
            pivot_idx = best_idx;
        }
        __syncthreads();
        if (tid == 0) {
            for (int j = 0; j < cap; ++j) {
                cand_val[j] = -CUDART_INF_F;
                cand_idx[j] = INT_MAX;
            }
            const int merge_n = (kSamplerBlock / 32) * kSamplerFastCandidates;
            for (int p = 0; p < merge_n; ++p) {
                const int idx = merge_idx[p];
                if (idx == INT_MAX) { continue; }
                sampling_insert_candidate(cand_val, cand_idx, cap, merge_val[p], idx);
            }
        }
        __syncthreads();
        if (normalize) {
            sampling_normalize_support(cfg, cand_val, cand_idx, prob, n_support, cap);
        } else {
            if (tid == 0) { *n_support = cap; }
            __syncthreads();
        }
        return;
    }

    if (tid == 0) {
        for (int j = 0; j < cap; ++j) {
            cand_val[j] = -CUDART_INF_F;
            cand_idx[j] = INT_MAX;
        }
        const int n = partial_blocks * cap;
        for (int p = 0; p < n; ++p) {
            const int partial = p / cap;
            const int j = p - partial * cap;
            const int off = sampling_partial_offset(col, partial, j);
            const int idx = sampling_partial_idx[off];
            if (idx == INT_MAX) { continue; }
            sampling_insert_candidate(cand_val, cand_idx, cap, sampling_partial_val[off], idx);
        }
    }
    __syncthreads();
    if (normalize) {
        sampling_normalize_support(cfg, cand_val, cand_idx, prob, n_support, cap);
    } else {
        if (tid == 0) { *n_support = cap; }
        __syncthreads();
    }
}

// thread-0 helper: inverse-CDF pick over a normalized `prob[0..n-1]` support,
// optionally excluding `exclude` (a rejected draft token) and renormalizing over
// the remainder. `u` is a uniform in [0,1). Returns the chosen vocab id.
__device__ __forceinline__ int sampling_pick_from_support(const int* cand_idx, const float* prob,
                                                          int n, int exclude, float u) {
    float mass = 0.0f;
    for (int j = 0; j < n; ++j) {
        if (cand_idx[j] == exclude) { continue; }
        mass += prob[j];
    }
    if (mass <= 0.0f) {
        // Degenerate: support is only the excluded token. Return it (caller only
        // reaches this when accept probability was ~1, i.e. it will not happen).
        return cand_idx[0];
    }
    const float goal = u * mass;
    float acc        = 0.0f;
    int picked       = -1;
    for (int j = 0; j < n; ++j) {
        if (cand_idx[j] == exclude) { continue; }
        acc += prob[j];
        picked = cand_idx[j];
        if (goal < acc) { return picked; }
    }
    return picked;
}

} // namespace qus::kernels
