#pragma once

// qus::kernels - shared device-side sampling primitives (no __global__ symbols,
// so this header can be included by multiple translation units). Holds the
// counter-based RNG, the candidate ordering, and the block-collaborative
// truncated-distribution builder used by both sample and the MTP
// rejection-sampling accept kernel.

#include "kernels/common/math.h"
#include "qus/kernels/sampling.h"

#include <cub/block/block_merge_sort.cuh>

#include <cuda_bf16.h>
#include <climits>
#include <cstdint>
#include <math_constants.h>

namespace qus::kernels {

inline constexpr int kSamplerBlock         = 256;
inline constexpr int kSamplerTileItems     = 256;
inline constexpr int kSamplerItemsPerThread = 3;
inline constexpr int kSamplerPartialTileItems = kSamplerBlock * kSamplerItemsPerThread;
inline constexpr int kSamplerGroupItemsPerThread = 2;
inline constexpr int kSamplerGroupTileItems = kSamplerBlock * kSamplerGroupItemsPerThread;
inline constexpr int kSamplerPartialsPerGroup = 20;
inline constexpr int kSamplerFastCandidates = 20;
// Effective per-column candidate cap. The multi-block partial/group merge tiles
// are sized for this cap, so the sampler clamps top_k to it: top_k <= 0 or a
// larger top_k both collapse to this cap.
inline constexpr int kSamplerCandidateCap = kSamplerFastCandidates;
inline constexpr int kSamplerScratchColumns = 8;
inline constexpr int kSamplerScratchPartialBlocks = 1024;

// One group merge sorts a single kSamplerGroupTileItems-wide tile. It must hold
// one group's inputs (kSamplerPartialsPerGroup * cap); the final stage's
// group_count * cap is enforced per launch by sampler_multiblock_ok().
static_assert(kSamplerPartialsPerGroup * kSamplerCandidateCap <= kSamplerGroupTileItems,
              "group merge tile must hold one group's candidates");

// Number of group blocks needed to merge `partial_blocks` partials.
__host__ __device__ inline int sampler_group_count(int partial_blocks) {
    return div_up(partial_blocks, kSamplerPartialsPerGroup);
}

// True when the fixed sampler scratch and merge tiles can represent this launch.
// The launcher and the single-block kernels share this predicate so the
// single-block fallback and the scratch path never both claim the same shape.
__host__ __device__ inline bool sampler_multiblock_ok(int vocab, int cols, int partial_blocks,
                                                      int group_count) {
    return vocab > kSamplerTileItems && cols <= kSamplerScratchColumns &&
           partial_blocks <= kSamplerScratchPartialBlocks &&
           (partial_blocks + group_count) <= kSamplerScratchPartialBlocks &&
           (group_count * kSamplerCandidateCap) <= kSamplerGroupTileItems;
}

static __device__ unsigned long long sampling_partial_key[kSamplerScratchColumns *
                                                          kSamplerScratchPartialBlocks *
                                                          kSamplerCandidateCap];
static __device__ int sampling_dist_idx[kSamplerScratchColumns * kSamplerCandidateCap];
static __device__ float sampling_dist_prob[kSamplerScratchColumns * kSamplerCandidateCap];
static __device__ int sampling_dist_support[kSamplerScratchColumns];
static __device__ int sampling_mtp_finalize_init;
static __device__ int sampling_mtp_finalize_count;
static __device__ int sampling_group_done[kSamplerScratchColumns];

using SamplingPartialMergeSort =
    cub::BlockMergeSort<unsigned long long, kSamplerBlock, kSamplerItemsPerThread>;
using SamplingGroupMergeSort =
    cub::BlockMergeSort<unsigned long long, kSamplerBlock, kSamplerGroupItemsPerThread>;

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

__device__ __forceinline__ int sampling_key_index(unsigned long long key) {
    if (key == 0ull) { return INT_MAX; }
    return static_cast<int>(0xffffffffu - static_cast<unsigned int>(key));
}

__device__ __forceinline__ int sampling_candidate_cap(const SamplingConfig& cfg,
                                                      std::int32_t vocab) {
    // top_k is clamped to the pipeline cap: top_k <= 0 (no explicit limit) or a
    // top_k larger than the cap both select the full kSamplerCandidateCap set.
    int cap = kSamplerCandidateCap;
    if (cfg.top_k > 0 && cfg.top_k < cap) { cap = cfg.top_k; }
    if (vocab < cap) { cap = vocab; }
    return cap;
}

__device__ __forceinline__ int sampling_partial_offset(int col, int partial, int j) {
    return ((col * kSamplerScratchPartialBlocks + partial) * kSamplerCandidateCap) + j;
}

__device__ __forceinline__ int sampling_dist_offset(int col, int j) {
    return col * kSamplerCandidateCap + j;
}

// Applies presence/frequency penalties to a raw logit. `overlay`/`overlay_len`
// carry a round-local count overlay: tokens already committed earlier in the
// current MTP round but not yet flushed to the global `token_counts`. For MTP
// verify column `col` the overlay is exactly drafts[0..col-1] (statically known,
// since column `col` is only consumed when every earlier draft was accepted), so
// the penalty at each column sees the same prefix a per-token sampler would.
// Non-MTP callers pass no overlay. The scan is bounded by k (a few tokens) and
// only runs when penalties are active, so it is free on the no-penalty path.
__device__ __forceinline__ float sampling_adjusted_logit(float raw, int v,
                                                         const SamplingConfig& c,
                                                         const std::int32_t* overlay = nullptr,
                                                         int overlay_len = 0) {
    float x = raw;
    if (c.token_counts != nullptr) {
        int cnt = c.token_counts[v];
        for (int j = 0; j < overlay_len; ++j) {
            if (overlay[j] == v) { ++cnt; }
        }
        if (cnt > 0) { x -= c.presence_penalty; }
        if (c.frequency_penalty != 0.0f) {
            x -= c.frequency_penalty * static_cast<float>(cnt);
        }
    }
    return x;
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
        const float inv_temp = 1.0f / cfg.temperature;
        const float m = cand_val[0] * inv_temp;
        float sum     = 0.0f;
        for (int j = 0; j < n; ++j) {
            const float e = __expf(cand_val[j] * inv_temp - m);
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
                                                      int* n_support,
                                                      const std::int32_t* overlay = nullptr,
                                                      int overlay_len = 0) {
    const int tid = threadIdx.x;
    const int cap = sampling_candidate_cap(cfg, vocab);
    if (tid < kSamplerTileItems) {
        if (tid < vocab) {
            const float x = sampling_adjusted_logit(__bfloat162float(logits[base + tid]), tid, cfg,
                                                    overlay, overlay_len);
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
                                                           float* prob, int* n_support,
                                                           const std::int32_t* overlay = nullptr,
                                                           int overlay_len = 0) {
    const int tid = threadIdx.x;
    const int cap = sampling_candidate_cap(cfg, vocab);  // always <= kSamplerFastCandidates

    float local_val[kSamplerFastCandidates];
    int local_idx[kSamplerFastCandidates];
#pragma unroll
    for (int j = 0; j < kSamplerFastCandidates; ++j) {
        local_val[j] = -CUDART_INF_F;
        local_idx[j] = INT_MAX;
    }

    const int fast_cap = cap;
    for (int v = tid; v < vocab; v += blockDim.x) {
        const float x = sampling_adjusted_logit(__bfloat162float(logits[base + v]), v, cfg,
                                                overlay, overlay_len);
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
