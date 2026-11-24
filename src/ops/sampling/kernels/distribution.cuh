#pragma once

#include "ops/sampling/kernels/random.cuh"
#include "ops/common/numeric/math.h"
#include "ops/sampling/kernels/workspace.h"
#include "ops/common/cuda/reduce.cuh"
#include "ninfer/ops/sampling.h"

#include <cub/block/block_merge_sort.cuh>
#include <cub/warp/warp_merge_sort.cuh>

#include <cuda_bf16.h>
#include <climits>
#include <cstdint>
#include <math_constants.h>

namespace ninfer::ops::sampling_detail {
using namespace common;

struct TokenRow {
    const std::int32_t* data = nullptr;
    std::int64_t stride      = 1;

    __host__ __device__ TokenRow() = default;

    __host__ __device__ TokenRow(const std::int32_t* data, std::int64_t stride = 1)
        : data(data), stride(stride) {}

    __device__ std::int32_t operator[](int column) const { return data[column * stride]; }

    __device__ explicit operator bool() const { return data != nullptr; }
};

struct TokenRows {
    const std::int32_t* data   = nullptr;
    std::int64_t column_stride = 1, row_stride = 0;

    __device__ TokenRow operator[](int row) const {
        return {data ? data + row * row_stride : nullptr, column_stride};
    }
};

inline TokenRows token_rows(const Tensor& tensor) {
    return {static_cast<const std::int32_t*>(tensor.data), tensor.nb[0] / 4, tensor.nb[1] / 4};
}

template <class S>
using SamplingPartialSort =
    cub::BlockMergeSort<unsigned long long, S::kSamplerBlock, S::kSamplerItemsPerThread>;
template <class S>
using SamplingGroupSort =
    cub::BlockMergeSort<unsigned long long, S::kSamplerGroupBlock, S::kSamplerGroupItemsPerThread>;

template <class S>
using SamplingTileWarpSort = cub::WarpMergeSort<unsigned long long, S::kSamplerItemsPerThread>;
template <class S>
using SamplingBf16TileWarpSort = cub::WarpMergeSort<unsigned int, S::kSamplerItemsPerThread>;

template <class S>
struct SamplingTileTopKStorage {
    typename SamplingTileWarpSort<S>::TempStorage warp[S::kSamplingTileWarps];
    unsigned long long candidates[S::kSamplingTileWarps * kSamplerCandidateCap];
};

template <class S>
struct SamplingBf16TileTopKStorage {
    typename SamplingBf16TileWarpSort<S>::TempStorage warp[S::kSamplingTileWarps];
    unsigned int candidates[S::kSamplingTileWarps * kSamplerCandidateCap];
};

template <class S>
union SamplingPartialTopKStorage {
    SamplingTileTopKStorage<S> fp32;
    SamplingBf16TileTopKStorage<S> bf16;
};

template <class S>
__device__ __forceinline__ unsigned int sampling_bf16_tile_sort_key(__nv_bfloat16 v,
                                                                    int tile_index) {
    unsigned int bits = __bfloat16_as_ushort(v);
    if ((bits & 0x7fffu) == 0u) { bits = 0u; }
    const unsigned int ordered = (bits & 0x8000u) ? ((~bits) & 0xffffu) : (bits ^ 0x8000u);
    return (ordered << S::kSamplingTileIndexBits) |
           (S::kSamplingTileIndexMask - static_cast<unsigned int>(tile_index));
}

template <class S>
__device__ __forceinline__ unsigned long long sampling_expand_bf16_tile_sort_key(unsigned int key,
                                                                                 int tile_start) {
    if (key == 0u) { return 0ull; }
    const unsigned int ordered = key >> S::kSamplingTileIndexBits;
    const unsigned int bits    = (ordered & 0x8000u) ? (ordered ^ 0x8000u) : ((~ordered) & 0xffffu);
    const int token            = tile_start + static_cast<int>(S::kSamplingTileIndexMask) -
                      static_cast<int>(key & S::kSamplingTileIndexMask);
    return score_id_order_key(__uint_as_float(bits << 16), token);
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

__device__ __forceinline__ int sampling_partial_offset(const Workspace& workspace, int col,
                                                       int partial, int j) {
    return ((col * workspace.partial_stride + partial) * kSamplerCandidateCap) + j;
}

template <class S>
__device__ inline void
sampling_store_tile_topk(unsigned long long (&keys)[S::kSamplerItemsPerThread], int cap,
                         Workspace workspace, int col, int partial,
                         SamplingTileTopKStorage<S>& storage) {
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    SamplingTileWarpSort<S>(storage.warp[warp]).Sort(keys, ScoreIdOrderGreater{});
#pragma unroll
    for (int item = 0; item < S::kSamplerItemsPerThread; ++item) {
        const int rank = lane * S::kSamplerItemsPerThread + item;
        if (rank < kSamplerCandidateCap) {
            storage.candidates[warp * kSamplerCandidateCap + rank] = keys[item];
        }
    }
    __syncthreads();

    if (warp != 0) { return; }
    if (lane < S::kSamplingTileWarps) {
        constexpr unsigned int kMergeMask = (1u << S::kSamplingTileWarps) - 1u;
        int position                      = 0;
        for (int rank = 0; rank < cap; ++rank) {
            const unsigned long long key =
                storage.candidates[lane * kSamplerCandidateCap + position];
            const unsigned int high     = static_cast<unsigned int>(key >> 32);
            const unsigned int max_high = __reduce_max_sync(kMergeMask, high);
            const unsigned int low      = high == max_high ? static_cast<unsigned int>(key) : 0u;
            const unsigned int max_low  = __reduce_max_sync(kMergeMask, low);
            const unsigned int winners =
                __ballot_sync(kMergeMask, high == max_high && low == max_low);
            const int source = __ffs(static_cast<int>(winners)) - 1;
            if (lane == 0) {
                const int off = sampling_partial_offset(workspace, col, partial, rank);
                workspace.partial_keys[off] =
                    (static_cast<unsigned long long>(max_high) << 32) | max_low;
            }
            if (lane == source) { ++position; }
        }
    }
}

template <class S>
__device__ inline void
sampling_store_bf16_tile_topk(unsigned int (&keys)[S::kSamplerItemsPerThread], int cap,
                              int tile_start, Workspace workspace, int col, int partial,
                              SamplingBf16TileTopKStorage<S>& storage) {
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    SamplingBf16TileWarpSort<S>(storage.warp[warp]).Sort(keys, ScoreIdOrderGreater{});
#pragma unroll
    for (int item = 0; item < S::kSamplerItemsPerThread; ++item) {
        const int rank = lane * S::kSamplerItemsPerThread + item;
        if (rank < kSamplerCandidateCap) {
            storage.candidates[warp * kSamplerCandidateCap + rank] = keys[item];
        }
    }
    __syncthreads();

    if (warp != 0) { return; }
    if (lane < S::kSamplingTileWarps) {
        constexpr unsigned int kMergeMask = (1u << S::kSamplingTileWarps) - 1u;
        int position                      = 0;
        for (int rank = 0; rank < cap; ++rank) {
            const unsigned int key     = storage.candidates[lane * kSamplerCandidateCap + position];
            const unsigned int best    = __reduce_max_sync(kMergeMask, key);
            const unsigned int winners = __ballot_sync(kMergeMask, key == best);
            const int source           = __ffs(static_cast<int>(winners)) - 1;
            if (lane == 0) {
                const int off = sampling_partial_offset(workspace, col, partial, rank);
                workspace.partial_keys[off] =
                    sampling_expand_bf16_tile_sort_key<S>(best, tile_start);
            }
            if (lane == source) { ++position; }
        }
    }
}

__device__ __forceinline__ int sampling_dist_offset(int col, int j) {
    return col * kSamplerCandidateCap + j;
}

// Applies presence/frequency penalties to a raw logit. `overlay`/`overlay_len`
// carry a round-local count overlay: tokens already committed earlier in the
// current speculative round but not yet flushed to the global `token_counts`. For speculative
// verify column `col` the overlay is exactly drafts[0..col-1] (statically known,
// since column `col` is only consumed when every earlier draft was accepted), so
// the penalty at each column sees the same prefix a per-token sampler would.
// Non-speculative callers pass no overlay. The scan is bounded by k and
// only runs when penalties are active, so it is free on the no-penalty path.
__device__ __forceinline__ float sampling_adjusted_logit(float raw, int v, const SamplingConfig& c,
                                                         TokenRow overlay = {},
                                                         int overlay_len  = 0) {
    if (c.mask.words && !(c.mask.words[overlay_len * c.mask.stride + v / 32] & (1u << (v % 32)))) {
        return -CUDART_INF_F;
    }
    float x = raw;
    if (c.presence_penalty == 0.0f && c.frequency_penalty == 0.0f) { return x; }
    int cnt = c.token_counts != nullptr ? c.token_counts[v] : 0;
    for (int j = 0; j < overlay_len; ++j) {
        if (overlay[j] == v) { ++cnt; }
    }
    if (cnt > 0) { x -= c.presence_penalty; }
    if (c.frequency_penalty != 0.0f) { x -= c.frequency_penalty * static_cast<float>(cnt); }
    return x;
}

__device__ __forceinline__ void sampling_insert_candidate(float* vals, int* idxs, int cap, float v,
                                                          int idx) {
    if (cap <= 0 || !score_id_better(v, idx, vals[cap - 1], idxs[cap - 1])) { return; }
    int pos = cap - 1;
    while (pos > 0 && score_id_better(v, idx, vals[pos - 1], idxs[pos - 1])) {
        vals[pos] = vals[pos - 1];
        idxs[pos] = idxs[pos - 1];
        --pos;
    }
    vals[pos] = v;
    idxs[pos] = idx;
}

template <class S>
__device__ inline void sampling_sort_tile_desc(float* vals, int* idxs) {
    const int tid = threadIdx.x;
    for (int size = 2; size <= S::kSamplerTileItems; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            const int other = tid ^ stride;
            __syncthreads();
            if (other > tid) {
                const float ov        = vals[other];
                const int oi          = idxs[other];
                const bool descending = ((tid & size) == 0);
                const bool swap       = descending ? score_id_better(ov, oi, vals[tid], idxs[tid])
                                                   : score_id_better(vals[tid], idxs[tid], ov, oi);
                if (swap) {
                    const float tv = vals[tid];
                    const int ti   = idxs[tid];
                    vals[tid]      = ov;
                    idxs[tid]      = oi;
                    vals[other]    = tv;
                    idxs[other]    = ti;
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
        while (n > 0 && !isfinite(cand_val[n - 1])) { --n; }
        if (n == 0 || !isfinite(cand_val[0])) {
            // The caller provides nonempty masks. Missing finite support is a model execution
            // failure, not a distribution from which a placeholder token can be published.
            asm volatile("trap;");
        } else {
            const float inv_temp = 1.0f / cfg.temperature;
            const float m        = cand_val[0] * inv_temp;
            float sum            = 0.0f;
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
    }
    __syncthreads();
}

// All threads of the block must call. Small-column exact truncated support
// builder used by unit tests and any <=256-token column. It sorts one shared
// tile and then applies the same top-p/min-p renormalization as the large path.
template <class S>
__device__ inline void
sampling_build_truncated_small(const __nv_bfloat16* logits, std::int64_t base, std::int32_t vocab,
                               const SamplingConfig& cfg, float* tile_val, int* tile_idx,
                               float* cand_val, int* cand_idx, float* prob, int* n_support,
                               TokenRow overlay = {}, int overlay_len = 0) {
    const int tid = threadIdx.x;
    const int cap = sampling_candidate_cap(cfg, vocab);
    if (tid < S::kSamplerTileItems) {
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
    sampling_sort_tile_desc<S>(tile_val, tile_idx);
    if (tid < cap) {
        cand_val[tid] = tile_val[tid];
        cand_idx[tid] = tile_idx[tid];
    }
    __syncthreads();
    sampling_normalize_support(cfg, cand_val, cand_idx, prob, n_support, cap);
}

// Single-block fallback for large columns when the finite multi-block workspace
// route cannot represent the launch. It still reads each vocab entry once and
// keeps a bounded per-thread top-20, so it avoids a top_k*vocab global reread.
__device__ inline void
sampling_build_truncated_block_fast(const __nv_bfloat16* logits, std::int64_t base,
                                    std::int32_t vocab, const SamplingConfig& cfg, float* merge_val,
                                    int* merge_idx, float* cand_val, int* cand_idx, float* prob,
                                    int* n_support, TokenRow overlay = {}, int overlay_len = 0) {
    const int tid = threadIdx.x;
    const int cap = sampling_candidate_cap(cfg, vocab); // always <= kSamplerFastCandidates

    float local_val[kSamplerFastCandidates];
    int local_idx[kSamplerFastCandidates];
#pragma unroll
    for (int j = 0; j < kSamplerFastCandidates; ++j) {
        local_val[j] = -CUDART_INF_F;
        local_idx[j] = INT_MAX;
    }

    const int fast_cap = cap;
    for (int v = tid; v < vocab; v += blockDim.x) {
        const float x = sampling_adjusted_logit(__bfloat162float(logits[base + v]), v, cfg, overlay,
                                                overlay_len);
        sampling_insert_candidate(local_val, local_idx, fast_cap, x, v);
    }

    for (int j = 0; j < kSamplerFastCandidates; ++j) {
        const int off  = tid * kSamplerFastCandidates + j;
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

} // namespace ninfer::ops::sampling_detail
