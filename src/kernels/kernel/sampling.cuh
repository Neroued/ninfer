#pragma once

// qus::kernels - sample_column kernel. One CUDA block handles one logits column
// and reduces over vocab. The greedy branch (temperature <= 0) is an exact
// argmax with lowest-index tie-break; the sampling branch builds the truncated
// target distribution and draws a token with a stateless counter-based RNG.

#include "kernels/kernel/sampling_device.cuh"

namespace qus::kernels {

__launch_bounds__(kSamplerBlock) __global__ void sample_column_kernel(
    const __nv_bfloat16* logits, std::int32_t* out, const SamplingConfig* cfg_ptr,
    const std::int32_t* pos_base, std::int32_t purpose, std::int32_t vocab) {
    const int t              = static_cast<int>(blockIdx.x);
    const std::int64_t base  = static_cast<std::int64_t>(t) * vocab;
    const int tid            = threadIdx.x;
    const SamplingConfig cfg = *cfg_ptr;

    __shared__ float red_val[kSamplerBlock];
    __shared__ int red_idx[kSamplerBlock];

    // Greedy: exact argmax over raw logits. Bit-identical to argmax().
    if (!(cfg.temperature > 0.0f)) {
        float bv = -CUDART_INF_F;
        int bi   = INT_MAX;
        for (int v = tid; v < vocab; v += blockDim.x) {
            const float x = __bfloat162float(logits[base + v]);
            if (sampling_better(x, v, bv, bi)) {
                bv = x;
                bi = v;
            }
        }
        red_val[tid] = bv;
        red_idx[tid] = bi;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s &&
                sampling_better(red_val[tid + s], red_idx[tid + s], red_val[tid], red_idx[tid])) {
                red_val[tid] = red_val[tid + s];
                red_idx[tid] = red_idx[tid + s];
            }
            __syncthreads();
        }
        if (tid == 0) { out[t] = red_idx[0]; }
        return;
    }

    const int partial_blocks = (vocab + kSamplerPartialTileItems - 1) / kSamplerPartialTileItems;
    const bool scratch_capable =
        (gridDim.x <= kSamplerScratchColumns) && (partial_blocks <= kSamplerScratchPartialBlocks);
    if (vocab > kSamplerTileItems && scratch_capable) { return; }

    __shared__ float cand_val[kSamplerMaxCandidates];
    __shared__ int cand_idx[kSamplerMaxCandidates];
    __shared__ float prob[kSamplerMaxCandidates];
    __shared__ int n_support;
    __shared__ float merge_val[kSamplerBlock * kSamplerFastCandidates];
    __shared__ int merge_idx[kSamplerBlock * kSamplerFastCandidates];

    if (vocab <= kSamplerTileItems) {
        sampling_build_truncated_small(logits, base, vocab, cfg, red_val, red_idx, cand_val,
                                       cand_idx, prob, &n_support);
    } else {
        sampling_build_truncated_block_fast(logits, base, vocab, cfg, merge_val, merge_idx,
                                            cand_val, cand_idx, prob, &n_support);
    }

    if (tid != 0) { return; }
    const int support = n_support;
    const float u     = sampling_uniform(cfg.seed, *pos_base + t, purpose, 0u);
    float acc         = 0.0f;
    int picked        = cand_idx[support - 1];
    for (int j = 0; j < support; ++j) {
        acc += prob[j];  // prob is normalized: goal == u
        if (u < acc) {
            picked = cand_idx[j];
            break;
        }
    }
    out[t] = picked;
    if (cfg.token_counts != nullptr) { atomicAdd(&cfg.token_counts[picked], 1); }
}

__launch_bounds__(kSamplerBlock) __global__ void sampling_partial_topk_kernel(
    const __nv_bfloat16* logits, const SamplingConfig* cfg_ptr, std::int32_t vocab) {
    const int col            = static_cast<int>(blockIdx.y);
    const int partial        = static_cast<int>(blockIdx.x);
    const SamplingConfig cfg = *cfg_ptr;
    const int partial_blocks = static_cast<int>(gridDim.x);
    if (col >= kSamplerScratchColumns || partial_blocks > kSamplerScratchPartialBlocks) {
        return;
    }

    __shared__ typename SamplingPartialMergeSort::TempStorage sort_storage;
    unsigned long long keys[kSamplerItemsPerThread];
    int items[kSamplerItemsPerThread];

    const bool greedy = !(cfg.temperature > 0.0f);
    const int cap = greedy ? 1 : sampling_candidate_cap(cfg, vocab);
    const std::int64_t base = static_cast<std::int64_t>(col) * vocab;
    const int tile_start = partial * kSamplerPartialTileItems;
#pragma unroll
    for (int item = 0; item < kSamplerItemsPerThread; ++item) {
        const int v = tile_start + item * blockDim.x + threadIdx.x;
        if (v < vocab) {
            const float raw = __bfloat162float(logits[base + v]);
            const float x = greedy ? raw : sampling_adjusted_logit(raw, v, cfg);
            keys[item] = sampling_sort_key(x, v);
            items[item] = v;
        } else {
            keys[item] = 0ull;
            items[item] = INT_MAX;
        }
    }
    SamplingPartialMergeSort(sort_storage).Sort(keys, items, SamplingKeyGreater{});

#pragma unroll
    for (int item = 0; item < kSamplerItemsPerThread; ++item) {
        const int rank = threadIdx.x * kSamplerItemsPerThread + item;
        if (rank < cap) {
            const int off = sampling_partial_offset(col, partial, rank);
            sampling_partial_val[off] = sampling_key_float(keys[item]);
            sampling_partial_idx[off] = items[item];
        }
    }
}

__launch_bounds__(kSamplerBlock) __global__ void sampling_fused_sample_kernel(
    const __nv_bfloat16* logits, std::int32_t* out, const SamplingConfig* cfg_ptr,
    const std::int32_t* pos_base, std::int32_t purpose, std::int32_t vocab,
    std::int32_t partial_blocks, std::int32_t group_count) {
    const int col            = static_cast<int>(blockIdx.y);
    const int partial        = static_cast<int>(blockIdx.x);
    const int tid            = threadIdx.x;
    const SamplingConfig cfg = *cfg_ptr;
    if (col >= kSamplerScratchColumns || partial_blocks > kSamplerScratchPartialBlocks ||
        partial_blocks + group_count > kSamplerScratchPartialBlocks) {
        return;
    }

    __shared__ SamplingFusedShared fused_shared;
    unsigned long long partial_keys[kSamplerFusedItemsPerThread];
    int partial_items[kSamplerFusedItemsPerThread];

    const bool greedy = !(cfg.temperature > 0.0f);
    const int cap = greedy ? 1 : sampling_candidate_cap(cfg, vocab);
    const std::int64_t base = static_cast<std::int64_t>(col) * vocab;
    const int tile_start = partial * kSamplerFusedPartialTileItems;
#pragma unroll
    for (int item = 0; item < kSamplerFusedItemsPerThread; ++item) {
        const int v = tile_start + item * blockDim.x + tid;
        if (v < vocab) {
            const float raw = __bfloat162float(logits[base + v]);
            const float x = greedy ? raw : sampling_adjusted_logit(raw, v, cfg);
            partial_keys[item] = sampling_sort_key(x, v);
            partial_items[item] = v;
        } else {
            partial_keys[item] = 0ull;
            partial_items[item] = INT_MAX;
        }
    }
    SamplingFusedPartialMergeSort(fused_shared.partial_sort_storage)
        .Sort(partial_keys, partial_items, SamplingKeyGreater{});

#pragma unroll
    for (int item = 0; item < kSamplerFusedItemsPerThread; ++item) {
        const int rank = tid * kSamplerFusedItemsPerThread + item;
        if (rank < cap) {
            const int off = sampling_partial_offset(col, partial, rank);
            sampling_partial_val[off] = sampling_key_float(partial_keys[item]);
            sampling_partial_idx[off] = partial_items[item];
        }
    }
    __syncthreads();

    if (tid == 0) {
        __threadfence();
        atomicAdd(&sampling_partial_done[col], 1);
    }
    if (partial >= group_count) { return; }
    if (tid == 0) {
        while (atomicAdd(&sampling_partial_done[col], 0) < partial_blocks) {}
    }
    __syncthreads();

    SamplingFusedGroupShared& group_shared = fused_shared.group;
    unsigned long long keys[kSamplerGroupItemsPerThread];
    int items[kSamplerGroupItemsPerThread];

    const int group = partial;
    const int group_begin = group * kSamplerPartialsPerGroup;
    int group_partials = partial_blocks - group_begin;
    if (group_partials < 0) { group_partials = 0; }
    if (group_partials > kSamplerPartialsPerGroup) { group_partials = kSamplerPartialsPerGroup; }
    const int group_n = group_partials * cap;
#pragma unroll
    for (int item = 0; item < kSamplerGroupItemsPerThread; ++item) {
        const int p = item * blockDim.x + tid;
        if (p < group_n) {
            const int src_partial = group_begin + p / cap;
            const int j = p - (p / cap) * cap;
            const int off = sampling_partial_offset(col, src_partial, j);
            const int idx = sampling_partial_idx[off];
            const float v = sampling_partial_val[off];
            keys[item] = sampling_sort_key(v, idx);
            items[item] = idx;
        } else {
            keys[item] = 0ull;
            items[item] = INT_MAX;
        }
    }
    SamplingGroupMergeSort(group_shared.sort_storage).Sort(keys, items, SamplingKeyGreater{});

#pragma unroll
    for (int item = 0; item < kSamplerGroupItemsPerThread; ++item) {
        const int rank = tid * kSamplerGroupItemsPerThread + item;
        if (rank < cap) {
            const int out_off = sampling_partial_offset(col, partial_blocks + group, rank);
            sampling_partial_val[out_off] = sampling_key_float(keys[item]);
            sampling_partial_idx[out_off] = items[item];
        }
    }
    __syncthreads();

    if (tid == 0) {
        __threadfence();
        const int done = atomicAdd(&sampling_group_done[col], 1) + 1;
        group_shared.is_last = (done == group_count) ? 1 : 0;
    }
    __syncthreads();
    if (!group_shared.is_last) { return; }

    const int final_n = group_count * cap;
#pragma unroll
    for (int item = 0; item < kSamplerGroupItemsPerThread; ++item) {
        const int p = item * blockDim.x + tid;
        if (p < final_n) {
            const int src_partial = partial_blocks + p / cap;
            const int j = p - (p / cap) * cap;
            const int off = sampling_partial_offset(col, src_partial, j);
            const int idx = sampling_partial_idx[off];
            const float v = sampling_partial_val[off];
            keys[item] = sampling_sort_key(v, idx);
            items[item] = idx;
        } else {
            keys[item] = 0ull;
            items[item] = INT_MAX;
        }
    }
    SamplingGroupMergeSort(group_shared.sort_storage).Sort(keys, items, SamplingKeyGreater{});

#pragma unroll
    for (int item = 0; item < kSamplerGroupItemsPerThread; ++item) {
        const int rank = tid * kSamplerGroupItemsPerThread + item;
        if (rank < cap) {
            group_shared.cand_val[rank] = sampling_key_float(keys[item]);
            group_shared.cand_idx[rank] = items[item];
        }
    }
    __syncthreads();

    if (greedy) {
        if (tid == 0) {
            out[col] = group_shared.cand_idx[0];
            sampling_group_done[col] = 0;
            sampling_partial_done[col] = 0;
        }
        return;
    }

    sampling_normalize_support(cfg, group_shared.cand_val, group_shared.cand_idx, group_shared.prob,
                               &group_shared.n_support, cap);
    if (tid == 0) {
        const int support = group_shared.n_support;
        const float u = sampling_uniform(cfg.seed, *pos_base + col, purpose, 0u);
        float acc = 0.0f;
        int picked = group_shared.cand_idx[support - 1];
        for (int j = 0; j < support; ++j) {
            acc += group_shared.prob[j];
            if (u < acc) {
                picked = group_shared.cand_idx[j];
                break;
            }
        }
        out[col] = picked;
        if (cfg.token_counts != nullptr) { atomicAdd(&cfg.token_counts[picked], 1); }
        sampling_group_done[col] = 0;
        sampling_partial_done[col] = 0;
    }
}

__launch_bounds__(kSamplerBlock) __global__ void sampling_finalize_sample_kernel(
    std::int32_t* out, const SamplingConfig* cfg_ptr, const std::int32_t* pos_base,
    std::int32_t purpose, std::int32_t vocab, std::int32_t partial_blocks) {
    const int col            = static_cast<int>(blockIdx.x);
    const int tid            = threadIdx.x;
    const SamplingConfig cfg = *cfg_ptr;
    if (gridDim.x > kSamplerScratchColumns || partial_blocks > kSamplerScratchPartialBlocks) {
        return;
    }

    __shared__ float merge_val[(kSamplerBlock / 32) * kSamplerFastCandidates];
    __shared__ int merge_idx[(kSamplerBlock / 32) * kSamplerFastCandidates];
    __shared__ typename SamplingFinalizeBlockSort::TempStorage sort_storage;
    __shared__ float cand_val[kSamplerMaxCandidates];
    __shared__ int cand_idx[kSamplerMaxCandidates];
    __shared__ float prob[kSamplerMaxCandidates];
    __shared__ int n_support;

    const bool greedy = !(cfg.temperature > 0.0f);
    const int cap = greedy ? 1 : sampling_candidate_cap(cfg, vocab);
    sampling_merge_partials_to_support(col, partial_blocks, cfg, sort_storage, merge_val,
                                       merge_idx, cand_val, cand_idx, prob, &n_support, vocab,
                                       cap, !greedy);

    if (tid != 0) { return; }
    if (greedy) {
        out[col] = cand_idx[0];
        return;
    }
    const int support = n_support;
    const float u     = sampling_uniform(cfg.seed, *pos_base + col, purpose, 0u);
    float acc         = 0.0f;
    int picked        = cand_idx[support - 1];
    for (int j = 0; j < support; ++j) {
        acc += prob[j];
        if (u < acc) {
            picked = cand_idx[j];
            break;
        }
    }
    out[col] = picked;
    if (cfg.token_counts != nullptr) { atomicAdd(&cfg.token_counts[picked], 1); }
}

__launch_bounds__(kSamplerBlock) __global__ void sampling_group_finalize_sample_kernel(
    std::int32_t* out, const SamplingConfig* cfg_ptr, const std::int32_t* pos_base,
    std::int32_t purpose, std::int32_t vocab, std::int32_t partial_blocks,
    std::int32_t group_count) {
    const int group          = static_cast<int>(blockIdx.x);
    const int col            = static_cast<int>(blockIdx.y);
    const int tid            = threadIdx.x;
    const SamplingConfig cfg = *cfg_ptr;
    if (col >= kSamplerScratchColumns || partial_blocks > kSamplerScratchPartialBlocks ||
        partial_blocks + group_count > kSamplerScratchPartialBlocks) {
        return;
    }

    __shared__ typename SamplingGroupMergeSort::TempStorage sort_storage;
    __shared__ float cand_val[kSamplerMaxCandidates];
    __shared__ int cand_idx[kSamplerMaxCandidates];
    __shared__ float prob[kSamplerMaxCandidates];
    __shared__ int n_support;
    __shared__ int is_last;
    unsigned long long keys[kSamplerGroupItemsPerThread];
    int items[kSamplerGroupItemsPerThread];

    const bool greedy = !(cfg.temperature > 0.0f);
    const int cap = greedy ? 1 : sampling_candidate_cap(cfg, vocab);
    if (tid == 0 && atomicCAS(&sampling_group_init[col], 0, 1) == 0) {
        sampling_group_done[col] = 0;
        __threadfence();
    }

    const int group_begin = group * kSamplerPartialsPerGroup;
    int group_partials = partial_blocks - group_begin;
    if (group_partials < 0) { group_partials = 0; }
    if (group_partials > kSamplerPartialsPerGroup) { group_partials = kSamplerPartialsPerGroup; }
    const int group_n = group_partials * cap;
#pragma unroll
    for (int item = 0; item < kSamplerGroupItemsPerThread; ++item) {
        const int p = item * blockDim.x + tid;
        if (p < group_n) {
            const int partial = group_begin + p / cap;
            const int j = p - (p / cap) * cap;
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
    SamplingGroupMergeSort(sort_storage).Sort(keys, items, SamplingKeyGreater{});

#pragma unroll
    for (int item = 0; item < kSamplerGroupItemsPerThread; ++item) {
        const int rank = tid * kSamplerGroupItemsPerThread + item;
        if (rank < cap) {
            const int out_off = sampling_partial_offset(col, partial_blocks + group, rank);
            sampling_partial_val[out_off] = sampling_key_float(keys[item]);
            sampling_partial_idx[out_off] = items[item];
        }
    }
    __syncthreads();

    if (tid == 0) {
        __threadfence();
        const int done = atomicAdd(&sampling_group_done[col], 1) + 1;
        is_last = (done == group_count) ? 1 : 0;
    }
    __syncthreads();
    if (!is_last) { return; }

    const int final_n = group_count * cap;
#pragma unroll
    for (int item = 0; item < kSamplerGroupItemsPerThread; ++item) {
        const int p = item * blockDim.x + tid;
        if (p < final_n) {
            const int partial = partial_blocks + p / cap;
            const int j = p - (p / cap) * cap;
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
    SamplingGroupMergeSort(sort_storage).Sort(keys, items, SamplingKeyGreater{});

#pragma unroll
    for (int item = 0; item < kSamplerGroupItemsPerThread; ++item) {
        const int rank = tid * kSamplerGroupItemsPerThread + item;
        if (rank < cap) {
            cand_val[rank] = sampling_key_float(keys[item]);
            cand_idx[rank] = items[item];
        }
    }
    __syncthreads();

    if (greedy) {
        if (tid == 0) {
            out[col] = cand_idx[0];
            sampling_group_done[col] = 0;
            atomicExch(&sampling_group_init[col], 0);
        }
        return;
    }

    sampling_normalize_support(cfg, cand_val, cand_idx, prob, &n_support, cap);
    if (tid == 0) {
        const int support = n_support;
        const float u = sampling_uniform(cfg.seed, *pos_base + col, purpose, 0u);
        float acc = 0.0f;
        int picked = cand_idx[support - 1];
        for (int j = 0; j < support; ++j) {
            acc += prob[j];
            if (u < acc) {
                picked = cand_idx[j];
                break;
            }
        }
        out[col] = picked;
        if (cfg.token_counts != nullptr) { atomicAdd(&cfg.token_counts[picked], 1); }
        sampling_group_done[col] = 0;
        atomicExch(&sampling_group_init[col], 0);
    }
}

} // namespace qus::kernels
