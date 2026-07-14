#pragma once

// ninfer::kernels - sample kernel. One CUDA block handles one logits column
// and reduces over vocab. The greedy branch (temperature <= 0) is an exact
// argmax with lowest-index tie-break; the sampling branch builds the truncated
// target distribution and draws a token with a stateless counter-based RNG.

#include "kernels/sampling/device.cuh"

namespace ninfer::kernels {

__launch_bounds__(kSamplerBlock) __global__ void sample_column_kernel(
    const __nv_bfloat16* logits, std::int32_t* out, const SamplingConfig* cfg_ptr,
    const std::int32_t* pos_base, std::int32_t purpose, std::int32_t token_domain,
    std::int32_t physical_rows) {
    const int t              = static_cast<int>(blockIdx.x);
    const std::int64_t base  = static_cast<std::int64_t>(t) * physical_rows;
    const int tid            = threadIdx.x;
    const SamplingConfig cfg = *cfg_ptr;

    __shared__ float red_val[kSamplerBlock];
    __shared__ int red_idx[kSamplerBlock];

    // Greedy: exact argmax over raw logits. Bit-identical to argmax().
    if (!(cfg.temperature > 0.0f)) {
        float bv = -CUDART_INF_F;
        int bi   = INT_MAX;
        for (int v = tid; v < token_domain; v += blockDim.x) {
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

    const int partial_blocks = div_up(token_domain, kSamplerPartialTileItems);
    const int group_count    = sampler_group_count(partial_blocks);
    // No-op when the scratch/group path owns this shape (see sample_column_launch).
    if (sampler_multiblock_ok(token_domain, static_cast<int>(gridDim.x), partial_blocks,
                              group_count)) {
        return;
    }

    __shared__ float cand_val[kSamplerCandidateCap];
    __shared__ int cand_idx[kSamplerCandidateCap];
    __shared__ float prob[kSamplerCandidateCap];
    __shared__ int n_support;
    __shared__ float merge_val[kSamplerBlock * kSamplerFastCandidates];
    __shared__ int merge_idx[kSamplerBlock * kSamplerFastCandidates];

    if (token_domain <= kSamplerTileItems) {
        sampling_build_truncated_small(logits, base, token_domain, cfg, red_val, red_idx, cand_val,
                                       cand_idx, prob, &n_support);
    } else {
        sampling_build_truncated_block_fast(logits, base, token_domain, cfg, merge_val, merge_idx,
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
    const __nv_bfloat16* logits, const SamplingConfig* cfg_ptr, std::int32_t token_domain,
    std::int32_t physical_rows) {
    const int col            = static_cast<int>(blockIdx.y);
    const int partial        = static_cast<int>(blockIdx.x);
    const SamplingConfig cfg = *cfg_ptr;
    const int partial_blocks = static_cast<int>(gridDim.x);
    if (col >= kSamplerScratchColumns || partial_blocks > kSamplerScratchPartialBlocks) {
        return;
    }

    __shared__ typename SamplingPartialMergeSort::TempStorage sort_storage;
    unsigned long long keys[kSamplerItemsPerThread];

    const bool greedy = !(cfg.temperature > 0.0f);
    const int cap = greedy ? 1 : sampling_candidate_cap(cfg, token_domain);
    const std::int64_t base = static_cast<std::int64_t>(col) * physical_rows;
    const int tile_start = partial * kSamplerPartialTileItems;
#pragma unroll
    for (int item = 0; item < kSamplerItemsPerThread; ++item) {
        const int v = tile_start + item * blockDim.x + threadIdx.x;
        if (v < token_domain) {
            const float raw = __bfloat162float(logits[base + v]);
            const float x = greedy ? raw : sampling_adjusted_logit(raw, v, cfg);
            keys[item] = sampling_sort_key(x, v);
        } else {
            keys[item] = 0ull;
        }
    }
    SamplingPartialMergeSort(sort_storage).Sort(keys, SamplingKeyGreater{});

#pragma unroll
    for (int item = 0; item < kSamplerItemsPerThread; ++item) {
        const int rank = threadIdx.x * kSamplerItemsPerThread + item;
        if (rank < cap) {
            const int off = sampling_partial_offset(col, partial, rank);
            sampling_partial_key[off] = keys[item];
        }
    }
}

__launch_bounds__(kSamplerBlock) __global__ void sampling_group_finalize_sample_kernel(
    std::int32_t* out, const SamplingConfig* cfg_ptr, const std::int32_t* pos_base,
    std::int32_t purpose, std::int32_t token_domain, std::int32_t partial_blocks,
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
    __shared__ float cand_val[kSamplerCandidateCap];
    __shared__ int cand_idx[kSamplerCandidateCap];
    __shared__ float prob[kSamplerCandidateCap];
    __shared__ int n_support;
    __shared__ int is_last;
    unsigned long long keys[kSamplerGroupItemsPerThread];

    const bool greedy = !(cfg.temperature > 0.0f);
    const int cap = greedy ? 1 : sampling_candidate_cap(cfg, token_domain);
    // sampling_group_done[col] is 0 at entry: it is zero-initialized and the
    // terminating (last) group block resets it before returning, and consecutive
    // launches are ordered on the same stream. No per-entry re-init is needed
    // (the old atomicCAS re-init raced with the counting atomicAdds).

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
            keys[item] = sampling_partial_key[off];
        } else {
            keys[item] = 0ull;
        }
    }
    SamplingGroupMergeSort(sort_storage).Sort(keys, SamplingKeyGreater{});

#pragma unroll
    for (int item = 0; item < kSamplerGroupItemsPerThread; ++item) {
        const int rank = tid * kSamplerGroupItemsPerThread + item;
        if (rank < cap) {
            const int out_off = sampling_partial_offset(col, partial_blocks + group, rank);
            sampling_partial_key[out_off] = keys[item];
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
            keys[item] = sampling_partial_key[off];
        } else {
            keys[item] = 0ull;
        }
    }
    SamplingGroupMergeSort(sort_storage).Sort(keys, SamplingKeyGreater{});

#pragma unroll
    for (int item = 0; item < kSamplerGroupItemsPerThread; ++item) {
        const int rank = tid * kSamplerGroupItemsPerThread + item;
        if (rank < cap) {
            cand_val[rank] = sampling_key_float(keys[item]);
            cand_idx[rank] = sampling_key_index(keys[item]);
        }
    }
    __syncthreads();

    if (greedy) {
        if (tid == 0) {
            out[col] = cand_idx[0];
            sampling_group_done[col] = 0;
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
    }
}

} // namespace ninfer::kernels
