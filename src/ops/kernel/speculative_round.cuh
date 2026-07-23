#pragma once

// Implements: include/ninfer/ops/speculative_round.h
// Match: contiguous scalar/vector state and BF16 verification logits.
// Algorithm assumptions: small vocabularies use one cooperative block; the
// registered full-vocabulary stochastic route uses the sampling partial/group
// pipeline and caller-owned workspace, while greedy commit remains one thread.

#include "ops/kernel/sampling_device.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kSpeculativeAcceptedPerPositionOffset = 4;

__global__ void speculative_prepare_verify_inputs_kernel(const std::int32_t* token,
                                                         const std::int32_t* drafts,
                                                         const std::int32_t* length,
                                                         std::int32_t* verify_ids,
                                                         std::int32_t* positions, std::int32_t T) {
    for (int i = threadIdx.x; i < T; i += blockDim.x) {
        verify_ids[i] = i == 0 ? *token : drafts[i - 1];
        positions[i]  = *length + i;
    }
}

// Commits the round's accepted tokens plus one correction/bonus token, then
// advances the target length scalar and acceptance stats. The greedy branch
// (config temperature <= 0) is bit-identical to the original argmax accept: keep
// the longest draft prefix whose target argmax matches, then take the target
// argmax at the divergence column. The sampling branch (temperature > 0) runs
// distribution-correct speculative rejection sampling over the verify logits with
// a one-hot (greedy) draft: accept drafts[i] with probability p_i(drafts[i]) under
// the truncated target distribution, resample from the masked residual on the
// first rejection, and draw a bonus from the last column when every draft accepts.
// The draft-proposal path stays greedy, so q is one-hot and the accept test
// collapses to `u < p_i(drafts[i])`. Launch with a single block of kSamplerBlock
// threads; only thread 0 performs the sequential accept/commit while the whole
// block cooperates on the per-column truncated-distribution build.
__launch_bounds__(kSamplerBlock) __global__ void speculative_accept_greedy_drafts_kernel(
    const std::int32_t* target_tokens, const __nv_bfloat16* logits, const std::int32_t* drafts,
    std::int32_t* length, std::int32_t* token, std::int32_t* sampled_out, std::int32_t* num_sampled,
    std::int32_t* accepted, std::int64_t* stats, const SamplingConfig* cfg_ptr,
    std::int32_t token_domain, std::int32_t physical_rows, std::int32_t k) {
    const int tid            = threadIdx.x;
    const SamplingConfig cfg = *cfg_ptr;

    if (!(cfg.temperature > 0.0f)) {
        if (tid == 0) {
            int a = 0;
            while (a < k && target_tokens[a] == drafts[a]) { ++a; }
            const int t_star = target_tokens[a];

            for (int i = 0; i <= k; ++i) { sampled_out[i] = 0; }
            for (int i = 0; i < a; ++i) { sampled_out[i] = drafts[i]; }
            sampled_out[a] = t_star;

            const int produced = a + 1;
            *num_sampled       = produced;
            *accepted          = a;
            *token             = t_star;
            *length += produced;
            stats[0] += k;
            stats[1] += a;
            stats[2] += 1;
            for (int i = 0; i < a; ++i) { stats[kSpeculativeAcceptedPerPositionOffset + i] += 1; }
        }
        return;
    }

    __shared__ float red_val[kSamplerBlock];
    __shared__ int red_idx[kSamplerBlock];
    __shared__ float cand_val[kSamplerCandidateCap];
    __shared__ int cand_idx[kSamplerCandidateCap];
    __shared__ float prob[kSamplerCandidateCap];
    __shared__ float merge_val[kSamplerBlock * kSamplerFastCandidates];
    __shared__ int merge_idx[kSamplerBlock * kSamplerFastCandidates];
    __shared__ int n_support;
    __shared__ int a_sh;
    __shared__ int done_sh;
    __shared__ int tstar_sh;
    __shared__ int L_sh;

    const int partial_blocks = div_up(token_domain, kSamplerPartialTileItems);
    const int group_count    = sampler_group_count(partial_blocks);
    // No-op when the scratch/group path owns this shape.
    if (sampler_multiblock_ok(token_domain, k + 1, partial_blocks, group_count)) { return; }

    if (tid == 0) {
        a_sh     = 0;
        done_sh  = 0;
        tstar_sh = 0;
        L_sh     = *length;
    }
    __syncthreads();

    for (int i = 0; i <= k; ++i) {
        // Column i is only reached when drafts[0..i-1] were all accepted, so the
        // round-local penalty overlay for this column is exactly those i drafts.
        const std::int64_t base = static_cast<std::int64_t>(i) * physical_rows;
        if (token_domain <= kSamplerTileItems) {
            sampling_build_truncated_small(logits, base, token_domain, cfg, red_val, red_idx,
                                           cand_val, cand_idx, prob, &n_support, drafts, i);
        } else {
            sampling_build_truncated_block_fast(logits, base, token_domain, cfg, merge_val,
                                                merge_idx, cand_val, cand_idx, prob, &n_support,
                                                drafts, i);
        }
        if (tid == 0 && done_sh == 0) {
            const int L = L_sh;
            if (i < k) {
                const int d = drafts[i];
                float pd    = 0.0f;
                for (int j = 0; j < n_support; ++j) {
                    if (cand_idx[j] == d) {
                        pd = prob[j];
                        break;
                    }
                }
                const float u =
                    sampling_uniform(cfg.seed, L + i + 1, kSamplePurposeSpeculativeAccept, 0u);
                if (u < pd) {
                    a_sh = i + 1; // accept drafts[i], keep verifying
                } else {
                    const float ur = sampling_uniform(cfg.seed, L + i + 1,
                                                      kSamplePurposeSpeculativeCorrection, 0u);
                    tstar_sh       = sampling_pick_from_support(cand_idx, prob, n_support, d, ur);
                    done_sh        = 1;
                }
            } else {
                // Every draft accepted: bonus token from the last verify column.
                const float u =
                    sampling_uniform(cfg.seed, L + k + 1, kSamplePurposeSpeculativeBonus, 0u);
                tstar_sh = sampling_pick_from_support(cand_idx, prob, n_support, -1, u);
                done_sh  = 1;
            }
        }
        __syncthreads();
        if (done_sh) { break; }
    }

    if (tid == 0) {
        const int a     = a_sh;
        const int tstar = tstar_sh;
        const int L     = L_sh;

        for (int i = 0; i <= k; ++i) { sampled_out[i] = 0; }
        for (int i = 0; i < a; ++i) { sampled_out[i] = drafts[i]; }
        sampled_out[a] = tstar;

        const int produced = a + 1;
        *num_sampled       = produced;
        *accepted          = a;
        *token             = tstar;
        *length            = L + produced;
        stats[0] += k;
        stats[1] += a;
        stats[2] += 1;
        for (int i = 0; i < a; ++i) { stats[kSpeculativeAcceptedPerPositionOffset + i] += 1; }
        if (cfg.token_counts != nullptr) {
            for (int i = 0; i < produced; ++i) { atomicAdd(&cfg.token_counts[sampled_out[i]], 1); }
        }
    }
}

__launch_bounds__(kSamplerBlock) __global__ void speculative_sampling_partial_topk_kernel(
    const __nv_bfloat16* logits, const std::int32_t* drafts, const SamplingConfig* cfg_ptr,
    std::int32_t token_domain, std::int32_t physical_rows, SamplingWorkspace workspace) {
    const int col            = static_cast<int>(blockIdx.y);
    const int partial        = static_cast<int>(blockIdx.x);
    const SamplingConfig cfg = *cfg_ptr;
    if (!(cfg.temperature > 0.0f) || token_domain <= kSamplerTileItems) { return; }
    if (partial == 0 && threadIdx.x == 0) {
        workspace.group_done[col] = 0;
        if (col == 0) { *workspace.speculative_finalize_count = 0; }
    }

    __shared__ typename SamplingPartialSort::TempStorage sort_storage;
    unsigned long long keys[kSamplerItemsPerThread];

    const int cap           = sampling_candidate_cap(cfg, token_domain);
    const std::int64_t base = static_cast<std::int64_t>(col) * physical_rows;
    const int tile_start    = partial * kSamplerPartialTileItems;
    // Column col's penalty overlay is the first `col` drafts (see accept loop);
    // applying it before top-k selection lets it change the candidate set, not
    // just the post-truncation probabilities.
#pragma unroll
    for (int item = 0; item < kSamplerItemsPerThread; ++item) {
        const int v = tile_start + item * blockDim.x + threadIdx.x;
        if (v < token_domain) {
            const float x =
                sampling_adjusted_logit(__bfloat162float(logits[base + v]), v, cfg, drafts, col);
            keys[item] = sampling_sort_key(x, v);
        } else {
            keys[item] = 0ull;
        }
    }
    SamplingPartialSort(sort_storage).Sort(keys, SamplingKeyGreater{});

#pragma unroll
    for (int item = 0; item < kSamplerItemsPerThread; ++item) {
        const int rank = threadIdx.x * kSamplerItemsPerThread + item;
        if (rank < cap) {
            const int off               = sampling_partial_offset(workspace, col, partial, rank);
            workspace.partial_keys[off] = keys[item];
        }
    }
}

__launch_bounds__(kSamplerGroupBlock) __global__ void speculative_sampling_group_finalize_kernel(
    const std::int32_t* target_tokens, const std::int32_t* drafts, std::int32_t* length,
    std::int32_t* token, std::int32_t* sampled_out, std::int32_t* num_sampled,
    std::int32_t* accepted, std::int64_t* stats, const SamplingConfig* cfg_ptr,
    std::int32_t token_domain, std::int32_t cols, std::int32_t partial_blocks,
    std::int32_t group_count, SamplingWorkspace workspace) {
    const int group          = static_cast<int>(blockIdx.x);
    const int col            = static_cast<int>(blockIdx.y);
    const int tid            = threadIdx.x;
    const SamplingConfig cfg = *cfg_ptr;
    if (token_domain <= kSamplerTileItems) { return; }

    if (!(cfg.temperature > 0.0f)) {
        if (tid == 0 && col == 0 && group == 0) {
            const int k = cols - 1;
            int a       = 0;
            while (a < k && target_tokens[a] == drafts[a]) { ++a; }
            const int t_star = target_tokens[a];
            for (int i = 0; i <= k; ++i) { sampled_out[i] = 0; }
            for (int i = 0; i < a; ++i) { sampled_out[i] = drafts[i]; }
            sampled_out[a]     = t_star;
            const int produced = a + 1;
            *num_sampled       = produced;
            *accepted          = a;
            *token             = t_star;
            *length += produced;
            stats[0] += k;
            stats[1] += a;
            stats[2] += 1;
            for (int i = 0; i < a; ++i) { stats[kSpeculativeAcceptedPerPositionOffset + i] += 1; }
        }
        return;
    }

    __shared__ typename SamplingGroupSort::TempStorage sort_storage;
    __shared__ float cand_val[kSamplerCandidateCap];
    __shared__ int cand_idx[kSamplerCandidateCap];
    __shared__ float prob[kSamplerCandidateCap];
    __shared__ int n_support;
    __shared__ int is_last_group;
    unsigned long long keys[kSamplerGroupItemsPerThread];

    const int cap = sampling_candidate_cap(cfg, token_domain);
    // The preceding partial launch initializes all caller-owned counters. CUDA
    // stream ordering makes those writes visible before this launch begins.

    const int group_begin = group * kSamplerPartialsPerGroup;
    int group_partials    = partial_blocks - group_begin;
    if (group_partials < 0) { group_partials = 0; }
    if (group_partials > kSamplerPartialsPerGroup) { group_partials = kSamplerPartialsPerGroup; }
    const int group_n = group_partials * cap;
#pragma unroll
    for (int item = 0; item < kSamplerGroupItemsPerThread; ++item) {
        const int p = item * blockDim.x + tid;
        if (p < group_n) {
            const int partial = group_begin + p / cap;
            const int j       = p - (p / cap) * cap;
            const int off     = sampling_partial_offset(workspace, col, partial, j);
            keys[item]        = workspace.partial_keys[off];
        } else {
            keys[item] = 0ull;
        }
    }
    SamplingGroupSort(sort_storage).Sort(keys, SamplingKeyGreater{});

#pragma unroll
    for (int item = 0; item < kSamplerGroupItemsPerThread; ++item) {
        const int rank = tid * kSamplerGroupItemsPerThread + item;
        if (rank < cap) {
            const int out_off =
                sampling_partial_offset(workspace, col, partial_blocks + group, rank);
            workspace.partial_keys[out_off] = keys[item];
        }
    }
    __syncthreads();

    if (tid == 0) {
        __threadfence();
        const int done = atomicAdd(&workspace.group_done[col], 1) + 1;
        is_last_group  = (done == group_count) ? 1 : 0;
    }
    __syncthreads();
    if (!is_last_group) { return; }

    const int final_n = group_count * cap;
#pragma unroll
    for (int item = 0; item < kSamplerGroupItemsPerThread; ++item) {
        const int p = item * blockDim.x + tid;
        if (p < final_n) {
            const int partial = partial_blocks + p / cap;
            const int j       = p - (p / cap) * cap;
            const int off     = sampling_partial_offset(workspace, col, partial, j);
            keys[item]        = workspace.partial_keys[off];
        } else {
            keys[item] = 0ull;
        }
    }
    SamplingGroupSort(sort_storage).Sort(keys, SamplingKeyGreater{});

#pragma unroll
    for (int item = 0; item < kSamplerGroupItemsPerThread; ++item) {
        const int rank = tid * kSamplerGroupItemsPerThread + item;
        if (rank < cap) {
            cand_val[rank] = sampling_key_float(keys[item]);
            cand_idx[rank] = sampling_key_index(keys[item]);
        }
    }
    __syncthreads();

    sampling_normalize_support(cfg, cand_val, cand_idx, prob, &n_support, cap);

    if (tid == 0) {
        workspace.dist_support[col] = n_support;
        for (int j = 0; j < n_support; ++j) {
            const int off            = sampling_dist_offset(col, j);
            workspace.dist_idx[off]  = cand_idx[j];
            workspace.dist_prob[off] = prob[j];
        }
        workspace.group_done[col] = 0;
        __threadfence();
        const int done_cols = atomicAdd(workspace.speculative_finalize_count, 1) + 1;
        if (done_cols == cols) {
            const int k = cols - 1;
            const int L = *length;
            int a       = 0;
            int tstar   = 0;
            for (int i = 0; i <= k; ++i) {
                const int n            = workspace.dist_support[i];
                const int* dist_idx    = workspace.dist_idx + sampling_dist_offset(i, 0);
                const float* dist_prob = workspace.dist_prob + sampling_dist_offset(i, 0);
                if (i < k) {
                    const int d = drafts[i];
                    float pd    = 0.0f;
                    for (int j = 0; j < n; ++j) {
                        if (dist_idx[j] == d) {
                            pd = dist_prob[j];
                            break;
                        }
                    }
                    const float u =
                        sampling_uniform(cfg.seed, L + i + 1, kSamplePurposeSpeculativeAccept, 0u);
                    if (u < pd) {
                        a = i + 1;
                        continue;
                    }
                    const float ur = sampling_uniform(cfg.seed, L + i + 1,
                                                      kSamplePurposeSpeculativeCorrection, 0u);
                    tstar          = sampling_pick_from_support(dist_idx, dist_prob, n, d, ur);
                    break;
                }
                const float u =
                    sampling_uniform(cfg.seed, L + k + 1, kSamplePurposeSpeculativeBonus, 0u);
                tstar = sampling_pick_from_support(dist_idx, dist_prob, n, -1, u);
            }
            for (int i = 0; i <= k; ++i) { sampled_out[i] = 0; }
            for (int i = 0; i < a; ++i) { sampled_out[i] = drafts[i]; }
            sampled_out[a]     = tstar;
            const int produced = a + 1;
            *num_sampled       = produced;
            *accepted          = a;
            *token             = tstar;
            *length            = L + produced;
            stats[0] += k;
            stats[1] += a;
            stats[2] += 1;
            for (int i = 0; i < a; ++i) { stats[kSpeculativeAcceptedPerPositionOffset + i] += 1; }
            if (cfg.token_counts != nullptr) {
                for (int i = 0; i < produced; ++i) {
                    atomicAdd(&cfg.token_counts[sampled_out[i]], 1);
                }
            }
            *workspace.speculative_finalize_count = 0;
        }
    }
}

__global__ void speculative_select_accepted_hidden_kernel(const __nv_bfloat16* hidden,
                                                          const std::int32_t* accepted,
                                                          __nv_bfloat16* out, std::int32_t rows,
                                                          std::int32_t cols) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) { return; }
    const int col = *accepted;
    if (col < 0 || col >= cols) { return; }
    out[row] = hidden[static_cast<std::int64_t>(col) * rows + row];
}

__global__ void proposal_remap_token_ids_kernel(std::int32_t* proposal_tokens,
                                                std::int32_t proposal_count,
                                                const std::int32_t* id_map, std::int32_t n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= proposal_count) { return; }
    const int idx = proposal_tokens[i];
    if (idx >= 0 && idx < n) { proposal_tokens[i] = id_map[idx]; }
}

} // namespace ninfer::ops
