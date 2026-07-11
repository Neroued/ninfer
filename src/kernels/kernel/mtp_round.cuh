#pragma once

#include "kernels/kernel/sampling_device.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace qus::kernels {

inline constexpr int kMtpRoundAcceptedPerPosOffset = 4;
inline constexpr int kMtpRoundAcceptedPerPosLimit  = 5;

__global__ void mtp_prepare_verify_inputs_kernel(const std::int32_t* token,
                                                 const std::int32_t* drafts,
                                                 const std::int32_t* length,
                                                 std::int32_t* window_base,
                                                 std::int32_t* verify_ids,
                                                 std::int32_t* positions, std::int32_t T) {
    const int i = threadIdx.x;
    if (i == 0) { *window_base = *length; }
    if (i >= T) { return; }
    verify_ids[i] = i == 0 ? *token : drafts[i - 1];
    positions[i]  = *length + i;
}

// Commits the round's accepted tokens plus one correction/bonus token, then
// advances the length/ar_pos scalars and the acceptance stats. The greedy branch
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
__launch_bounds__(kSamplerBlock) __global__ void mtp_accept_tokens_kernel(
    const std::int32_t* target_tokens, const __nv_bfloat16* logits, const std::int32_t* drafts,
    std::int32_t* length, std::int32_t* token, std::int32_t* sampled_out,
    std::int32_t* num_sampled, std::int32_t* accepted, std::int32_t* ar_pos, std::int64_t* stats,
    const SamplingConfig* cfg_ptr, std::int32_t vocab, std::int32_t k) {
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
            *ar_pos = *length;

            stats[0] += k;
            stats[1] += a;
            stats[2] += 1;
            for (int i = 0; i < a && i < kMtpRoundAcceptedPerPosLimit; ++i) {
                stats[kMtpRoundAcceptedPerPosOffset + i] += 1;
            }
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

    const int partial_blocks = div_up(vocab, kSamplerPartialTileItems);
    const int group_count    = sampler_group_count(partial_blocks);
    // No-op when the scratch/group path owns this shape (see mtp_accept_tokens_launch).
    if (sampler_multiblock_ok(vocab, k + 1, partial_blocks, group_count)) { return; }

    if (tid == 0) {
        a_sh    = 0;
        done_sh = 0;
        tstar_sh = 0;
        L_sh    = *length;
    }
    __syncthreads();

    for (int i = 0; i <= k; ++i) {
        // Column i is only reached when drafts[0..i-1] were all accepted, so the
        // round-local penalty overlay for this column is exactly those i drafts.
        if (vocab <= kSamplerTileItems) {
            sampling_build_truncated_small(logits, static_cast<std::int64_t>(i) * vocab, vocab,
                                           cfg, red_val, red_idx, cand_val, cand_idx, prob,
                                           &n_support, drafts, i);
        } else {
            sampling_build_truncated_block_fast(logits, static_cast<std::int64_t>(i) * vocab,
                                                vocab, cfg, merge_val, merge_idx, cand_val,
                                                cand_idx, prob, &n_support, drafts, i);
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
                    sampling_uniform(cfg.seed, L + i + 1, kSamplePurposeMtpAccept, 0u);
                if (u < pd) {
                    a_sh = i + 1;  // accept drafts[i], keep verifying
                } else {
                    const float ur =
                        sampling_uniform(cfg.seed, L + i + 1, kSamplePurposeMtpResample, 0u);
                    tstar_sh = sampling_pick_from_support(cand_idx, prob, n_support, d, ur);
                    done_sh  = 1;
                }
            } else {
                // Every draft accepted: bonus token from the last verify column.
                const float u =
                    sampling_uniform(cfg.seed, L + k + 1, kSamplePurposeMtpBonus, 0u);
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
        *ar_pos            = *length;

        stats[0] += k;
        stats[1] += a;
        stats[2] += 1;
        for (int i = 0; i < a && i < kMtpRoundAcceptedPerPosLimit; ++i) {
            stats[kMtpRoundAcceptedPerPosOffset + i] += 1;
        }
        if (cfg.token_counts != nullptr) {
            for (int i = 0; i < produced; ++i) { atomicAdd(&cfg.token_counts[sampled_out[i]], 1); }
        }
    }
}

__launch_bounds__(kSamplerBlock) __global__ void mtp_sampling_partial_topk_kernel(
    const __nv_bfloat16* logits, const std::int32_t* drafts, const SamplingConfig* cfg_ptr,
    std::int32_t vocab) {
    const int col            = static_cast<int>(blockIdx.y);
    const int partial        = static_cast<int>(blockIdx.x);
    const SamplingConfig cfg = *cfg_ptr;
    const int partial_blocks = static_cast<int>(gridDim.x);
    if (!(cfg.temperature > 0.0f) || vocab <= kSamplerTileItems ||
        col >= kSamplerScratchColumns || partial_blocks > kSamplerScratchPartialBlocks) {
        return;
    }

    __shared__ typename SamplingPartialMergeSort::TempStorage sort_storage;
    unsigned long long keys[kSamplerItemsPerThread];

    const int cap = sampling_candidate_cap(cfg, vocab);
    const std::int64_t base = static_cast<std::int64_t>(col) * vocab;
    const int tile_start = partial * kSamplerPartialTileItems;
    // Column col's penalty overlay is the first `col` drafts (see accept loop);
    // applying it before top-k selection lets it change the candidate set, not
    // just the post-truncation probabilities.
#pragma unroll
    for (int item = 0; item < kSamplerItemsPerThread; ++item) {
        const int v = tile_start + item * blockDim.x + threadIdx.x;
        if (v < vocab) {
            const float x =
                sampling_adjusted_logit(__bfloat162float(logits[base + v]), v, cfg, drafts, col);
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

__launch_bounds__(kSamplerBlock) __global__ void mtp_sampling_group_finalize_kernel(
    const std::int32_t* target_tokens, const std::int32_t* drafts, std::int32_t* length,
    std::int32_t* token, std::int32_t* sampled_out, std::int32_t* num_sampled,
    std::int32_t* accepted, std::int32_t* ar_pos, std::int64_t* stats,
    const SamplingConfig* cfg_ptr, std::int32_t vocab, std::int32_t cols,
    std::int32_t partial_blocks, std::int32_t group_count) {
    const int group          = static_cast<int>(blockIdx.x);
    const int col            = static_cast<int>(blockIdx.y);
    const int tid            = threadIdx.x;
    const SamplingConfig cfg = *cfg_ptr;
    if (vocab <= kSamplerTileItems || cols > kSamplerScratchColumns ||
        partial_blocks > kSamplerScratchPartialBlocks ||
        partial_blocks + group_count > kSamplerScratchPartialBlocks) {
        return;
    }

    if (!(cfg.temperature > 0.0f)) {
        if (tid == 0 && col == 0 && group == 0) {
            const int k = cols - 1;
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
            *ar_pos = *length;
            stats[0] += k;
            stats[1] += a;
            stats[2] += 1;
            for (int i = 0; i < a && i < kMtpRoundAcceptedPerPosLimit; ++i) {
                stats[kMtpRoundAcceptedPerPosOffset + i] += 1;
            }
        }
        return;
    }

    __shared__ typename SamplingGroupMergeSort::TempStorage sort_storage;
    __shared__ float cand_val[kSamplerCandidateCap];
    __shared__ int cand_idx[kSamplerCandidateCap];
    __shared__ float prob[kSamplerCandidateCap];
    __shared__ int n_support;
    __shared__ int is_last_group;
    unsigned long long keys[kSamplerGroupItemsPerThread];

    const int cap = sampling_candidate_cap(cfg, vocab);
    // sampling_group_done[col] is 0 at entry: it is zero-initialized and the
    // terminating group block resets it before returning, with consecutive
    // launches ordered on the same stream. The old atomicCAS re-init raced with
    // the counting atomicAdds and is removed. The sampling_mtp_finalize_* count
    // barrier below still gates the cross-column reduction that runs on the last
    // group of the last column.
    if (tid == 0 && atomicCAS(&sampling_mtp_finalize_init, 0, 2) == 0) {
        sampling_mtp_finalize_count = 0;
        __threadfence();
        atomicExch(&sampling_mtp_finalize_init, 1);
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
        is_last_group = (done == group_count) ? 1 : 0;
    }
    __syncthreads();
    if (!is_last_group) { return; }

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

    sampling_normalize_support(cfg, cand_val, cand_idx, prob, &n_support, cap);

    if (tid == 0) {
        while (atomicAdd(&sampling_mtp_finalize_init, 0) != 1) {}
        sampling_dist_support[col] = n_support;
        for (int j = 0; j < n_support; ++j) {
            const int off = sampling_dist_offset(col, j);
            sampling_dist_idx[off] = cand_idx[j];
            sampling_dist_prob[off] = prob[j];
        }
        sampling_group_done[col] = 0;
        __threadfence();
        const int done_cols = atomicAdd(&sampling_mtp_finalize_count, 1) + 1;
        if (done_cols == cols) {
            const int k = cols - 1;
            const int L = *length;
            int a       = 0;
            int tstar   = 0;
            for (int i = 0; i <= k; ++i) {
                const int n = sampling_dist_support[i];
                const int* dist_idx = sampling_dist_idx + sampling_dist_offset(i, 0);
                const float* dist_prob = sampling_dist_prob + sampling_dist_offset(i, 0);
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
                        sampling_uniform(cfg.seed, L + i + 1, kSamplePurposeMtpAccept, 0u);
                    if (u < pd) {
                        a = i + 1;
                        continue;
                    }
                    const float ur =
                        sampling_uniform(cfg.seed, L + i + 1, kSamplePurposeMtpResample, 0u);
                    tstar = sampling_pick_from_support(dist_idx, dist_prob, n, d, ur);
                    break;
                }
                const float u = sampling_uniform(cfg.seed, L + k + 1, kSamplePurposeMtpBonus, 0u);
                tstar = sampling_pick_from_support(dist_idx, dist_prob, n, -1, u);
            }
            for (int i = 0; i <= k; ++i) { sampled_out[i] = 0; }
            for (int i = 0; i < a; ++i) { sampled_out[i] = drafts[i]; }
            sampled_out[a] = tstar;
            const int produced = a + 1;
            *num_sampled       = produced;
            *accepted          = a;
            *token             = tstar;
            *length            = L + produced;
            *ar_pos            = *length;
            stats[0] += k;
            stats[1] += a;
            stats[2] += 1;
            for (int i = 0; i < a && i < kMtpRoundAcceptedPerPosLimit; ++i) {
                stats[kMtpRoundAcceptedPerPosOffset + i] += 1;
            }
            if (cfg.token_counts != nullptr) {
                for (int i = 0; i < produced; ++i) {
                    atomicAdd(&cfg.token_counts[sampled_out[i]], 1);
                }
            }
            sampling_mtp_finalize_count = 0;
            atomicExch(&sampling_mtp_finalize_init, 0);
        }
    }
}

__global__ void mtp_prepare_shifted_ids_kernel(const std::int32_t* verify_ids,
                                               const std::int32_t* token,
                                               const std::int32_t* accepted,
                                               std::int32_t* shifted_ids, std::int32_t T) {
    const int k = T - 1;
    for (int i = 0; i < k; ++i) { shifted_ids[i] = verify_ids[i + 1]; }
    shifted_ids[*accepted] = *token;
}

__global__ void mtp_gather_hidden_row_kernel(const __nv_bfloat16* hidden,
                                             const std::int32_t* accepted, __nv_bfloat16* out,
                                             std::int32_t rows, std::int32_t cols) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) { return; }
    const int col = *accepted;
    if (col < 0 || col >= cols) { return; }
    out[row] = hidden[static_cast<std::int64_t>(col) * rows + row];
}

__global__ void mtp_remap_draft_token_kernel(std::int32_t* draft_token,
                                             const std::int32_t* id_map, std::int32_t n) {
    const int idx = *draft_token;
    if (idx >= 0 && idx < n) { *draft_token = id_map[idx]; }
}

__global__ void mtp_increment_i32_kernel(std::int32_t* scalar) { *scalar += 1; }

__global__ void mtp_count_fallback_step_kernel(std::int64_t* stats) { stats[3] += 1; }

__global__ void mtp_reset_gdn_initial_slot_kernel(std::int32_t* gdn_initial_slot) {
    *gdn_initial_slot = 0;
}

__global__ void mtp_set_gdn_initial_slot_from_accepted_kernel(const std::int32_t* accepted,
                                                              std::int32_t* gdn_initial_slot) {
    *gdn_initial_slot = *accepted;
}

} // namespace qus::kernels
