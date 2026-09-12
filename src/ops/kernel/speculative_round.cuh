#pragma once

// Implements: include/ninfer/ops/speculative_round.h
// Match: contiguous request-major state and BF16 verification logits.
// Algorithm assumptions: small vocabularies use one cooperative block; the
// registered full-vocabulary stochastic route uses the sampling partial/group
// pipeline and caller-owned workspace. Sparse acceptance and emission use one warp per request.

#include "ops/kernel/sampling_device.cuh"

#include <cuda_bf16.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

inline constexpr int kSparseSpeculativeCandidates = 16;

__global__ void speculative_prepare_verify_inputs_kernel(const std::int32_t* anchors,
                                                         const std::int32_t* drafts,
                                                         const std::int32_t* base_positions,
                                                         const std::int32_t* current_extents,
                                                         std::int32_t* verify_ids,
                                                         std::int32_t* positions, std::int32_t k) {
    const int row = static_cast<int>(blockIdx.y);
    const int T   = k + 1;
    int extent    = current_extents[row];
    extent        = extent < 0 ? 0 : (extent > k ? k : extent);
    for (int j = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x; j < T;
         j += blockDim.x * gridDim.x) {
        const int off = row * T + j;
        verify_ids[off] =
            j == 0 ? anchors[row] : (j <= extent ? drafts[row * k + j - 1] : anchors[row]);
        if (positions != nullptr) {
            positions[off] = base_positions[row] + (j <= extent ? j : extent);
        }
    }
}

template <typename T>
__device__ inline T* speculative_workspace_offset(T* ptr, std::size_t byte_offset) {
    return ptr == nullptr
               ? nullptr
               : reinterpret_cast<T*>(reinterpret_cast<unsigned char*>(ptr) + byte_offset);
}

__device__ inline SamplingWorkspace
speculative_workspace_row(SamplingWorkspace workspace, std::size_t row_stride, std::int32_t row) {
    const std::size_t offset = static_cast<std::size_t>(row) * row_stride;
    workspace.partial_keys   = speculative_workspace_offset(workspace.partial_keys, offset);
    workspace.dist_idx       = speculative_workspace_offset(workspace.dist_idx, offset);
    workspace.dist_prob      = speculative_workspace_offset(workspace.dist_prob, offset);
    workspace.dist_support   = speculative_workspace_offset(workspace.dist_support, offset);
    workspace.group_done     = speculative_workspace_offset(workspace.group_done, offset);
    workspace.speculative_finalize_count =
        speculative_workspace_offset(workspace.speculative_finalize_count, offset);
    return workspace;
}

template <bool UpdateTokenCounts>
__device__ __forceinline__ void
speculative_store_accept_result(const std::int32_t* row_drafts, std::int32_t k, std::int32_t row,
                                std::int32_t accepted_count, std::int32_t terminal_token,
                                std::int32_t* lengths, std::int32_t* anchors,
                                std::int32_t* row_tokens, std::int32_t* licensed_counts,
                                std::int32_t* accepted, const SamplingConfig* config) {
    for (int i = 0; i <= k; ++i) { row_tokens[i] = 0; }
    for (int i = 0; i < accepted_count; ++i) { row_tokens[i] = row_drafts[i]; }
    row_tokens[accepted_count] = terminal_token;

    const int produced   = accepted_count + 1;
    licensed_counts[row] = produced;
    accepted[row]        = accepted_count;
    anchors[row]         = terminal_token;
    lengths[row] += produced;
    if constexpr (UpdateTokenCounts) {
        if (config->token_counts != nullptr) {
            for (int i = 0; i < produced; ++i) {
                atomicAdd(&config->token_counts[row_tokens[i]], 1);
            }
        }
    }
}

__device__ __forceinline__ float speculative_sparse_probability(const std::int32_t* candidate_ids,
                                                                const float* proposal_q,
                                                                std::int32_t token) {
    float probability = 0.0f;
#pragma unroll
    for (int candidate = 0; candidate < kSparseSpeculativeCandidates; ++candidate) {
        if (candidate_ids[candidate] == token) {
            probability = proposal_q[candidate];
            break;
        }
    }
    return probability;
}

// One warp owns a request. Each draft's acceptance event is independent given the provided
// path and its conditional p/q distributions; the first failed event determines the prefix.
__device__ __forceinline__ void speculative_sparse_warp_store(const int* drafts, int k, int row,
                                                              int accepted_count, int terminal,
                                                              int* lengths, int* anchors,
                                                              int* licensed_tokens,
                                                              int* licensed_counts, int* accepted) {
    const int lane = threadIdx.x & 31;
    if (lane <= k)
        licensed_tokens[row * (k + 1) + lane] = lane < accepted_count    ? drafts[row * k + lane]
                                                : lane == accepted_count ? terminal
                                                                         : 0;
    if (lane == 0) {
        licensed_counts[row] = accepted_count + 1;
        accepted[row]        = accepted_count;
        anchors[row]         = terminal;
        lengths[row] += accepted_count + 1;
    }
}

__device__ __forceinline__ void speculative_sparse_warp_greedy(const int* target_tokens,
                                                               const int* drafts, int* lengths,
                                                               int* anchors, int* licensed_tokens,
                                                               int* licensed_counts, int* accepted,
                                                               int row, int extent, int k) {
    const int lane = threadIdx.x & 31;
    const bool reject =
        lane < extent && target_tokens[row * (k + 1) + lane] != drafts[row * k + lane];
    const unsigned mask = __ballot_sync(0xffffffffU, reject);
    const int a         = mask ? __ffs(mask) - 1 : extent;
    const int terminal  = target_tokens[row * (k + 1) + a];
    speculative_sparse_warp_store(drafts, k, row, a, terminal, lengths, anchors, licensed_tokens,
                                  licensed_counts, accepted);
}

__global__ __launch_bounds__(256) void speculative_accept_sparse_warp_greedy_kernel(
    const int* target_tokens, const int* drafts, const int* current_extents, int* lengths,
    int* anchors, int* licensed_tokens, int* licensed_counts, int* accepted, int k) {
    const int row    = threadIdx.x / 32;
    const int extent = min(k, max(0, current_extents[row]));
    speculative_sparse_warp_greedy(target_tokens, drafts, lengths, anchors, licensed_tokens,
                                   licensed_counts, accepted, row, extent, k);
}

__device__ __forceinline__ void speculative_sparse_warp_accept(
    SamplingWorkspace workspace, const int* drafts, const int* candidate_ids,
    const float* proposal_q, const SamplingConfig& cfg, int k, int row, int extent, bool greedy,
    int* lengths, int* anchors, int* licensed_tokens, int* licensed_counts, int* accepted) {
    const int lane       = threadIdx.x & 31;
    const int old_length = lengths[row];
    bool reject          = false;
    if (lane < extent) {
        const int d = drafts[row * k + lane];
        if (greedy)
            reject = workspace.dist_idx[sampling_dist_offset(lane, 0)] != d;
        else {
            const int support = workspace.dist_support[lane];
            float pd          = 0.0f;
            for (int j = 0; j < support; ++j) {
                const int at = sampling_dist_offset(lane, j);
                if (workspace.dist_idx[at] == d) {
                    pd = workspace.dist_prob[at];
                    break;
                }
            }
            const int at   = (row * k + lane) * kSparseSpeculativeCandidates;
            const float qd = speculative_sparse_probability(candidate_ids + at, proposal_q + at, d);
            const float u  = sampling_uniform(cfg.seed, old_length + lane + 1,
                                              kSamplePurposeSpeculativeAccept, 0);
            reject         = !(pd >= qd || u * qd < pd);
        }
    }
    const unsigned failures = __ballot_sync(0xffffffffU, reject);
    const int a             = failures ? __ffs(failures) - 1 : extent;
    int terminal;
    if (greedy)
        terminal = workspace.dist_idx[sampling_dist_offset(a, 0)];
    else {
        const int n  = workspace.dist_support[a];
        int token    = 0;
        float weight = 0.0f;
        if (lane < n) {
            const int at = sampling_dist_offset(a, lane);
            token        = workspace.dist_idx[at];
            weight       = workspace.dist_prob[at];
            if (a < extent) {
                const int q_at = (row * k + a) * kSparseSpeculativeCandidates;
                weight         = fmaxf(weight - speculative_sparse_probability(candidate_ids + q_at,
                                                                               proposal_q + q_at, token),
                                       0.0f);
            }
        }
        float cdf = weight;
#pragma unroll
        for (int offset = 1; offset < 32; offset *= 2) {
            const float earlier = __shfl_up_sync(0xffffffffU, cdf, offset);
            if (lane >= offset) cdf += earlier;
        }
        const float mass = __shfl_sync(0xffffffffU, cdf, 31);
        const int purpose =
            a < extent ? kSamplePurposeSpeculativeCorrection : kSamplePurposeSpeculativeBonus;
        const float u           = sampling_uniform(cfg.seed, old_length + a + 1, purpose, 0);
        const float goal        = u * mass;
        const unsigned positive = __ballot_sync(0xffffffffU, lane < n && weight > 0.0f);
        const unsigned candidates =
            __ballot_sync(0xffffffffU, lane < n && weight > 0.0f && goal < cdf);
        const int selected = !(mass > 0.0f) ? 0
                             : candidates   ? __ffs(candidates) - 1
                             : positive     ? 31 - __clz(positive)
                                            : 0;
        terminal           = __shfl_sync(0xffffffffU, token, selected);
    }
    speculative_sparse_warp_store(drafts, k, row, a, terminal, lengths, anchors, licensed_tokens,
                                  licensed_counts, accepted);
}

// Commits the round's accepted tokens plus one correction/bonus token, then
// advances the target length. The greedy branch
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
    const std::int32_t* current_extents, std::int32_t* lengths, std::int32_t* anchors,
    std::int32_t* licensed_tokens, std::int32_t* licensed_counts, std::int32_t* accepted,
    const SamplingConfig* configs, std::int32_t token_domain, std::int32_t physical_rows,
    std::int32_t k) {
    const int tid                   = threadIdx.x;
    const int row                   = static_cast<int>(blockIdx.x);
    const int cols                  = k + 1;
    int extent                      = current_extents[row];
    extent                          = extent < 0 ? 0 : (extent > k ? k : extent);
    const SamplingConfig cfg        = configs[row];
    const std::int32_t* row_targets = target_tokens + row * cols;
    const std::int32_t* row_drafts  = drafts + row * k;
    std::int32_t* row_tokens        = licensed_tokens + row * cols;
    const __nv_bfloat16* row_logits =
        logits + static_cast<std::int64_t>(row) * cols * physical_rows;
    const bool penalties = cfg.presence_penalty != 0.0f || cfg.frequency_penalty != 0.0f;

    if (!(cfg.temperature > 0.0f) && !penalties) {
        if (tid == 0) {
            int a = 0;
            while (a < extent && row_targets[a] == row_drafts[a]) { ++a; }
            const int t_star = row_targets[a];
            speculative_store_accept_result<true>(row_drafts, k, row, a, t_star, lengths, anchors,
                                                  row_tokens, licensed_counts, accepted, &cfg);
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
    if (sampler_multiblock_ok(token_domain, cols, partial_blocks, group_count)) { return; }

    if (tid == 0) {
        a_sh     = 0;
        done_sh  = 0;
        tstar_sh = 0;
        L_sh     = lengths[row];
    }
    __syncthreads();

    if (!(cfg.temperature > 0.0f)) {
        for (int i = 0; i <= extent; ++i) {
            const std::int64_t base = static_cast<std::int64_t>(i) * physical_rows;
            float best_value        = -CUDART_INF_F;
            int best_index          = INT_MAX;
            for (int v = tid; v < token_domain; v += blockDim.x) {
                const float value = sampling_adjusted_logit(__bfloat162float(row_logits[base + v]),
                                                            v, cfg, row_drafts, i);
                if (sampling_better(value, v, best_value, best_index)) {
                    best_value = value;
                    best_index = v;
                }
            }
            red_val[tid] = best_value;
            red_idx[tid] = best_index;
            __syncthreads();
            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (tid < stride && sampling_better(red_val[tid + stride], red_idx[tid + stride],
                                                    red_val[tid], red_idx[tid])) {
                    red_val[tid] = red_val[tid + stride];
                    red_idx[tid] = red_idx[tid + stride];
                }
                __syncthreads();
            }
            if (tid == 0) {
                const int selected = red_idx[0];
                if (i < extent && selected == row_drafts[i]) {
                    a_sh = i + 1;
                } else {
                    tstar_sh = selected;
                    done_sh  = 1;
                }
            }
            __syncthreads();
            if (done_sh) { break; }
        }

        if (tid == 0) {
            const int a     = a_sh;
            const int tstar = tstar_sh;
            speculative_store_accept_result<true>(row_drafts, k, row, a, tstar, lengths, anchors,
                                                  row_tokens, licensed_counts, accepted, &cfg);
        }
        return;
    }

    for (int i = 0; i <= extent; ++i) {
        // Column i is only reached when drafts[0..i-1] were all accepted, so the
        // round-local penalty overlay for this column is exactly those i drafts.
        const std::int64_t base = static_cast<std::int64_t>(i) * physical_rows;
        if (token_domain <= kSamplerTileItems) {
            sampling_build_truncated_small(row_logits, base, token_domain, cfg, red_val, red_idx,
                                           cand_val, cand_idx, prob, &n_support, row_drafts, i);
        } else {
            sampling_build_truncated_block_fast(row_logits, base, token_domain, cfg, merge_val,
                                                merge_idx, cand_val, cand_idx, prob, &n_support,
                                                row_drafts, i);
        }
        if (tid == 0 && done_sh == 0) {
            const int L = L_sh;
            if (i < extent) {
                const int d = row_drafts[i];
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
                    sampling_uniform(cfg.seed, L + extent + 1, kSamplePurposeSpeculativeBonus, 0u);
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
        speculative_store_accept_result<true>(row_drafts, k, row, a, tstar, lengths, anchors,
                                              row_tokens, licensed_counts, accepted, &cfg);
    }
}

__launch_bounds__(kSamplerBlock) __global__ void speculative_sampling_partial_topk_kernel(
    const __nv_bfloat16* logits, const std::int32_t* drafts, const std::int32_t* current_extents,
    const SamplingConfig* configs, std::int32_t token_domain, std::int32_t physical_rows,
    std::int32_t cols, std::int32_t k, SamplingWorkspace workspace,
    std::size_t workspace_row_stride) {
    const int row     = static_cast<int>(blockIdx.z);
    const int col     = static_cast<int>(blockIdx.y);
    const int partial = static_cast<int>(blockIdx.x);
    int extent        = current_extents[row];
    extent            = extent < 0 ? 0 : (extent > k ? k : extent);
    if (col > extent) { return; }
    const SamplingConfig cfg = configs[row];
    const bool greedy        = !(cfg.temperature > 0.0f);
    const bool penalties     = cfg.presence_penalty != 0.0f || cfg.frequency_penalty != 0.0f;
    if ((greedy && !penalties) || token_domain <= kSamplerTileItems) { return; }
    workspace = speculative_workspace_row(workspace, workspace_row_stride, row);
    if (partial == 0 && threadIdx.x == 0) {
        workspace.group_done[col] = 0;
        if (col == 0) { *workspace.speculative_finalize_count = 0; }
    }

    __shared__ SamplingPartialTopKStorage topk_storage;
    __shared__ unsigned long long greedy_warp_keys[kSamplerBlock / 32];

    const int cap                  = greedy ? 1 : sampling_candidate_cap(cfg, token_domain);
    const std::int64_t base        = (static_cast<std::int64_t>(row) * cols + col) * physical_rows;
    const std::int32_t* row_drafts = drafts + row * k;
    const int tile_start           = partial * kSamplerPartialTileItems;
    if (!penalties) {
        unsigned int keys[kSamplerItemsPerThread];
#pragma unroll
        for (int item = 0; item < kSamplerItemsPerThread; ++item) {
            const int tile_index = item * blockDim.x + threadIdx.x;
            const int v          = tile_start + tile_index;
            keys[item] =
                v < token_domain ? sampling_bf16_tile_sort_key(logits[base + v], tile_index) : 0u;
        }
        sampling_store_bf16_tile_topk(keys, cap, tile_start, workspace, col, partial,
                                      topk_storage.bf16);
        return;
    }

    unsigned long long keys[kSamplerItemsPerThread];
    // Column col's penalty overlay is the first `col` drafts (see accept loop);
    // applying it before top-k selection lets it change the candidate set, not
    // just the post-truncation probabilities.
#pragma unroll
    for (int item = 0; item < kSamplerItemsPerThread; ++item) {
        const int v = tile_start + item * blockDim.x + threadIdx.x;
        if (v < token_domain) {
            const __nv_bfloat16 raw = logits[base + v];
            keys[item]              = sampling_sort_key(
                sampling_adjusted_logit(__bfloat162float(raw), v, cfg, row_drafts, col), v);
        } else {
            keys[item] = 0ull;
        }
    }
    if (greedy) {
        unsigned long long best = keys[0];
#pragma unroll
        for (int item = 1; item < kSamplerItemsPerThread; ++item) {
            if (keys[item] > best) { best = keys[item]; }
        }
        best = sampling_block_max_key(best, greedy_warp_keys);
        if (threadIdx.x == 0) {
            const int off               = sampling_partial_offset(workspace, col, partial, 0);
            workspace.partial_keys[off] = best;
        }
        return;
    }
    sampling_store_tile_topk(keys, cap, workspace, col, partial, topk_storage.fp32);
}

template <bool SparseProposal>
__launch_bounds__(kSamplerGroupBlock) __global__ void speculative_sampling_group_finalize_kernel(
    const std::int32_t* target_tokens, const std::int32_t* drafts,
    const std::int32_t* candidate_ids, const float* proposal_q, const std::int32_t* current_extents,
    std::int32_t* lengths, std::int32_t* anchors, std::int32_t* licensed_tokens,
    std::int32_t* licensed_counts, std::int32_t* accepted, const SamplingConfig* configs,
    std::int32_t token_domain, std::int32_t cols, std::int32_t partial_blocks,
    std::int32_t group_count, SamplingWorkspace workspace, std::size_t workspace_row_stride) {
    const int row   = static_cast<int>(blockIdx.z);
    const int group = static_cast<int>(blockIdx.x);
    const int col   = static_cast<int>(blockIdx.y);
    const int tid   = threadIdx.x;
    const int k     = cols - 1;
    int extent      = current_extents[row];
    extent          = extent < 0 ? 0 : (extent > k ? k : extent);
    if (col > extent) { return; }
    const SamplingConfig cfg        = configs[row];
    const std::int32_t* row_targets = target_tokens + row * cols;
    const std::int32_t* row_drafts  = drafts + row * k;
    std::int32_t* row_tokens        = licensed_tokens + row * cols;
    if (token_domain <= kSamplerTileItems) { return; }
    const bool greedy    = !(cfg.temperature > 0.0f);
    const bool penalties = cfg.presence_penalty != 0.0f || cfg.frequency_penalty != 0.0f;

    if (greedy && !penalties) {
        if constexpr (SparseProposal) {
            if (tid < 32 && col == 0 && group == 0)
                speculative_sparse_warp_greedy(target_tokens, drafts, lengths, anchors,
                                               licensed_tokens, licensed_counts, accepted, row,
                                               extent, k);
        } else {

            if (tid == 0 && col == 0 && group == 0) {
                int a = 0;
                while (a < extent && row_targets[a] == row_drafts[a]) { ++a; }
                const int t_star = row_targets[a];
                speculative_store_accept_result<!SparseProposal>(row_drafts, k, row, a, t_star,
                                                                 lengths, anchors, row_tokens,
                                                                 licensed_counts, accepted, &cfg);
            }
        }
        return;
    }


    workspace = speculative_workspace_row(workspace, workspace_row_stride, row);

    __shared__ SamplingTileTopKStorage topk_storage;
    __shared__ float cand_val[kSamplerCandidateCap];
    __shared__ int cand_idx[kSamplerCandidateCap];
    __shared__ float prob[kSamplerCandidateCap];
    __shared__ int n_support;
    __shared__ int is_last_group;
    unsigned long long keys[kSamplerGroupItemsPerThread];

    const int cap = greedy ? 1 : sampling_candidate_cap(cfg, token_domain);
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
    sampling_store_tile_topk(keys, cap, workspace, col, partial_blocks + group, topk_storage);
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
    sampling_store_tile_topk(keys, cap, workspace, col, partial_blocks + group, topk_storage);
    __syncthreads();

    if (tid < cap) {
        const int off = sampling_partial_offset(workspace, col, partial_blocks + group, tid);
        const unsigned long long key = workspace.partial_keys[off];
        cand_val[tid]                = sampling_key_float(key);
        cand_idx[tid]                = sampling_key_index(key);
    }
    __syncthreads();

    if constexpr (SparseProposal) {
        __shared__ int last_column;
        if (greedy) {
            if (tid == 0) {
                workspace.dist_idx[sampling_dist_offset(col, 0)] = cand_idx[0];
                workspace.group_done[col]                        = 0;
                __threadfence();
                last_column = atomicAdd(workspace.speculative_finalize_count, 1) + 1 == extent + 1;
            }
            __syncthreads();
            if (last_column && tid < 32)
                speculative_sparse_warp_accept(workspace, drafts, candidate_ids, proposal_q, cfg, k,
                                               row, extent, true, lengths, anchors, licensed_tokens,
                                               licensed_counts, accepted);
            if (last_column && tid == 0) *workspace.speculative_finalize_count = 0;
            return;
        }
        sampling_normalize_support(cfg, cand_val, cand_idx, prob, &n_support, cap);
        if (tid == 0) {
            workspace.dist_support[col] = n_support;
            for (int j = 0; j < n_support; ++j) {
                const int at            = sampling_dist_offset(col, j);
                workspace.dist_idx[at]  = cand_idx[j];
                workspace.dist_prob[at] = prob[j];
            }
            workspace.group_done[col] = 0;
            __threadfence();
            last_column = atomicAdd(workspace.speculative_finalize_count, 1) + 1 == extent + 1;
        }
        __syncthreads();
        if (last_column && tid < 32)
            speculative_sparse_warp_accept(workspace, drafts, candidate_ids, proposal_q, cfg, k,
                                           row, extent, false, lengths, anchors, licensed_tokens,
                                           licensed_counts, accepted);
        if (last_column && tid == 0) *workspace.speculative_finalize_count = 0;
        return;
    } else {

        if (greedy) {
            if (tid == 0) {
                workspace.dist_idx[sampling_dist_offset(col, 0)] = cand_idx[0];
                workspace.group_done[col]                        = 0;
                __threadfence();
                const int done_cols = atomicAdd(workspace.speculative_finalize_count, 1) + 1;
                if (done_cols == extent + 1) {
                    int a     = 0;
                    int tstar = 0;
                    for (int i = 0; i <= extent; ++i) {
                        const int selected = workspace.dist_idx[sampling_dist_offset(i, 0)];
                        if (i < extent && selected == row_drafts[i]) {
                            a = i + 1;
                            continue;
                        }
                        tstar = selected;
                        break;
                    }
                    speculative_store_accept_result<!SparseProposal>(
                        row_drafts, k, row, a, tstar, lengths, anchors, row_tokens, licensed_counts,
                        accepted, &cfg);
                    *workspace.speculative_finalize_count = 0;
                }
            }
            return;
        }

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
            if (done_cols == extent + 1) {
                const int L = lengths[row];
                int a       = 0;
                int tstar   = 0;
                for (int i = 0; i <= extent; ++i) {
                    const int n            = workspace.dist_support[i];
                    const int* dist_idx    = workspace.dist_idx + sampling_dist_offset(i, 0);
                    const float* dist_prob = workspace.dist_prob + sampling_dist_offset(i, 0);
                    if (i < extent) {
                        const int d = row_drafts[i];
                        float pd    = 0.0f;
                        for (int j = 0; j < n; ++j) {
                            if (dist_idx[j] == d) {
                                pd = dist_prob[j];
                                break;
                            }
                        }
                        const float u           = sampling_uniform(cfg.seed, L + i + 1,
                                                                   kSamplePurposeSpeculativeAccept, 0u);
                        const bool accept_draft = u < pd;
                        if (accept_draft) {
                            a = i + 1;
                            continue;
                        }
                        const float ur = sampling_uniform(cfg.seed, L + i + 1,
                                                          kSamplePurposeSpeculativeCorrection, 0u);

                        tstar = sampling_pick_from_support(dist_idx, dist_prob, n, d, ur);

                        break;
                    }
                    const float u = sampling_uniform(cfg.seed, L + extent + 1,
                                                     kSamplePurposeSpeculativeBonus, 0u);
                    tstar         = sampling_pick_from_support(dist_idx, dist_prob, n, -1, u);
                }
                speculative_store_accept_result<!SparseProposal>(row_drafts, k, row, a, tstar,
                                                                 lengths, anchors, row_tokens,
                                                                 licensed_counts, accepted, &cfg);
                *workspace.speculative_finalize_count = 0;
            }
        }
    }
}

__global__ void speculative_select_accepted_hidden_kernel(const __nv_bfloat16* hidden,
                                                          const std::int32_t* selectors,
                                                          __nv_bfloat16* out, std::int32_t rows,
                                                          std::int32_t cols) {
    const int batch = static_cast<int>(blockIdx.y);
    const int row   = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) { return; }
    const int col = selectors[batch];
    if (col < 0 || col >= cols) { return; }
    out[static_cast<std::int64_t>(batch) * rows + row] =
        hidden[(static_cast<std::int64_t>(batch) * cols + col) * rows + row];
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
