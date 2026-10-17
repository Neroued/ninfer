#pragma once

#include "ops/sampling/kernels/distribution.cuh"

#include <cuda_bf16.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::sampling_detail {
using namespace common;

inline constexpr int kSparseSpeculativeCandidates = 16;

template <typename T>
__device__ inline T* speculative_workspace_offset(T* ptr, std::size_t byte_offset) {
    return ptr == nullptr
               ? nullptr
               : reinterpret_cast<T*>(reinterpret_cast<unsigned char*>(ptr) + byte_offset);
}

__device__ inline Workspace speculative_workspace_row(Workspace workspace, std::size_t row_stride,
                                                      std::int32_t row) {
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

__device__ __forceinline__ void
speculative_sparse_warp_accept(Workspace workspace, const int* drafts, const int* candidate_ids,
                               const float* proposal_q, const SamplingConfig& cfg, int k, int row,
                               int extent, bool greedy, int* lengths, int* anchors,
                               int* licensed_tokens, int* licensed_counts, int* accepted) {
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

template <class S>
__launch_bounds__(S::kSamplerBlock) __global__ void speculative_sampling_partial_topk_kernel(
    const __nv_bfloat16* logits, const std::int32_t* drafts, const std::int32_t* current_extents,
    const SamplingConfig* configs, std::int32_t token_domain, std::int32_t physical_rows,
    std::int32_t cols, std::int32_t k, Workspace workspace, std::size_t workspace_row_stride) {
    const int row     = static_cast<int>(blockIdx.z);
    const int col     = static_cast<int>(blockIdx.y);
    const int partial = static_cast<int>(blockIdx.x);
    int extent        = current_extents[row];
    extent            = extent < 0 ? 0 : (extent > k ? k : extent);
    if (col > extent) { return; }
    const SamplingConfig cfg = configs[row];
    const bool greedy        = !(cfg.temperature > 0.0f);
    const bool penalties     = cfg.presence_penalty != 0.0f || cfg.frequency_penalty != 0.0f;
    if (greedy && !penalties && cfg.mask.words == nullptr) { return; }
    workspace = speculative_workspace_row(workspace, workspace_row_stride, row);
    if (partial == 0 && threadIdx.x == 0) {
        workspace.group_done[col] = 0;
        if (col == 0) { *workspace.speculative_finalize_count = 0; }
    }

    __shared__ SamplingPartialTopKStorage<S> topk_storage;
    __shared__ unsigned long long greedy_warp_keys[S::kSamplerBlock / 32];

    const int cap                  = greedy ? 1 : sampling_candidate_cap(cfg, token_domain);
    const std::int64_t base        = (static_cast<std::int64_t>(row) * cols + col) * physical_rows;
    const std::int32_t* row_drafts = drafts + row * k;
    const int tile_start           = partial * S::kSamplerPartialTileItems;
    if (!penalties) {
        unsigned int keys[S::kSamplerItemsPerThread];
#pragma unroll
        for (int item = 0; item < S::kSamplerItemsPerThread; ++item) {
            const int tile_index = item * blockDim.x + threadIdx.x;
            const int v          = tile_start + tile_index;
            keys[item]           = (v < token_domain &&
                          (!cfg.mask.words ||
                           (cfg.mask.words[col * cfg.mask.stride + v / 32] & (1u << (v % 32)))))
                                       ? sampling_bf16_tile_sort_key<S>(logits[base + v], tile_index)
                                       : 0u;
        }
        sampling_store_bf16_tile_topk<S>(keys, cap, tile_start, workspace, col, partial,
                                         topk_storage.bf16);
        return;
    }

    unsigned long long keys[S::kSamplerItemsPerThread];
    // Column col's penalty overlay is the first `col` drafts (see accept loop);
    // applying it before top-k selection lets it change the candidate set, not
    // just the post-truncation probabilities.
#pragma unroll
    for (int item = 0; item < S::kSamplerItemsPerThread; ++item) {
        const int v = tile_start + item * blockDim.x + threadIdx.x;
        if (v < token_domain) {
            const __nv_bfloat16 raw = logits[base + v];
            keys[item]              = score_id_order_key(
                sampling_adjusted_logit(__bfloat162float(raw), v, cfg, row_drafts, col), v);
        } else {
            keys[item] = 0ull;
        }
    }
    if (greedy) {
        unsigned long long best = keys[0];
#pragma unroll
        for (int item = 1; item < S::kSamplerItemsPerThread; ++item) {
            if (keys[item] > best) { best = keys[item]; }
        }
        best = block_reduce_max_key(best, greedy_warp_keys);
        if (threadIdx.x == 0) {
            const int off               = sampling_partial_offset(workspace, col, partial, 0);
            workspace.partial_keys[off] = best;
        }
        return;
    }
    sampling_store_tile_topk<S>(keys, cap, workspace, col, partial, topk_storage.fp32);
}

template <class S, bool SparseProposal>
__launch_bounds__(S::kSamplerGroupBlock) __global__ void speculative_sampling_group_finalize_kernel(
    const std::int32_t* target_tokens, const std::int32_t* drafts,
    const std::int32_t* candidate_ids, const float* proposal_q, const std::int32_t* current_extents,
    std::int32_t* lengths, std::int32_t* anchors, std::int32_t* licensed_tokens,
    std::int32_t* licensed_counts, std::int32_t* accepted, const SamplingConfig* configs,
    std::int32_t token_domain, std::int32_t cols, std::int32_t partial_blocks,
    std::int32_t group_count, Workspace workspace, std::size_t workspace_row_stride) {
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
    const bool greedy               = !(cfg.temperature > 0.0f);
    const bool penalties            = cfg.presence_penalty != 0.0f || cfg.frequency_penalty != 0.0f;

    if (greedy && !penalties && cfg.mask.words == nullptr) {
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

    __shared__ SamplingTileTopKStorage<S> topk_storage;
    __shared__ float cand_val[kSamplerCandidateCap];
    __shared__ int cand_idx[kSamplerCandidateCap];
    __shared__ float prob[kSamplerCandidateCap];
    __shared__ int n_support;
    __shared__ int is_last_group;
    unsigned long long keys[S::kSamplerGroupItemsPerThread];

    const int cap = greedy ? 1 : sampling_candidate_cap(cfg, token_domain);
    // The preceding partial launch initializes all caller-owned counters. CUDA
    // stream ordering makes those writes visible before this launch begins.

    const int group_begin = group * S::kSamplerPartialsPerGroup;
    int group_partials    = partial_blocks - group_begin;
    if (group_partials < 0) { group_partials = 0; }
    if (group_partials > S::kSamplerPartialsPerGroup) {
        group_partials = S::kSamplerPartialsPerGroup;
    }
    const int group_n = group_partials * cap;
#pragma unroll
    for (int item = 0; item < S::kSamplerGroupItemsPerThread; ++item) {
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
    sampling_store_tile_topk<S>(keys, cap, workspace, col, partial_blocks + group, topk_storage);
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
    for (int item = 0; item < S::kSamplerGroupItemsPerThread; ++item) {
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
    sampling_store_tile_topk<S>(keys, cap, workspace, col, partial_blocks + group, topk_storage);
    __syncthreads();

    if (tid < cap) {
        const int off = sampling_partial_offset(workspace, col, partial_blocks + group, tid);
        const unsigned long long key = workspace.partial_keys[off];
        cand_val[tid]                = score_from_order_key(key);
        cand_idx[tid]                = id_from_order_key(key);
    }
    __syncthreads();

    if (tid == 0 && greedy && cfg.mask.words && !isfinite(cand_val[0])) { asm volatile("trap;"); }

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

} // namespace ninfer::ops::sampling_detail
