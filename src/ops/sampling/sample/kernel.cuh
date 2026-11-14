#pragma once
#include "ops/sampling/kernels/selection.cuh"

#include <type_traits>

#include "ops/sampling/kernels/distribution.cuh"

namespace ninfer::ops::sampling_detail {
using namespace common;

template <class S>
__launch_bounds__(S::kSamplerBlock) __global__
    void sample_row_kernel(const __nv_bfloat16* logits, std::int32_t* out,
                           const SamplingConfig* configs, const std::int32_t* logical_positions,
                           std::int32_t purpose, std::int32_t token_domain,
                           std::int32_t physical_rows, const std::int32_t* live) {
    const int row = static_cast<int>(blockIdx.x);
    if (live && !live[row]) {
        if (threadIdx.x == 0) out[row] = 0;
        return;
    }
    const std::int64_t base  = static_cast<std::int64_t>(row) * physical_rows;
    const int tid            = threadIdx.x;
    const SamplingConfig cfg = configs[row];

    __shared__ float red_val[S::kSamplerBlock];
    __shared__ int red_idx[S::kSamplerBlock];

    // With penalties disabled this remains the exact raw-logit argmax route.
    if (!(cfg.temperature > 0.0f)) {
        float bv             = -CUDART_INF_F;
        int bi               = INT_MAX;
        const bool penalties = cfg.presence_penalty != 0.0f || cfg.frequency_penalty != 0.0f;
        if (!penalties && cfg.mask.words == nullptr) {
            for (int v = tid; v < token_domain; v += blockDim.x) {
                const float x = __bfloat162float(logits[base + v]);
                if (score_id_better(x, v, bv, bi)) {
                    bv = x;
                    bi = v;
                }
            }
        } else {
            for (int v = tid; v < token_domain; v += blockDim.x) {
                const float x = sampling_adjusted_logit(__bfloat162float(logits[base + v]), v, cfg);
                if (score_id_better(x, v, bv, bi)) {
                    bv = x;
                    bi = v;
                }
            }
        }
        red_val[tid] = bv;
        red_idx[tid] = bi;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s &&
                score_id_better(red_val[tid + s], red_idx[tid + s], red_val[tid], red_idx[tid])) {
                red_val[tid] = red_val[tid + s];
                red_idx[tid] = red_idx[tid + s];
            }
            __syncthreads();
        }
        if (tid == 0) {
            if (cfg.mask.words && !isfinite(red_val[0])) { asm volatile("trap;"); }
            out[row] = red_idx[0];
        }
        return;
    }

    __shared__ float cand_val[kSamplerCandidateCap];
    __shared__ int cand_idx[kSamplerCandidateCap];
    __shared__ float prob[kSamplerCandidateCap];
    __shared__ int n_support;
    __shared__ float merge_val[S::kSamplerBlock * kSamplerFastCandidates];
    __shared__ int merge_idx[S::kSamplerBlock * kSamplerFastCandidates];

    if (token_domain <= S::kSamplerTileItems) {
        sampling_build_truncated_small<S>(logits, base, token_domain, cfg, red_val, red_idx,
                                          cand_val, cand_idx, prob, &n_support);
    } else {
        sampling_build_truncated_block_fast(logits, base, token_domain, cfg, merge_val, merge_idx,
                                            cand_val, cand_idx, prob, &n_support);
    }

    if (tid != 0) { return; }
    const int support = n_support;
    const float u     = sampling_uniform(cfg.seed, logical_positions[row], purpose, 0u);
    float acc         = 0.0f;
    int picked        = cand_idx[support - 1];
    for (int j = 0; j < support; ++j) {
        acc += prob[j]; // prob is normalized: goal == u
        if (u < acc) {
            picked = cand_idx[j];
            break;
        }
    }
    out[row] = picked;
}

template <class S, bool WarpTiles = false>
__launch_bounds__(S::kSamplerBlock) __global__
    void sampling_partial_topk_kernel(const __nv_bfloat16* logits, const SamplingConfig* cfg_ptr,
                                      std::int32_t token_domain, std::int32_t physical_rows,
                                      Workspace workspace, const std::int32_t* live) {
    const int col = static_cast<int>(blockIdx.y);
    if (live && !live[col]) return;
    const int partial        = static_cast<int>(blockIdx.x);
    const SamplingConfig cfg = cfg_ptr[col];
    if (partial == 0 && threadIdx.x == 0) { workspace.group_done[col] = 0; }

    using SortStorage = std::conditional_t<WarpTiles, SamplingPartialTopKStorage<S>,
                                           typename SamplingPartialSort<S>::TempStorage>;
    __shared__ SortStorage sort_storage;
    __shared__ unsigned long long greedy_warp_keys[S::kSamplerBlock / 32];
    unsigned long long keys[S::kSamplerItemsPerThread];

    const bool greedy       = !(cfg.temperature > 0.0f);
    const bool penalties    = cfg.presence_penalty != 0.0f || cfg.frequency_penalty != 0.0f;
    const int cap           = greedy ? 1 : sampling_candidate_cap(cfg, token_domain);
    const std::int64_t base = static_cast<std::int64_t>(col) * physical_rows;
    const int tile_start    = partial * S::kSamplerPartialTileItems;
    if constexpr (WarpTiles) {
        if (!greedy && !penalties) {
            unsigned int raw_keys[S::kSamplerItemsPerThread];
#pragma unroll
            for (int item = 0; item < S::kSamplerItemsPerThread; ++item) {
                const int local = item * blockDim.x + threadIdx.x, v = tile_start + local;
                raw_keys[item] = (v < token_domain &&
                                  (!cfg.mask.words || (cfg.mask.words[v / 32] & (1u << (v % 32)))))
                                     ? sampling_bf16_tile_sort_key<S>(logits[base + v], local)
                                     : 0;
            }
            sampling_store_bf16_tile_topk<S>(raw_keys, cap, tile_start, workspace, col, partial,
                                             sort_storage.bf16);
            return;
        }
    }

#pragma unroll
    for (int item = 0; item < S::kSamplerItemsPerThread; ++item) {
        const int v = tile_start + item * blockDim.x + threadIdx.x;
        if (v < token_domain) {
            const float raw = __bfloat162float(logits[base + v]);
            const float x =
                (penalties || cfg.mask.words) ? sampling_adjusted_logit(raw, v, cfg) : raw;
            keys[item] = score_id_order_key(x, v);
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
    if constexpr (WarpTiles) {
        sampling_store_tile_topk<S>(keys, cap, workspace, col, partial, sort_storage.fp32);
    } else {
        SamplingPartialSort<S>(sort_storage).Sort(keys, ScoreIdOrderGreater{});

#pragma unroll
        for (int item = 0; item < S::kSamplerItemsPerThread; ++item) {
            const int rank = threadIdx.x * S::kSamplerItemsPerThread + item;
            if (rank < cap) {
                const int off = sampling_partial_offset(workspace, col, partial, rank);
                workspace.partial_keys[off] = keys[item];
            }
        }
    }
}

template <int Tile>
__launch_bounds__(256) __global__
    void sampling_greedy_partial_kernel(const __nv_bfloat16* logits, const SamplingConfig* configs,
                                        int token_domain, int physical_rows, Workspace workspace,
                                        const int* live) {
    const int row = blockIdx.y, partial = blockIdx.x, tid = threadIdx.x;
    if (live && !live[row]) return;
    const SamplingConfig cfg = configs[row];
    if (cfg.temperature > 0.F) return;
    if (partial == 0 && tid == 0) workspace.group_done[row] = 0;
    const int start = partial * Tile;
    const auto base = std::int64_t(row) * physical_rows;
    if (cfg.presence_penalty == 0.F && cfg.frequency_penalty == 0.F) {
        using TileKey = ThresholdSelection<256, Tile / 256>;
        __shared__ unsigned int warp_maxima[8];
        unsigned int best = 0;
#pragma unroll
        for (int i = 0; i < Tile / 256; ++i) {
            const int local = i * 256 + tid, token = start + local;
            if (token < token_domain &&
                (!cfg.mask.words || (cfg.mask.words[token / 32] & (1U << (token % 32)))))
                best = max(best, sampling_bf16_tile_sort_key<TileKey>(logits[base + token], local));
        }
        best = __reduce_max_sync(0xffffffffU, best);
        if ((tid & 31) == 0) warp_maxima[tid / 32] = best;
        __syncthreads();
        if (tid < 32) {
            best = __reduce_max_sync(0xffffffffU, tid < 8 ? warp_maxima[tid] : 0U);
            if (tid == 0)
                workspace.partial_keys[sampling_partial_offset(workspace, row, partial, 0)] =
                    sampling_expand_bf16_tile_sort_key<TileKey>(best, start);
        }
        return;
    }
    __shared__ unsigned long long warp_keys[8];
    unsigned long long best = 0;
#pragma unroll
    for (int i = 0; i < Tile / 256; ++i) {
        const int token = start + i * 256 + tid;
        if (token < token_domain) {
            const float score =
                sampling_adjusted_logit(__bfloat162float(logits[base + token]), token, cfg);
            best = max(best, static_cast<unsigned long long>(score_id_order_key(score, token)));
        }
    }
    best = block_reduce_max_key(best, warp_keys);
    if (tid == 0)
        workspace.partial_keys[sampling_partial_offset(workspace, row, partial, 0)] = best;
}

template <int Threads, int Items>
__launch_bounds__(Threads) __global__
    void sampling_threshold_topk_kernel(const __nv_bfloat16* logits, const SamplingConfig* configs,
                                        int token_domain, int physical_rows, Workspace workspace,
                                        const int* live) {
    const int row = blockIdx.y, partial = blockIdx.x;
    if (live && !live[row]) return;
    const SamplingConfig cfg = configs[row];
    if (!(cfg.temperature > 0.F)) return;
    if (partial == 0 && threadIdx.x == 0) workspace.group_done[row] = 0;
    using Selector     = ThresholdSelection<Threads, Items>;
    using Compact      = ThresholdSelection<Threads, Items, unsigned int>;
    constexpr int Tile = Threads * Items;

    __shared__ union {
        typename Selector::Storage wide;
        typename Compact::Storage compact;
        SamplingInputTile<Threads, Tile> inputs;
    } storage;

    __shared__ unsigned int compact_keys[kSamplerCandidateCap];
    const int start      = partial * Tile;
    const auto base      = std::int64_t(row) * physical_rows;
    const bool penalties = cfg.presence_penalty != 0.F || cfg.frequency_penalty != 0.F;
    storage.inputs.prefetch(logits + base + start,
                            cfg.token_counts ? cfg.token_counts + start : nullptr,
                            min(Tile, token_domain - start), penalties);
    cp_wait<0>();
    __syncthreads();
    const int cap = sampling_candidate_cap(cfg, token_domain);
    auto* result  = workspace.partial_keys + sampling_partial_offset(workspace, row, partial, 0);
    if (!penalties) {
        unsigned int keys[Items];
#pragma unroll
        for (int i = 0; i < Items; ++i) {
            const int local = threadIdx.x + i * Threads, token = start + local;
            keys[i] =
                token < token_domain &&
                        (!cfg.mask.words || (cfg.mask.words[token / 32] & (1U << (token % 32))))
                    ? sampling_bf16_tile_sort_key<Compact>(storage.inputs.logits[local], local)
                    : 0;
        }
        __syncthreads();
        Compact::select_values(keys, cap, compact_keys, storage.compact);
        __syncthreads();
        if (threadIdx.x < cap)
            result[threadIdx.x] =
                sampling_expand_bf16_tile_sort_key<Compact>(compact_keys[threadIdx.x], start);
    } else {
        unsigned long long keys[Items];
#pragma unroll
        for (int i = 0; i < Items; ++i) {
            const int local = threadIdx.x + i * Threads, token = start + local;
            keys[i] = 0;
            if (token < token_domain) {
                const bool valid =
                    !cfg.mask.words || (cfg.mask.words[token / 32] & (1U << (token % 32)));
                const float score =
                    valid ? sampling_penalized_logit(__bfloat162float(storage.inputs.logits[local]),
                                                     cfg, storage.inputs.counts[local])
                          : -CUDART_INF_F;
                keys[i] = score_id_order_key(score, token);
            }
        }
        __syncthreads();
        Selector::select_values(keys, cap, result, storage.wide);
    }
}

template <class S, bool DirectGreedy = false>
__launch_bounds__(S::kSamplerGroupBlock) __global__
    void sampling_group_finalize_sample_kernel(std::int32_t* out, const SamplingConfig* cfg_ptr,
                                               const std::int32_t* logical_positions,
                                               std::int32_t purpose, std::int32_t token_domain,
                                               std::int32_t partial_blocks,
                                               std::int32_t group_count, Workspace workspace,
                                               const std::int32_t* live) {
    const int group = static_cast<int>(blockIdx.x);
    const int col   = static_cast<int>(blockIdx.y);
    const int tid   = threadIdx.x;
    if (live && !live[col]) {
        if (group == 0 && tid == 0) out[col] = 0;
        return;
    }
    const SamplingConfig cfg = cfg_ptr[col];
    __shared__ typename SamplingGroupSort<S>::TempStorage sort_storage;
    __shared__ float cand_val[kSamplerCandidateCap];
    __shared__ int cand_idx[kSamplerCandidateCap];
    __shared__ float prob[kSamplerCandidateCap];
    __shared__ int n_support;
    __shared__ int is_last;
    __shared__ unsigned long long greedy_warp_keys[S::kSamplerGroupBlock / 32];
    unsigned long long keys[S::kSamplerGroupItemsPerThread];

    const bool greedy = !(cfg.temperature > 0.0f);
    const int cap     = greedy ? 1 : sampling_candidate_cap(cfg, token_domain);
    // The preceding partial launch initializes group_done[col], so caller-owned
    // workspace does not rely on prior contents or a separate memset launch.

    const int group_begin = group * S::kSamplerPartialsPerGroup;
    int group_partials    = partial_blocks - group_begin;
    if (group_partials < 0) { group_partials = 0; }
    if (group_partials > S::kSamplerPartialsPerGroup) {
        group_partials = S::kSamplerPartialsPerGroup;
    }

    if constexpr (DirectGreedy) {
        if (greedy) {
            if (group != 0) return;
            unsigned long long best = 0;
            for (int p = tid; p < partial_blocks; p += blockDim.x)
                best = max(best,
                           workspace.partial_keys[sampling_partial_offset(workspace, col, p, 0)]);
            best = block_reduce_max_key(best, greedy_warp_keys);
            if (tid == 0) {
                if (cfg.mask.words && !isfinite(score_from_order_key(best))) asm volatile("trap;");
                const int picked = id_from_order_key(best);
                out[col]         = picked;
            }
            return;
        }
    }
    if (greedy) {
        unsigned long long best = 0ull;
        for (int p = tid; p < group_partials; p += blockDim.x) {
            const int off = sampling_partial_offset(workspace, col, group_begin + p, 0);
            if (workspace.partial_keys[off] > best) { best = workspace.partial_keys[off]; }
        }
        best = block_reduce_max_key(best, greedy_warp_keys);
        if (tid == 0) {
            const int out_off = sampling_partial_offset(workspace, col, partial_blocks + group, 0);
            workspace.partial_keys[out_off] = best;
            __threadfence();
            const int done = atomicAdd(&workspace.group_done[col], 1) + 1;
            is_last        = (done == group_count) ? 1 : 0;
        }
        __syncthreads();
        if (!is_last) { return; }

        best = 0ull;
        for (int p = tid; p < group_count; p += blockDim.x) {
            const int off = sampling_partial_offset(workspace, col, partial_blocks + p, 0);
            if (workspace.partial_keys[off] > best) { best = workspace.partial_keys[off]; }
        }
        best = block_reduce_max_key(best, greedy_warp_keys);
        if (tid == 0) {
            if (cfg.mask.words && !isfinite(score_from_order_key(best))) { asm volatile("trap;"); }
            const int picked          = id_from_order_key(best);
            out[col]                  = picked;
            workspace.group_done[col] = 0;
        }
        return;
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
    SamplingGroupSort<S>(sort_storage).Sort(keys, ScoreIdOrderGreater{});

#pragma unroll
    for (int item = 0; item < S::kSamplerGroupItemsPerThread; ++item) {
        const int rank = tid * S::kSamplerGroupItemsPerThread + item;
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
        is_last        = (done == group_count) ? 1 : 0;
    }
    __syncthreads();
    if (!is_last) { return; }

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
    SamplingGroupSort<S>(sort_storage).Sort(keys, ScoreIdOrderGreater{});

#pragma unroll
    for (int item = 0; item < S::kSamplerGroupItemsPerThread; ++item) {
        const int rank = tid * S::kSamplerGroupItemsPerThread + item;
        if (rank < cap) {
            cand_val[rank] = score_from_order_key(keys[item]);
            cand_idx[rank] = id_from_order_key(keys[item]);
        }
    }
    __syncthreads();

    sampling_normalize_support(cfg, cand_val, cand_idx, prob, &n_support, cap);
    if (tid == 0) {
        const int support = n_support;
        const float u     = sampling_uniform(cfg.seed, logical_positions[col], purpose, 0u);
        float acc         = 0.0f;
        int picked        = cand_idx[support - 1];
        for (int j = 0; j < support; ++j) {
            acc += prob[j];
            if (u < acc) {
                picked = cand_idx[j];
                break;
            }
        }
        out[col]                  = picked;
        workspace.group_done[col] = 0;
    }
}

} // namespace ninfer::ops::sampling_detail
