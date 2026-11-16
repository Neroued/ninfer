#include "ops/sampling/acceptance/kernel.cuh"
#include "ops/sampling/kernels/selection.cuh"
#include "core/platform.h"

namespace ninfer::ops::sampling_detail {
namespace {
#if NINFER_TARGET_SPARK
using Schedule = TopKSchedule<128, 4, 128, 4, 25>;
#else
using Schedule = TopKSchedule<256, 2, 256, 2, 25>;
#endif

template <class S>
__launch_bounds__(S::kSamplerBlock) __global__ void speculative_sampling_partial_topk_kernel(
    const __nv_bfloat16* logits, TokenRows drafts, const std::int32_t* current_extents,
    const SamplingConfig* configs, std::int32_t token_domain, std::int32_t physical_rows,
    std::int32_t cols, std::int32_t k, Workspace workspace, std::size_t workspace_row_stride,
    const std::int32_t* live, std::int32_t row_begin) {
    const int row = row_begin + static_cast<int>(blockIdx.z);
    if (live && !live[row]) return;
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

    const int cap           = greedy ? 1 : sampling_candidate_cap(cfg, token_domain);
    const std::int64_t base = (static_cast<std::int64_t>(row) * cols + col) * physical_rows;
    const auto row_drafts   = drafts[row];
    const int tile_start    = partial * S::kSamplerPartialTileItems;
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

template <int Threads, int Items>
__launch_bounds__(Threads, 3) __global__ void speculative_sampling_threshold_topk_kernel(
    const __nv_bfloat16* logits, TokenRows drafts, const std::int32_t* current_extents,
    const SamplingConfig* configs, std::int32_t token_domain, std::int32_t physical_rows,
    std::int32_t cols, std::int32_t k, Workspace workspace, std::size_t workspace_row_stride,
    const std::int32_t* live, std::int32_t row_begin) {
    const int row = row_begin + int(blockIdx.z), col = blockIdx.y;
    if (live && !live[row]) return;
    const int extent = min(k, max(0, current_extents[row]));
    if (col > extent) return;
    const SamplingConfig cfg = configs[row];
    const bool greedy        = !(cfg.temperature > 0.F);
    const bool penalties     = cfg.presence_penalty != 0.F || cfg.frequency_penalty != 0.F;
    if (greedy && !penalties && !cfg.mask.words) return;
    workspace          = speculative_workspace_row(workspace, workspace_row_stride, row);
    using Selector     = ThresholdSelection<Threads, Items>;
    using Compact      = ThresholdSelection<Threads, Items, unsigned int>;
    constexpr int Tile = Threads * Items;

    __shared__ union {
        typename Selector::Storage wide;
        typename Compact::Storage compact;
        SamplingInputTile<Threads, Tile> inputs;
    } storage;

    __shared__ DraftCountTile<Tile> overlay;
    __shared__ unsigned int compact_keys[kSamplerCandidateCap];
    const int partials = (token_domain - 1) / Tile + 1;
    for (int partial = blockIdx.x; partial < partials; partial += gridDim.x) {
        if (partial == 0 && threadIdx.x == 0) {
            workspace.group_done[col] = 0;
            if (col == 0) *workspace.speculative_finalize_count = 0;
        }
        const int start     = partial * Tile;
        const auto base     = (std::int64_t(row) * cols + col) * physical_rows;
        const auto* history = cfg.token_counts ? cfg.token_counts + start : nullptr;
        storage.inputs.prefetch(logits + base + start, history, min(Tile, token_domain - start),
                                penalties);
        if (penalties) overlay.prepare(drafts[row], col, start);
        cp_wait<0>();
        __syncthreads();
        const int cap = greedy ? 1 : sampling_candidate_cap(cfg, token_domain);
        auto* result = workspace.partial_keys + sampling_partial_offset(workspace, col, partial, 0);
        if (!penalties) {
            unsigned int values[Items];
#pragma unroll
            for (int i = 0; i < Items; ++i) {
                const int local = i * Threads + threadIdx.x, token = start + local;
                values[i] =
                    token < token_domain &&
                            (!cfg.mask.words ||
                             (cfg.mask.words[std::int64_t(col) * cfg.mask.stride + token / 32] &
                              (1U << (token % 32))))
                        ? sampling_bf16_tile_sort_key<Compact>(storage.inputs.logits[local], local)
                        : 0;
            }
            __syncthreads();
            Compact::select_values(values, cap, compact_keys, storage.compact);
            __syncthreads();
            if (threadIdx.x < cap)
                result[threadIdx.x] =
                    sampling_expand_bf16_tile_sort_key<Compact>(compact_keys[threadIdx.x], start);
        } else {
            unsigned long long values[Items];
#pragma unroll
            for (int i = 0; i < Items; ++i) {
                const int local = i * Threads + threadIdx.x, token = start + local;
                values[i] = 0;
                if (token < token_domain) {
                    const bool valid =
                        !cfg.mask.words ||
                        (cfg.mask.words[std::int64_t(col) * cfg.mask.stride + token / 32] &
                         (1U << (token % 32)));
                    const float score =
                        valid ? sampling_penalized_logit(
                                    __bfloat162float(storage.inputs.logits[local]), cfg,
                                    storage.inputs.counts[local] + overlay.count(local))
                              : -CUDART_INF_F;
                    values[i] = score_id_order_key(score, token);
                }
            }
            __syncthreads();
            Selector::select_values(values, cap, result, storage.wide);
        }
        __syncthreads();
    }
}

} // namespace

void launch_speculative_selection(const SpeculativeSelectionInput& input, std::int32_t rows,
                                  std::int32_t partials, std::int32_t tile,
                                  CudaExecutionView execution) {
    const auto stream = execution.stream();
    // Bound empty CTAs without serializing too many tiles in a live CTA.
    const int tile_groups =
        std::min(partials, std::max(common::div_up(partials, 8),
                                    common::div_up(execution.multiprocessor_count() * 4, rows)));
    const dim3 grid(tile == 4096 && rows >= 32 ? tile_groups : partials, input.columns, rows);
    if (tile == 4096)
        speculative_sampling_threshold_topk_kernel<256, 16><<<grid, 256, 0, stream>>>(
            input.logits, input.drafts, input.current_extents, input.configs, input.token_domain,
            input.physical_rows, input.columns, input.draft_count, input.workspace,
            input.row_stride, input.live_rows, input.first_row);
    else
        speculative_sampling_partial_topk_kernel<Schedule>
            <<<grid, Schedule::kSamplerBlock, 0, stream>>>(
                input.logits, input.drafts, input.current_extents, input.configs,
                input.token_domain, input.physical_rows, input.columns, input.draft_count,
                input.workspace, input.row_stride, input.live_rows, input.first_row);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::sampling_detail
