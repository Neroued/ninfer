#include "ops/sampling/acceptance/sparse_drafts/plan.h"
#include "ops/sampling/acceptance/sparse_drafts/tile.cuh"
#include "core/device.h"
#include "core/pdl.cuh"

#include <cuda/atomic>

namespace ninfer::ops::sparse_drafts_detail {
using namespace sampling_detail;
using namespace common;

namespace {
__global__ __launch_bounds__(256) void speculative_accept_sparse_warp_greedy_kernel(
    const int* target_tokens, TokenRows drafts, const int* current_extents, int* licensed_tokens,
    int* licensed_counts, int k, int batch, const int* live) {
    const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= batch) return;
    if (live && !live[row]) {
        const int lane = threadIdx.x & 31;
        if (lane <= k) licensed_tokens[std::int64_t(row) * (k + 1) + lane] = 0;
        if (lane == 0) licensed_counts[row] = 0;
        return;
    }
    const int extent = drafts.extent(current_extents[row], k);
    speculative_sparse_warp_greedy(target_tokens, drafts, licensed_tokens, licensed_counts, row,
                                   extent, k);
}

__global__ void merge_sparse_partials(SpeculativeSelectionInput input, int read_begin,
                                      int read_lists, int write_begin) {
    const int row = input.first_row + static_cast<int>(blockIdx.z);
    const int col = blockIdx.y;
    if (input.live_rows && !input.live_rows[row]) return;
    if (col > input.drafts.extent(input.current_extents[row], input.columns - 1)) return;
    const auto cfg    = input.configs[row];
    const bool greedy = !(cfg.temperature > 0.F);
    if (greedy && cfg.presence_penalty == 0.F && cfg.frequency_penalty == 0.F && !cfg.mask.words)
        return;
    const int cap   = greedy ? 1 : sampling_candidate_cap(cfg, input.token_domain);
    const int begin = blockIdx.x * Schedule::kSamplerPartialsPerGroup;
    const int count = min(read_lists - begin, Schedule::kSamplerPartialsPerGroup);
    auto workspace  = speculative_workspace_row(input.workspace, input.row_stride, row);
    unsigned long long keys[Schedule::kSamplerItemsPerThread];
#pragma unroll
    for (int i = 0; i < Schedule::kSamplerItemsPerThread; ++i) {
        const int p = i * blockDim.x + threadIdx.x;
        keys[i]     = p < count * cap ? workspace.partial_keys[sampling_partial_offset(
                                        workspace, col, read_begin + begin + p / cap, p % cap)]
                                      : 0ull;
    }
    __shared__ SamplingTileTopKStorage<Schedule> storage;
    sampling_store_tile_topk<Schedule>(keys, cap, workspace, col, write_begin + blockIdx.x,
                                       storage);
}

__device__ void publish_sparse_distribution(Workspace workspace, const SamplingConfig& cfg,
                                            bool greedy, int col, int extent, int cap,
                                            const unsigned long long* top, bool& last_column) {
    if (threadIdx.x == 0) {
        const int offset = sampling_dist_offset(col, 0);
        if (greedy || cap == 1) {
            if ((!greedy || cfg.mask.words) && !isfinite(score_from_order_key(top[0])))
                asm volatile("trap;");
            workspace.dist_idx[offset] = id_from_order_key(top[0]);
            if (!greedy) {
                workspace.dist_support[col] = 1;
                workspace.dist_prob[offset] = 1.F;
            }
        } else {
            float scores[kSamplerCandidateCap], probabilities[kSamplerCandidateCap];
            for (int i = 0; i < cap; ++i) scores[i] = score_from_order_key(top[i]);
            int support = 0;
            sampling_normalize_support_values(cfg, scores, probabilities, &support, cap);
            workspace.dist_support[col] = support;
            for (int i = 0; i < support; ++i) {
                workspace.dist_idx[offset + i]  = id_from_order_key(top[i]);
                workspace.dist_prob[offset + i] = probabilities[i];
            }
        }
        cuda::atomic_ref<int, cuda::thread_scope_device> finished(
            *workspace.speculative_finalize_count);
        last_column = finished.fetch_add(1, cuda::memory_order_acq_rel) == extent;
    }
    __syncthreads();
}

template <int Threads, int Items>
__global__ void finalize_sparse_partials(SpeculativeSelectionInput input, int read_begin,
                                         int read_lists, const int* target_tokens,
                                         SparseProposalView proposal, const int* base_positions,
                                         int* licensed_tokens, int* licensed_counts) {
    const int row = input.first_row + static_cast<int>(blockIdx.z);
    const int col = blockIdx.y, tid = threadIdx.x, k = input.columns - 1;
    if (input.live_rows && !input.live_rows[row]) {
        if (col == 0 && tid <= k) {
            licensed_tokens[std::int64_t(row) * input.columns + tid] = 0;
            if (tid == 0) licensed_counts[row] = 0;
        }
        return;
    }
    const int extent = input.drafts.extent(input.current_extents[row], k);
    if (col > extent) return;
    const auto cfg    = input.configs[row];
    const bool greedy = !(cfg.temperature > 0.F);
    if (greedy && cfg.presence_penalty == 0.F && cfg.frequency_penalty == 0.F && !cfg.mask.words) {
        if (col == 0 && tid < 32)
            speculative_sparse_warp_greedy(target_tokens, input.drafts, licensed_tokens,
                                           licensed_counts, row, extent, k);
        return;
    }

    auto workspace = speculative_workspace_row(input.workspace, input.row_stride, row);
    const int cap  = greedy ? 1 : sampling_candidate_cap(cfg, input.token_domain);
    using Selector = ThresholdSelection<Threads, Items>;
    __shared__ typename Selector::Storage storage;
    __shared__ unsigned long long top[kSamplerCandidateCap];
    __shared__ bool last_column;
    const auto* lists =
        workspace.partial_keys + sampling_partial_offset(workspace, col, read_begin, 0);
    unsigned long long keys[Items];
#pragma unroll
    for (int i = 0; i < Items; ++i) {
        const int at = i * Threads + tid;
        keys[i]      = at < read_lists * kSamplerCandidateCap && at % kSamplerCandidateCap < cap
                           ? lists[at]
                           : 0ull;
    }
    Selector::select_values(keys, cap, top, storage);
    __syncthreads();

    publish_sparse_distribution(workspace, cfg, greedy, col, extent, cap, top, last_column);
    if (last_column && tid < 32)
        speculative_sparse_warp_accept(workspace, input.drafts, proposal, cfg, k, row, extent,
                                       greedy, base_positions, licensed_tokens, licensed_counts);
}

// Nonnegative penalties let the first pass rank raw logits, deferring the history
// lookup until the leading candidate is known.
__device__ bool raw_scores_are_bounds(const SamplingConfig& cfg) {
    return cfg.presence_penalty >= 0.F && cfg.frequency_penalty >= 0.F;
}

using SparseKey = unsigned long long;

__device__ __forceinline__ float sparse_bf16_score(__nv_bfloat16 value) {
    return __uint_as_float(unsigned(__bfloat16_as_ushort(value)) << 16);
}

__device__ bool sparse_token_allowed(const SamplingConfig& cfg, int col, int token) {
    return !cfg.mask.words || (cfg.mask.words[std::int64_t(col) * cfg.mask.stride + token / 32] &
                               (1U << (token % 32)));
}

__device__ float adjust_sparse_score(float score, int token, const SamplingConfig& cfg,
                                     TokenRow drafts, int col) {
    int count = cfg.token_counts ? cfg.token_counts[token] : 0;
    for (int i = 0; i < col; ++i) count += drafts[i] == token;
    return sampling_penalized_logit(score, cfg, count);
}

__device__ __noinline__ BestPair<SparseKey> scan_sparse_scalar(const __nv_bfloat16* logits,
                                                               int first, int last,
                                                               const SamplingConfig* configs,
                                                               TokenRows drafts, int row, int col) {
    const auto cfg    = configs[row];
    const bool adjust = !raw_scores_are_bounds(cfg);
    BestPair<SparseKey> pair{};
    for (int token = first + threadIdx.x; token < last; token += blockDim.x) {
        if (!sparse_token_allowed(cfg, col, token)) continue;
        float score = sparse_bf16_score(logits[token]);
        if (adjust) score = adjust_sparse_score(score, token, cfg, drafts[row], col);
        pair.add(score_id_order_key(score, token));
    }
    return pair;
}

struct StreamingKey {
    static constexpr int kSamplingTileIndexBits      = 16;
    static constexpr unsigned kSamplingTileIndexMask = 65535;
};

__device__ __forceinline__ __nv_bfloat16 bf16_threshold(unsigned key) {
    if (!key) return __ushort_as_bfloat16(0xff80);
    const unsigned ordered = key >> 16;
    const unsigned bits    = ordered & 0x8000U ? ordered ^ 0x8000U : ~ordered;
    return __ushort_as_bfloat16(static_cast<unsigned short>(bits));
}

__device__ __forceinline__ __nv_bfloat162 bf16_pair(unsigned bits) {
    return __nv_bfloat162(__nv_bfloat162_raw{static_cast<unsigned short>(bits),
                                             static_cast<unsigned short>(bits >> 16)});
}

__device__ __forceinline__ __nv_bfloat162 sparse_vector_max(uint4 v) {
    const auto max02 = __hmax2_nan(bf16_pair(v.x), bf16_pair(v.y));
    const auto max46 = __hmax2_nan(bf16_pair(v.z), bf16_pair(v.w));
    return __hmax2_nan(max02, max46);
}

__device__ __forceinline__ void consider_sparse_vector(uint4 v, int ordinal,
                                                       BestPair<unsigned>& pair) {
    const auto maximum = sparse_vector_max(v);
    const auto top     = __hmax_nan(maximum.x, maximum.y);
    // This lane visits increasing token IDs. A later tie cannot displace its
    // runner-up. An unordered comparison always goes through exact key ordering.
    if (pair.second && !__hgtu(top, bf16_threshold(pair.second))) return;
    const unsigned words[] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const auto value = static_cast<unsigned short>(words[i / 2] >> (16 * (i & 1)));
        pair.add(
            sampling_bf16_tile_sort_key<StreamingKey>(__ushort_as_bfloat16(value), ordinal + i));
    }
}

template <int Threads>
__device__ __forceinline__ BestPair<SparseKey> scan_sparse_vector(const __nv_bfloat16* logits,
                                                                  int first, int last) {
    const int tid          = threadIdx.x;
    constexpr int stride   = 8 * Threads;
    constexpr int Prefetch = 4;
    const int available    = last - first;
    int local = tid * 8, ordinal = 0;
    BestPair<unsigned> pair{};
    for (; local + (Prefetch - 1) * stride + 8 <= available;
         local += Prefetch * stride, ordinal += Prefetch * 8) {
        uint4 values[Prefetch];
#pragma unroll
        for (int i = 0; i < Prefetch; ++i)
            values[i] = load_ldg<uint4>(logits + first + local + i * stride);
#pragma unroll
        for (int i = 0; i < Prefetch; ++i) consider_sparse_vector(values[i], ordinal + 8 * i, pair);
    }
    for (; local + 8 <= available; local += stride, ordinal += 8)
        consider_sparse_vector(load_ldg<uint4>(logits + first + local), ordinal, pair);
    for (int tail = 0; tail < 8 && local + tail < available; ++tail)
        pair.add(sampling_bf16_tile_sort_key<StreamingKey>(logits[first + local + tail],
                                                           ordinal + tail));
    const auto expand = [=](unsigned key) -> SparseKey {
        if (!key) return 0;
        const int at    = StreamingKey::kSamplingTileIndexMask - (key & 65535U);
        const int token = first + (at / 8) * stride + tid * 8 + at % 8;
        return score_id_order_key(sparse_bf16_score(bf16_threshold(key)), token);
    };
    return {expand(pair.first), expand(pair.second)};
}

template <int Threads>
__device__ void stream_sparse_body(SpeculativeSelectionInput input, SparseKey* summaries,
                                   int partition_size, bool transpose) {
    const int row       = input.first_row + static_cast<int>(blockIdx.z);
    const int col       = transpose ? blockIdx.x : blockIdx.y;
    const int partition = transpose ? blockIdx.y : blockIdx.x;
    const int partials  = transpose ? gridDim.y : gridDim.x;
    if (input.live_rows && !input.live_rows[row]) return;
    if (col > input.drafts.extent(input.current_extents[row], input.columns - 1)) return;
    const auto* cfg = input.configs + row;
    if (!(cfg->temperature > 0.F) && cfg->presence_penalty == 0.F &&
        cfg->frequency_penalty == 0.F && !cfg->mask.words)
        return;
    const auto column  = std::int64_t(row) * input.columns + col;
    const auto* logits = input.logits + column * input.physical_rows;
    const int start    = partition * partition_size;
    const int end      = min(start + partition_size, input.token_domain);
    BestPair<SparseKey> pair;
    if (cfg->mask.words || !raw_scores_are_bounds(*cfg) ||
        (reinterpret_cast<std::uintptr_t>(logits) & 15U) != 0)
        pair = scan_sparse_scalar(logits, start, end, input.configs, input.drafts, row, col);
    else
        pair = scan_sparse_vector<Threads>(logits, start, end);
    __shared__ BestPair<SparseKey> warp_pairs[Threads / 32];
    const auto best = block_best_pair<Threads>(pair, warp_pairs);
    if (threadIdx.x == 0) {
        auto* result = summaries + 2 * (column * partials + partition);
        store_vec(result, uint4{unsigned(best.first), unsigned(best.first >> 32),
                                unsigned(best.second), unsigned(best.second >> 32)});
    }
}

template <int Threads, int Registers>
__maxnreg__(Registers) __global__
    void stream_sparse_summaries(SpeculativeSelectionInput input, SparseKey* summaries,
                                 int partition_size, bool transpose) {
    pdl::trigger_dependents();
    stream_sparse_body<Threads>(input, summaries, partition_size, transpose);
}

__device__ bool singleton_support(const SamplingConfig& cfg, int cap, SparseKey first,
                                  SparseKey second) {
    if (cap == 1 || !second) return true;
    const float best = score_from_order_key(first), next = score_from_order_key(second);
    if (!isfinite(best)) return false;
    if (!isfinite(next)) return next < 0.F;
    const float inverse   = 1.F / cfg.temperature;
    const float runner_up = __expf(next * inverse - best * inverse) + 0x1p-20f;
    if (cfg.min_p > 0.F && runner_up < cfg.min_p) return true;
    if (cfg.top_p >= 1.F) return false;
    // Bound both softmax rounding and the serial accumulation of at most 20 weights.
    const float total_bound = 1.F + float(cap - 1) * runner_up;
    return cfg.top_p * fmaf(total_bound, 0x1p-18f, total_bound) <= 1.F;
}

__device__ SparseKey merge_sparse_top_keys(SparseKey key, SparseKey next) {
    const int lane = threadIdx.x & 31;
    key            = max(key, __shfl_xor_sync(0xffffffffU, next, 31));
#pragma unroll
    for (int distance = 16; distance > 0; distance /= 2) {
        const auto other = __shfl_xor_sync(0xffffffffU, key, distance);
        key              = (lane & distance) == 0 ? max(key, other) : min(key, other);
    }
    return key;
}

constexpr int kRescanItems = 2;
using SparseWarpSort       = cub::WarpMergeSort<SparseKey, kRescanItems>;

struct SparseRowDistributions {
    int ids[16 * kSamplerCandidateCap];
    float p[16 * kSamplerCandidateCap];
    int support[16];
};

__device__ __noinline__ void
rescan_sparse_column(const __nv_bfloat16* logits, const SparseKey* summaries, int partials,
                     int partition_size, int domain, const SamplingConfig& cfg, TokenRow drafts,
                     int col, int cap, bool bounds, SparseWarpSort::TempStorage& sort,
                     SparseRowDistributions& dist) {
    constexpr unsigned mask = 0xffffffffU;
    const int lane          = threadIdx.x & 31;
    const bool penalties    = cfg.presence_penalty != 0.F || cfg.frequency_penalty != 0.F;
    SparseKey best          = 0;
    for (int part = 0; part < partials; ++part) {
        const auto cutoff = __shfl_sync(mask, best, cap - 1);
        const auto first = summaries[2 * part], second = summaries[2 * part + 1];
        if (first <= cutoff) continue;
        if (second <= cutoff) {
            SparseKey candidate = lane == 0 ? first : 0ULL;
            if (bounds && lane == 0) {
                const int token = id_from_order_key(first);
                candidate       = score_id_order_key(
                    adjust_sparse_score(score_from_order_key(first), token, cfg, drafts, col),
                    token);
            }
            best = merge_sparse_top_keys(best, candidate);
            continue;
        }
        const int begin = part * partition_size;
        const int end   = min(domain, begin + partition_size);
        for (int offset = begin; offset < end; offset += 32 * kRescanItems) {
            const auto threshold = __shfl_sync(mask, best, cap - 1);
            // After passing the partition winner, the runner-up bounds every
            // remaining score. Even a tie can only use an ID at or past offset.
            const auto upper = offset > id_from_order_key(first) ? second : first;
            if (score_id_order_key(score_from_order_key(upper), offset) <= threshold) break;
            SparseKey keys[kRescanItems];
            SparseKey maximum = 0;
#pragma unroll
            for (int i = 0; i < kRescanItems; ++i) {
                const int token = offset + i * 32 + lane;
                keys[i]         = 0;
                if (token < end && sparse_token_allowed(cfg, col, token)) {
                    float score = sparse_bf16_score(logits[token]);
                    if (penalties) score = adjust_sparse_score(score, token, cfg, drafts, col);
                    keys[i] = score_id_order_key(score, token);
                }
                maximum = max(maximum, keys[i]);
            }
            if (!__any_sync(mask, maximum > threshold)) continue;
            SparseWarpSort(sort).Sort(keys, ScoreIdOrderGreater{});
            // Each sorted thread owns eight consecutive ranks. Collect the one
            // required by each destination lane using the intra-thread item index.
            SparseKey rank = 0;
#pragma unroll
            for (int i = 0; i < kRescanItems; ++i) {
                const auto candidate = __shfl_sync(mask, keys[i], lane / kRescanItems);
                if ((lane & (kRescanItems - 1)) == i) rank = candidate;
            }
            best = merge_sparse_top_keys(best, lane < cap ? rank : 0ULL);
            __syncwarp();
        }
    }

    const int offset = sampling_dist_offset(col, 0);
    if (lane < cap) {
        dist.ids[offset + lane] = id_from_order_key(best);
        dist.p[offset + lane]   = score_from_order_key(best);
    }
    __syncwarp();
    if (lane == 0) {
        if (cap == 1) {
            if ((cfg.mask.words || cfg.temperature > 0.F) && !isfinite(dist.p[offset]))
                asm volatile("trap;");
            dist.p[offset]    = 1.F;
            dist.support[col] = 1;
        } else {
            sampling_normalize_support_values(cfg, dist.p + offset, dist.p + offset,
                                              dist.support + col, cap);
        }
    }
}

__device__ __forceinline__ void finalize_sparse_row(SpeculativeSelectionInput input,
                                                    const SparseKey* summaries, int partials,
                                                    int partition_size, const int* target_tokens,
                                                    SparseProposalView proposal,
                                                    const int* base_positions, int* licensed_tokens,
                                                    int* licensed_counts, int row) {
    const int col = threadIdx.x / 32, lane = threadIdx.x & 31, k = input.columns - 1;
    const auto row_base = std::int64_t(row) * input.columns;
    if (input.live_rows && !input.live_rows[row]) {
        if (threadIdx.x <= k) licensed_tokens[row_base + threadIdx.x] = 0;
        if (threadIdx.x == 0) licensed_counts[row] = 0;
        return;
    }
    const int extent     = input.drafts.extent(input.current_extents[row], k);
    const auto cfg       = input.configs[row];
    const bool greedy    = !(cfg.temperature > 0.F);
    const bool penalties = cfg.presence_penalty != 0.F || cfg.frequency_penalty != 0.F;
    if (greedy && !penalties && !cfg.mask.words) {
        if (col == 0)
            speculative_sparse_warp_greedy(target_tokens, input.drafts, licensed_tokens,
                                           licensed_counts, row, extent, k);
        return;
    }

    pdl::wait_for_dependencies();
    __shared__ SparseRowDistributions dist;
    __shared__ SparseWarpSort::TempStorage sort[16];
    __shared__ int winner_ids[16];
    const auto* pairs = summaries + 2 * (row_base + col) * partials;
    const bool bounds = penalties && raw_scores_are_bounds(cfg);
    const int cap     = greedy ? 1 : sampling_candidate_cap(cfg, input.token_domain);
    bool singleton    = true;
    if (col <= extent) {
        BestPair<SparseKey> local{};
        const auto read_pair = [&](int partition) {
            const auto words = load_vec<uint4>(pairs + 2 * partition);
            return BestPair<SparseKey>{(SparseKey(words.y) << 32) | words.x,
                                       (SparseKey(words.w) << 32) | words.z};
        };
        if (lane < partials) local = read_pair(lane);
        if (lane + 32 < partials) {
            const auto extra = read_pair(lane + 32);
            local.add(extra.first);
            local.add(extra.second);
        }
        const auto first  = warp_max(local.first);
        const auto second = warp_max(first == local.first ? local.second : local.first);
        SparseKey best    = first;
        if (bounds && first) {
            const auto token      = id_from_order_key(first);
            int count             = lane == 0 && cfg.token_counts ? cfg.token_counts[token] : 0;
            const bool matched    = lane < col && input.drafts[row][lane] == token;
            const unsigned prefix = __ballot_sync(0xffffffffU, matched);
            if (lane == 0) {
                count += __popc(prefix);
                best = score_id_order_key(
                    sampling_penalized_logit(score_from_order_key(first), cfg, count), token);
            }
        }
        if (lane == 0) {
            singleton = (!bounds || best >= second) && singleton_support(cfg, cap, best, second);
            if (singleton && (!greedy || cfg.mask.words) && !isfinite(score_from_order_key(best)))
                asm volatile("trap;");
            winner_ids[col] = id_from_order_key(best);
        }
        singleton = __shfl_sync(0xffffffffU, int(singleton), 0);
    }
    const bool rescan = __syncthreads_or(!singleton);
    if (!rescan) {
        if (col == 0) {
            const bool reject     = lane < extent && winner_ids[lane] != input.drafts[row][lane];
            const unsigned failed = __ballot_sync(0xffffffffU, reject);
            const int accepted    = failed ? __ffs(failed) - 1 : extent;
            speculative_sparse_warp_store(input.drafts, k, row, accepted, winner_ids[accepted],
                                          licensed_tokens, licensed_counts);
        }
        return;
    }

    if (col <= extent) {
        if (singleton) {
            if (lane == 0) {
                const int offset  = sampling_dist_offset(col, 0);
                dist.ids[offset]  = winner_ids[col];
                dist.p[offset]    = 1.F;
                dist.support[col] = 1;
            }
        } else {
            const auto* logits = input.logits + (row_base + col) * input.physical_rows;
            rescan_sparse_column(logits, pairs, partials, partition_size, input.token_domain, cfg,
                                 input.drafts[row], col, cap, bounds, sort[col], dist);
        }
    }
    __syncthreads();
    if (col == 0) {
        const Workspace shared{
            .dist_idx = dist.ids, .dist_prob = dist.p, .dist_support = dist.support};
        speculative_sparse_warp_accept(shared, input.drafts, proposal, cfg, k, row, extent, greedy,
                                       base_positions, licensed_tokens, licensed_counts);
    }
}

__launch_bounds__(512) __global__
    void finalize_sparse_rows(SpeculativeSelectionInput input, const SparseKey* summaries,
                              int partials, int partition_size, const int* target_tokens,
                              SparseProposalView proposal, const int* base_positions,
                              int* licensed_tokens, int* licensed_counts) {
    finalize_sparse_row(input, summaries, partials, partition_size, target_tokens, proposal,
                        base_positions, licensed_tokens, licensed_counts,
                        input.first_row + static_cast<int>(blockIdx.x));
}

} // namespace

void launch(const Tensor& target_tokens, const Tensor& logits, const Tensor& drafts,
            const Tensor& candidate_ids, const Tensor& proposal_q, const Tensor& current_extents,
            const Tensor& base_positions, Tensor& licensed_tokens, Tensor& licensed_counts,
            std::int32_t token_domain, const SamplingConfig* configs, const Tensor& live_rows,
            const Plan& plan, MemorySpan workspace, CudaExecutionView execution) {
    const auto stream        = execution.stream();
    const std::int32_t batch = drafts.ne[1];
    const std::int32_t k     = licensed_tokens.ne[0] - 1;
    const std::int32_t cols  = k + 1;
    const auto* live         = static_cast<const std::int32_t*>(live_rows.data);
    if (plan.raw_greedy) {
        const int warps = std::min(batch, 8);
        speculative_accept_sparse_warp_greedy_kernel<<<div_up(batch, warps), 32 * warps, 0,
                                                       stream>>>(
            static_cast<const int*>(target_tokens.data), token_rows(drafts),
            static_cast<const int*>(current_extents.data), static_cast<int*>(licensed_tokens.data),
            static_cast<int*>(licensed_counts.data), k, batch, live);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const auto& layout                = plan.workspace;
    const std::int32_t partial_blocks = plan.partial_blocks;
    const std::int32_t groups         = plan.groups;
    const Workspace scratch           = plan.streaming ? Workspace{} : layout.bind(workspace);
    for (int begin = 0; begin < batch;) {
        const int rows = std::min(batch - begin, 65535);
        const SpeculativeSelectionInput input{
            .logits          = static_cast<const __nv_bfloat16*>(logits.data),
            .drafts          = token_rows(drafts),
            .current_extents = static_cast<const std::int32_t*>(current_extents.data),
            .configs         = configs,
            .live_rows       = live,
            .token_domain    = token_domain,
            .physical_rows   = logits.ne[0],
            .columns         = cols,
            .workspace       = scratch,
            .row_stride      = layout.bytes,
            .first_row       = begin};
        const auto* targets   = static_cast<const int*>(target_tokens.data);
        const auto proposal   = sparse_proposal(candidate_ids, proposal_q);
        const auto* positions = static_cast<const int*>(base_positions.data);
        auto* tokens          = static_cast<int*>(licensed_tokens.data);
        auto* counts          = static_cast<int*>(licensed_counts.data);
        if (plan.streaming) {
            auto* summaries              = static_cast<SparseKey*>(workspace.data);
            const auto columns_in_flight = std::int64_t(rows) * cols;
            const bool transpose         = columns_in_flight <= 8;
            const dim3 grid(transpose ? cols : partial_blocks, transpose ? partial_blocks : cols,
                            rows);
            constexpr int threads = 256;
            if (columns_in_flight >= 64)
                stream_sparse_summaries<threads, 32>
                    <<<grid, threads, 0, stream>>>(input, summaries, plan.partial_tile, transpose);
            else
                stream_sparse_summaries<threads, 40>
                    <<<grid, threads, 0, stream>>>(input, summaries, plan.partial_tile, transpose);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(pdl::launch_dependent(
                {dim3(rows), dim3(32 * cols), 0, stream}, finalize_sparse_rows, input, summaries,
                partial_blocks, plan.partial_tile, targets, proposal, positions, tokens, counts));
            begin += rows;
            continue;
        }

        launch_speculative_selection(input, rows, partial_blocks, plan.partial_tile, execution);
        if (groups * kSamplerCandidateCap > Schedule::kSamplerGroupTileItems) {
            // Alternate disjoint ranges until one block can merge all remaining lists.
            int read_begin = 0, read_lists = partial_blocks;
            while (read_lists > kFinalMergePartials) {
                const int merged      = div_up(read_lists, Schedule::kSamplerPartialsPerGroup);
                const int write_begin = read_begin == 0 ? partial_blocks : 0;
                const dim3 merge_grid(merged, cols, rows);
                merge_sparse_partials<<<merge_grid, Schedule::kSamplerBlock, 0, stream>>>(
                    input, read_begin, read_lists, write_begin);
                CUDA_CHECK(cudaGetLastError());
                read_begin = write_begin;
                read_lists = merged;
            }
            constexpr int threads = 128, items = 8;
            static_assert(threads * items >= kFinalMergePartials * kSamplerCandidateCap);
            const dim3 grid(1, cols, rows);
            finalize_sparse_partials<threads, items><<<grid, threads, 0, stream>>>(
                input, read_begin, read_lists, targets, proposal, positions, tokens, counts);
            CUDA_CHECK(cudaGetLastError());
            begin += rows;
            continue;
        }

        const dim3 group_grid(static_cast<unsigned int>(groups), static_cast<unsigned int>(cols),
                              static_cast<unsigned int>(rows));

        speculative_sampling_group_finalize_kernel<Schedule, true>
            <<<group_grid, Schedule::kSamplerGroupBlock, 0, stream>>>(
                targets, input.drafts, proposal, input.current_extents, positions, tokens, counts,
                configs, token_domain, cols, partial_blocks, groups, scratch, layout.bytes, live,
                begin);

        CUDA_CHECK(cudaGetLastError());
        begin += rows;
    }
}

} // namespace ninfer::ops::sparse_drafts_detail
