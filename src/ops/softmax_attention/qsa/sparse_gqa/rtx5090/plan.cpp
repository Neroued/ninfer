#include "ops/softmax_attention/qsa/sparse_gqa/plan.h"
#include "ops/softmax_attention/qsa/sparse_gqa/rtx5090/compact_schedule.h"
#include "core/paged_kv_cache.h"

#include <algorithm>
#include <limits>

namespace ninfer::ops::detail::qsa {
namespace {
constexpr int kSplitQueries                = 4096;
constexpr std::uint64_t kSplitCacheBytes   = 64ULL << 20;
constexpr std::uint32_t kMaxSelectedTokens = 512 * 4 + 3;

std::uint32_t two_head_prefix_limit(KvCacheStorage storage, std::int64_t queries, int sms) {
    if (storage == KvCacheStorage::BFloat16) {
        if (queries + 1 >= std::int64_t(sms) * 3 / 2) return kMaxSelectedTokens;
        return std::uint32_t(std::min<std::int64_t>(kMaxSelectedTokens, queries));
    }
    if (storage == KvCacheStorage::Fp8E4m3fnGroup128Ue8m0 ||
        storage == KvCacheStorage::Fp8KeyNvfp4Value)
        return kMaxSelectedTokens;
    const int reuse   = storage == KvCacheStorage::Nvfp4Group16 ? 11 : 12;
    const auto cutoff = (kMaxSelectedTokens + reuse - 1) / reuse;
    return queries < cutoff ? std::uint32_t(queries * reuse) : kMaxSelectedTokens;
}

int native_natural_partitions(std::int64_t queries) {
    return queries <= 6 ? 64 : queries <= 12 ? 33 : queries == 13 ? 32 : 22;
}

double partial_merge_rate(KvCacheStorage storage, int kv_heads, std::int64_t queries, int batch,
                          int sms, int resident, bool large, bool sparse_history) {
    const bool native = storage == KvCacheStorage::BFloat16;
    const bool fp4    = storage == KvCacheStorage::Nvfp4Group16;
    const bool k8v4   = storage == KvCacheStorage::Fp8KeyNvfp4Value;
    if (kv_heads == 4) {
        if (large) return fp4 ? 3. : 4.3;
        if (!native && queries > std::int64_t(sms) * 3) return fp4 ? 1.64 : 1.66;
        return .05;
    }
    if (queries >= std::int64_t(sms) * 4) {
        if (!sparse_history || fp4) return large ? 2. : native ? .6 : 1.2;
        if (large) return k8v4 ? 2.84 : 3.;
        if (native) return queries >= std::int64_t(sms) * 8 ? 1.5 : .6;
        if (queries >= std::int64_t(sms) * 8) {
            if (storage == KvCacheStorage::Int8Group64) return 2.245;
            if (storage == KvCacheStorage::Fp8E4m3fnGroup128Ue8m0) return 2.37;
        }
        return 2.5;
    }
    if (!sparse_history || fp4)
        return large ? .8 : storage == KvCacheStorage::Int8Group64 ? .6 : .5;
    if (large) return k8v4 ? (batch == 2 ? 2.3 : .8) : 2.5;
    if (native) return .635;
    return queries * kv_heads >= std::int64_t(resident) * sms ? 2. : .5;
}

struct MmaProblem {
    KvCacheStorage storage;
    int kv_heads, batch, sms;
    std::int64_t queries;
    bool large, sparse_history;
};

struct MmaTileCost {
    int resident;
    double tile, startup, merge_rate;
    bool native_reuse_window;
};

MmaTileCost mma_tile_cost(const MmaProblem& p, int rows, bool reuse, bool delay) {
    const bool native  = p.storage == KvCacheStorage::BFloat16;
    const bool fp4     = p.storage == KvCacheStorage::Nvfp4Group16;
    const int resident = compact_mma_resident_blocks(p.storage, p.kv_heads, rows, reuse, delay);
    double tile        = 1.;
    if (reuse && p.kv_heads == 4) {
        if (p.large)
            tile = fp4 ? 1.31 : 1.46;
        else if (fp4)
            tile = p.queries >= 256 && p.queries <= std::int64_t(p.sms) * 3 ? 1.165 : 1.20;
        else
            tile = 1.21;
    } else if (reuse) {
        switch (p.storage) {
        case KvCacheStorage::Nvfp4Group16:
            tile = 1.8;
            break;
        case KvCacheStorage::Int8Group64:
            tile = 2.3;
            break;
        case KvCacheStorage::Fp8E4m3fnGroup128Ue8m0:
        case KvCacheStorage::Fp8KeyNvfp4Value:
            tile = 2.4;
            break;
        default:
            tile = 2.65;
            break;
        }
    } else if (p.kv_heads == 2) {
        if (rows == 16)
            tile = 1.25;
        else if (native && delay)
            tile = p.sparse_history ? 1.63 : 1.65;
        else
            tile = fp4 ? 1.3 : 1.5;
    }
    // Once two partitions spill a five-CTA wave, three still fit in two waves.
    const bool native_reuse = native && p.kv_heads == 4 &&
                              p.queries * 8 > std::int64_t(p.sms) * 5 &&
                              p.queries * 12 <= std::int64_t(p.sms) * 10;
    double startup;
    if (native_reuse)
        startup = 2;
    else if (p.kv_heads == 2)
        startup = p.large ? 8 : 3;
    else if (p.large)
        startup = fp4 ? 4 : 8;
    else
        startup = reuse ? 3.5 : 2;
    return {resident, tile, startup,
            partial_merge_rate(p.storage, p.kv_heads, p.queries, p.batch, p.sms, resident, p.large,
                               p.sparse_history),
            native_reuse};
}

SparseGqaPlan partition_mma(KvCacheStorage storage, int kv_heads, std::int64_t queries, int batch,
                            std::uint32_t visible, int sms, SparseGqaPlan plan) {
    const bool native         = storage == KvCacheStorage::BFloat16;
    const bool fp4            = storage == KvCacheStorage::Nvfp4Group16;
    const bool sparse_history = visible > kMaxSelectedTokens;
    const auto row_bytes = paged_kv_storage_layout(storage, 256).physical_bytes_per_token_head();
    // Leave space in L2 for Q and the partials.
    const bool large = std::uint64_t(visible) * batch * kv_heads * row_bytes > kSplitCacheBytes;
    const MmaProblem problem{storage, kv_heads, batch, sms, queries, large, sparse_history};
    const int maximum_splits =
        queries > kSplitQueries || (kv_heads == 4 && native && queries >= std::int64_t(sms) * 4) ? 1
        : queries >= std::int64_t(sms) * 4                                                       ? 2
                                           : 64;
    // Native tiles already have ample CTA work here; additional partitions add
    // Q prologues and partial traffic without improving the measured throughput.
    const int limit = native && kv_heads == 2 && large && queries * 4 > std::int64_t(sms) * 5 ? 1
                      : native && kv_heads == 2 && queries >= 256 ? std::min(maximum_splits, 4)
                                                                  : maximum_splits;
    double best     = std::numeric_limits<double>::max();
    SparseGqaPlan selected = plan;
    const int first_rows   = kv_heads == 2 ? 16 : plan.key_rows;
    const int last_rows    = kv_heads == 2 ? 32 : plan.key_rows;
    for (int rows = first_rows; rows <= last_rows; rows *= 2) {
        const int tiles = (int(std::min(visible, kMaxSelectedTokens)) + rows - 1) / rows;
        // One-tile native partitions spend more on partial traffic and merging
        // than they recover in parallelism. Keep two tiles where history permits.
        const int partitions =
            native && kv_heads == 2 && tiles >= 32 ? std::min(limit, (tiles + 1) / 2) : limit;
        for (bool reuse : {false, true}) {
            // Prefer the reused tile once the rotated-value workload fills a
            // five-CTA wave. INT8 also wins just beyond that wave, before the
            // aliased two-partition schedule crosses its next residency step.
            const bool reuse_wave = storage == KvCacheStorage::Fp8KeyNvfp4Value
                                        ? queries * 2 >= std::int64_t(sms) * 5
                                        : storage == KvCacheStorage::Int8Group64 &&
                                              queries * 2 > std::int64_t(sms) * 5 &&
                                              queries <= std::int64_t(sms) * 3;
            if (!reuse && kv_heads == 4 && !large && reuse_wave) continue;
            const bool streaming_reuse =
                queries >= 512 && (storage == KvCacheStorage::Fp8E4m3fnGroup128Ue8m0 ||
                                   storage == KvCacheStorage::Fp8KeyNvfp4Value);
            if (reuse &&
                (native || (kv_heads == 2 && (rows != 32 || (large && !streaming_reuse))) ||
                 (kv_heads == 4 && rows != 16)))
                continue;
            if (reuse && kv_heads == 4 && large && storage != KvCacheStorage::Nvfp4Group16 &&
                queries * 4 > std::int64_t(sms) * 10)
                continue;
            if (kv_heads == 2 && !reuse) {
                const bool wide = native || storage == KvCacheStorage::Nvfp4Group16;
                if (rows == 16 && wide) continue;
                // Only a single wide wave beats narrow tiles for the remaining
                // codecs in the DRAM regime. Else use their higher-throughput tile.
                const bool single_wide_wave = large && queries * 4 > std::int64_t(sms) * 5 &&
                                              queries * 2 <= std::int64_t(sms) * 3;
                if (rows == 32 && !wide && !single_wide_wave) continue;
            }
            const int schedules = kv_heads == 2 && native && rows == 32 ? 2 : 1;
            for (int schedule = 0; schedule < schedules; ++schedule) {
                const bool delay = kv_heads == 2 && rows == 32 && (native ? schedule == 1 : !reuse);
                // Account for per-wave work as well as latency: additional
                // resident warps share issue/decode bandwidth rather than running free.
                const auto model = mma_tile_cost(problem, rows, reuse, delay);
                for (int count = 1; count <= std::min(partitions, tiles); ++count) {
                    // Separate batch rows benefit more from the three-partition
                    // two-wave schedule than from extra Q/partial traffic.
                    if (model.native_reuse_window && batch >= 4 && count > 3) continue;
                    const auto blocks   = queries * kv_heads * count;
                    const auto resident = std::int64_t(model.resident) * sms;
                    const auto waves    = (blocks + resident - 1) / resident;
                    const bool partial  = count > 1;
                    // Native partitions retain a streaming wave for random gathers
                    // while recovering KV reuse when the selected union is smaller.
                    const double merge = partial && !model.native_reuse_window
                                             ? (kv_heads == 2 ? 3.
                                                : large       ? (fp4 ? 4. : 8.)
                                                              : 1.) +
                                                   model.merge_rate * double(queries) * count / sms
                                             : 0.;
                    const double cost =
                        waves * (((tiles + count - 1) / count) * model.tile + model.startup) +
                        merge;
                    if (cost < best) {
                        best                   = cost;
                        selected               = {rows, count, SparseGqaInstance::Mma, partial};
                        selected.delay_softmax = delay;
                        selected.reuse_buffer  = reuse;
                    }
                }
            }
        }
    }
    // A wide batch may contain fewer live queries than its rectangular extent.
    // For FP4, budget two resident waves rather than penalizing those batches
    // with a single long partition wave. This trades a small full-batch cost for
    // substantially lower latency when rows have uneven valid lengths.
    if (kv_heads == 4 && fp4 && large && batch >= 8 && selected.reuse_buffer &&
        selected.splits == 3 && queries <= std::int64_t(sms)) {
        const int two_waves = int(std::int64_t(2 * 10) * sms / (queries * kv_heads));
        selected.splits     = std::min(6, two_waves);
    }
    selected.min_split_tiles = queries >= 256 ? 4 : 1;
    selected.direct_single   = true;
    return selected;
}
} // namespace

std::int64_t sparse_gqa_direct_queries(int kv_heads, int) {
    return kv_heads == 1 ? 256 : kSplitQueries + 1;
}

namespace {
SparseGqaPlan select_plan(KvCacheStorage storage, std::int64_t queries, int batch, int kv_heads,
                          ContextBucket bucket, int sms) {
    const auto visible = context_bucket_tokens(bucket);
    const bool prefix  = visible <= kMaxSelectedTokens;
    const bool native  = storage == KvCacheStorage::BFloat16;
    if (kv_heads == 4) {
        if (!prefix) {
            const int rows = queries <= 8 ? 64 : native ? 32 : 16;
            return partition_mma(storage, kv_heads, queries, batch, visible, sms,
                                 {rows, 1, SparseGqaInstance::Mma, false});
        }
        constexpr int rows = 64, packet = rows / 16;
        int splits = 1;
        if (queries < 256 && (!native || queries < std::int64_t(sms) * 4)) {
            const auto groups = (queries + packet - 1) / packet * 4;
            const auto slots  = std::int64_t(native && queries <= 8 ? 1 : 2) * sms;
            const int tiles   = (int(visible) + rows - 1) / rows;
            // Compare complete waves, including the Q prologue.
            std::int64_t best = std::numeric_limits<std::int64_t>::max();
            for (int count = 1; count <= tiles; ++count) {
                const auto waves = (groups * count + slots - 1) / slots;
                const auto cost  = waves * ((tiles + count - 1) / count + 2);
                if (cost < best) {
                    best   = cost;
                    splits = count;
                }
            }
        }
        return {rows, splits, SparseGqaInstance::FullPrefix, splits > 1};
    }
    if (kv_heads == 1) {
        if (queries >= sparse_gqa_direct_queries(kv_heads, sms))
            return {prefix ? 64 : 32, 1,
                    prefix ? SparseGqaInstance::FullPrefix : SparseGqaInstance::Mma, false};
        const auto target =
            std::max<std::int64_t>(1, (std::int64_t(sms) * 3 / 4 + queries - 1) / queries);
        int splits = 1;
        while (splits < target && splits < 32) splits *= 2;
        return {32, splits, SparseGqaInstance::Mma, true};
    }
    if (prefix && queries >= 32 && visible <= two_head_prefix_limit(storage, queries, sms)) {
        const int rows   = native && queries <= 1024 ? 32 : 64;
        const int splits = !native && queries < 256 ? int((256 + queries - 1) / queries) : 1;
        return {rows, splits, SparseGqaInstance::FullPrefix, splits > 1};
    }
    if (!prefix && queries == 1 && storage == KvCacheStorage::Fp8E4m3fnGroup128Ue8m0) {
        // A wide tile doubles useful key work per CTA while retaining the one-query wave.
        SparseGqaPlan plan{32, 64, SparseGqaInstance::Mma, true};
        plan.direct_single = true;
        return plan;
    }
    auto plan = partition_mma(storage, kv_heads, queries, batch, visible, sms,
                              {32, 1, SparseGqaInstance::Mma, false});
    if (native && !prefix && queries <= 16 && plan.partial)
        plan.natural_splits = native_natural_partitions(queries);
    return plan;
}
} // namespace

SparseGqaPlan select_sparse_gqa_plan(KvCacheStorage storage, int tokens, int batch, int kv_heads,
                                     ContextBucket bucket, int sms) {
    const auto queries = std::int64_t(tokens) * batch;
    auto plan          = select_plan(storage, queries, batch, kv_heads, bucket, sms);
    // Keep the merge slot even when this bucket's attention kernel writes output directly.
    const auto direct_queries = kv_heads == 4 && storage == KvCacheStorage::BFloat16
                                    ? std::int64_t(sms) * 4
                                    : sparse_gqa_direct_queries(kv_heads, sms);
    plan.merge_node           = queries < direct_queries;
    return plan;
}

int sparse_gqa_long_direct_tokens(ContextBucket bucket) {
    return bucket == ContextBucket::K2 ? 0 : 1024;
}

SparseGqaPackedWindows sparse_gqa_packed_windows(KvCacheStorage storage, int kv_heads,
                                                 ContextBucket bucket, bool peeled_run) {
    if (bucket == ContextBucket::K2) return {kSparseGqaSlab};
    // Short BF16 windows restore the split parallelism of compact segment histories.
    // Longer segments reuse their cache across a wider window. The 24/4 FP8 path
    // keeps its warp-addressed key gathering at 128 queries.
    if (kv_heads == 1) return {kSparseGqaSlab};
    if (storage == KvCacheStorage::BFloat16) {
        // The 128-query split schedule reduces scratch and latency in the short
        // run beside a direct span. Unpeeled 24/4 batches retain the wide window.
        return {64, kv_heads == 4 && !peeled_run ? 256 : 128};
    }
    if (kv_heads == 2 && storage == KvCacheStorage::Fp8E4m3fnGroup128Ue8m0) return {256};
    return {128};
}

SparseGqaPlan select_sparse_gqa_packed_window_plan(KvCacheStorage storage, int width, int, int,
                                                   int kv_heads, ContextBucket bucket, int sms) {
    return select_sparse_gqa_plan(storage, width, 1, kv_heads, bucket, sms);
}

SparseGqaLaunchPlan select_sparse_gqa_launch(KvCacheStorage storage, int kv_heads, ContextBucket,
                                             SparseGqaPlan plan, int queries, int, int sms,
                                             bool vector_cache) {
    const bool native = storage == KvCacheStorage::BFloat16;
    if (kv_heads == 2) {
        const bool natural = !vector_cache && plan.natural_splits > 0;
        return {native || plan.reuse_buffer ? SparseGqaLoad::Reused : SparseGqaLoad::Aliased,
                vector_cache, natural ? plan.natural_splits : plan.splits,
                natural || plan.delay_softmax ? SparseGqaKernel::CompactLateSoftmax
                                              : SparseGqaKernel::Sequence};
    }
    if (kv_heads == 4) {
        const auto load = native ? (plan.instance == SparseGqaInstance::FullPrefix && queries <= 8
                                        ? SparseGqaLoad::Direct
                                        : SparseGqaLoad::Reused)
                          : plan.reuse_buffer ? SparseGqaLoad::Reused
                                              : SparseGqaLoad::Aliased;
        return {load, vector_cache, plan.splits};
    }
    if (native || !vector_cache) return {SparseGqaLoad::Direct, vector_cache, plan.splits};
    if (plan.instance == SparseGqaInstance::FullPrefix)
        return {SparseGqaLoad::InPlace, true, plan.splits};
    // In-place expansion adds barriers but retains three CTAs/SM. Choose it only when
    // that residency removes a wave; otherwise use a separate packed staging slot.
    const auto blocks  = std::int64_t(queries) * plan.splits;
    const auto wave3   = std::int64_t(3) * sms;
    const auto wave2   = std::int64_t(2) * sms;
    const bool inplace = (blocks + wave3 - 1) / wave3 < (blocks + wave2 - 1) / wave2;
    return {inplace ? SparseGqaLoad::InPlace : SparseGqaLoad::Staged, true, plan.splits};
}

} // namespace ninfer::ops::detail::qsa
