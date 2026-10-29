#include "ops/softmax_attention/qsa/sparse_gqa/plan.h"
#include "ops/softmax_attention/qsa/sparse_gqa/rtx5090/compact_schedule.h"
#include "core/paged_kv_cache.h"
#include "ops/common/host/operands.h"
#include <limits>

namespace ninfer::ops::detail::qsa {
namespace {
constexpr int kSplitQueries                = 4096;
constexpr std::uint64_t kSplitCacheBytes   = 64ULL << 20;
constexpr std::uint32_t kMaxSelectedTokens = 512 * 4 + 3;

std::uint32_t two_head_prefix_limit(KvCacheStorage storage, std::int64_t queries, int sms) {
    if (storage == KvCacheStorage::BFloat16) {
        if (queries + 1 >= std::int64_t(sms) * 3 / 2) return kMaxSelectedTokens;
        return std::uint32_t(
            std::min<std::int64_t>(kMaxSelectedTokens, std::max<std::int64_t>(128, queries)));
    }
    const int reuse = storage == KvCacheStorage::Nvfp4Group16 ? 11 : 12;
    return std::uint32_t(std::min<std::int64_t>(kMaxSelectedTokens, queries * reuse));
}

int native_natural_partitions(std::int64_t queries) {
    return queries <= 6 ? 64 : queries <= 12 ? 33 : queries == 13 ? 32 : 22;
}

double partial_merge_rate(KvCacheStorage storage, int kv_heads, std::int64_t queries, int batch,
                          int sms, int resident, bool large, bool long_envelope) {
    const bool native = storage == KvCacheStorage::BFloat16;
    const bool fp4    = storage == KvCacheStorage::Nvfp4Group16;
    const bool k8v4   = storage == KvCacheStorage::Fp8KeyNvfp4Value;
    if (kv_heads == 4) {
        if (large) return fp4 ? 3. : 4.3;
        if (!native && queries > std::int64_t(sms) * 3) return fp4 ? 1.64 : 1.66;
        return .05;
    }
    if (queries >= std::int64_t(sms) * 4) {
        if (!long_envelope || fp4) return large ? 2. : native ? .6 : 1.2;
        if (large) return k8v4 ? 2.84 : 3.;
        if (native) return queries >= std::int64_t(sms) * 8 ? 1.5 : .6;
        if (queries >= std::int64_t(sms) * 8) {
            if (storage == KvCacheStorage::Int8Group64) return 2.245;
            if (storage == KvCacheStorage::Fp8E4M3Row256) return 2.37;
        }
        return 2.5;
    }
    if (!long_envelope || fp4) return large ? .8 : storage == KvCacheStorage::Int8Group64 ? .6 : .5;
    if (large) return k8v4 ? (batch == 2 ? 2.3 : .8) : 2.5;
    if (native) return .635;
    return queries * kv_heads >= std::int64_t(resident) * sms ? 2. : .5;
}

struct MmaProblem {
    KvCacheStorage storage;
    int kv_heads, batch, sms;
    std::int64_t queries;
    bool large, long_envelope;
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
        case KvCacheStorage::Fp8E4M3Row256:
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
            tile = p.long_envelope ? 1.63 : 1.65;
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
                               p.long_envelope),
            native_reuse};
}

SparseGqaPlan partition_mma(KvCacheStorage storage, int kv_heads, std::int64_t queries, int batch,
                            QsaSparseGqaExecutionEnvelope envelope, int sms, SparseGqaPlan plan) {
    if (plan.instance != SparseGqaInstance::Mma || kv_heads == 1) return plan;
    if (kv_heads == 4 && envelope.max_visible_tokens <= kMaxSelectedTokens) return plan;
    const bool native        = storage == KvCacheStorage::BFloat16;
    const bool fp4           = storage == KvCacheStorage::Nvfp4Group16;
    const bool long_envelope = envelope.max_visible_tokens > kMaxSelectedTokens;
    const auto row_bytes = paged_kv_storage_layout(storage, 256).physical_bytes_per_token_head();
    // Leave space for Q and partial rows. The envelope bounds the regime without
    // assuming that Device selection visits the entire physical cache.
    const bool large = std::uint64_t(envelope.max_visible_tokens) * batch * kv_heads * row_bytes >
                       kSplitCacheBytes;
    const MmaProblem problem{storage, kv_heads, batch, sms, queries, large, long_envelope};
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
        const int tiles = std::max(
            1, (int(std::min(envelope.max_visible_tokens, kMaxSelectedTokens)) + rows - 1) / rows);
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
            if (reuse && (native || (kv_heads == 2 && (rows != 32 || large)) ||
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
                    const auto blocks = queries * kv_heads * count;
                    const auto waves = (blocks + model.resident * sms - 1) / (model.resident * sms);
                    const bool partial = count > 1;
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

int sparse_gqa_direct_queries(int kv_heads, int) { return kv_heads == 1 ? 256 : kSplitQueries + 1; }

SparseGqaCapacityEnvelopes sparse_gqa_capacity_envelopes(KvCacheStorage storage, int tokens,
                                                         int batch, int kv_heads,
                                                         QsaSparseGqaExecutionEnvelope envelope,
                                                         int sms) {
    SparseGqaCapacityEnvelopes out;
    const auto add = [&](std::uint32_t bound) {
        if (bound <= envelope.max_visible_tokens) out.values[out.count++] = bound;
    };
    if (kv_heads != 1) {
        add(0);
        if (std::int64_t(tokens) * batch < 256) {
            for (std::uint32_t bound = 1; bound <= kMaxSelectedTokens; bound += 16) add(bound);
            add(kMaxSelectedTokens);
            const auto queries = std::int64_t(tokens) * batch;
            if (kv_heads == 2 && queries >= 32) {
                // The prefix/MMA crossover can occur inside a key-tile interval.
                // Include its first MMA envelope before the next tile boundary.
                const auto prefix_limit = two_head_prefix_limit(storage, queries, sms);
                if (prefix_limit < kMaxSelectedTokens) add(prefix_limit + 1);
            }
        }
        add(kMaxSelectedTokens + 1);
        const auto bytes = std::uint64_t(batch) * kv_heads *
                           paged_kv_storage_layout(storage, 256).physical_bytes_per_token_head();
        add(std::uint32_t(kSplitCacheBytes / bytes + 1));
    }
    add(envelope.max_visible_tokens);
    return out;
}

namespace {
SparseGqaPlan select_plan(KvCacheStorage storage, int tokens, int batch, int kv_heads,
                          QsaSparseGqaExecutionEnvelope envelope, int sms) {
    const auto queries = std::int64_t(tokens) * batch;
    const auto visible = envelope.max_visible_tokens;
    if (kv_heads == 4) {
        const bool bf16 = storage == KvCacheStorage::BFloat16;
        const bool fp4_value =
            storage == KvCacheStorage::Nvfp4Group16 || storage == KvCacheStorage::Fp8KeyNvfp4Value;
        const bool partial = queries < 256;
        // The short-prefix crossover includes the complete output/gate stage. A
        // smaller tile avoids decoding padded key rows for a short causal window.
        const int short_queries = bf16 ? (visible <= 32 ? 24 : 16)
                                  : storage == KvCacheStorage::Nvfp4Group16
                                      ? (visible <= 32 ? 28 : 12)
                                  : fp4_value ? (visible <= 32 ? 28 : 20)
                                              : (visible <= 32 ? 32 : 20);
        if (visible <= 64 && queries < short_queries)
            return {32, 1, SparseGqaInstance::Simt, false};
        const bool short_prefix = queries >= short_queries;
        const bool prefix       = visible <= kMaxSelectedTokens &&
                            (bf16 || (queries >= 2 && (visible > 64 || short_prefix)));
        if (!prefix && visible <= 64) return {32, 1, SparseGqaInstance::Simt, false};
        const int rows = prefix && visible <= 32  ? 32
                         : prefix || queries <= 8 ? 64
                         : bf16                   ? 32
                                                  : 16;
        if (!prefix && visible > kMaxSelectedTokens)
            return partition_mma(storage, kv_heads, queries, batch, envelope, sms,
                                 {rows, 1, SparseGqaInstance::Mma, false});
        const int packet   = prefix ? rows / 16 : 1;
        const auto groups  = (queries + packet - 1) / packet * 4;
        const int resident = bf16 ? (rows == 32               ? (prefix ? 3 : 5)
                                     : prefix && queries <= 8 ? 1
                                                              : 2)
                                  : (rows == 32 ? 3 : 2);
        const int tiles =
            std::max(1, (int(std::min(visible, kMaxSelectedTokens)) + rows - 1) / rows);
        int splits = 1;
        if (partial && !(fp4_value && prefix && visible <= 256)) {
            // Compare complete waves, including the Q prologue. A fixed CTA target can
            // spill one partition into another wave and create a latency inversion.
            std::int64_t best = std::numeric_limits<std::int64_t>::max();
            for (int candidate = 1; candidate <= std::min(64, tiles); ++candidate) {
                const int count = std::min(candidate, tiles);
                const auto waves =
                    (std::int64_t(groups) * count + resident * sms - 1) / (resident * sms);
                const auto cost = waves * ((tiles + count - 1) / count + 2);
                if (cost < best) {
                    best   = cost;
                    splits = count;
                }
            }
        }
        return {rows, splits, prefix ? SparseGqaInstance::FullPrefix : SparseGqaInstance::Mma,
                partial &&
                    (visible <= 64 || storage == KvCacheStorage::Fp8E4M3Row256 || splits > 1)};
    }
    if (kv_heads == 2 && visible <= 64 && queries < 256) {
        const int crossover = storage == KvCacheStorage::Nvfp4Group16 ? (visible <= 32 ? 28 : 12)
                                                                      : (visible <= 32 ? 32 : 20);
        return {32, 1,
                queries >= crossover ? SparseGqaInstance::FullPrefix : SparseGqaInstance::Simt,
                false};
    }
    if (kv_heads == 2 && visible <= kMaxSelectedTokens && queries >= 32 &&
        visible <= two_head_prefix_limit(storage, queries, sms)) {
        const int rows =
            visible <= 128 || (storage == KvCacheStorage::BFloat16 && queries <= 1024) ? 32 : 64;
        return {rows, 1, SparseGqaInstance::FullPrefix, false};
    }
    if (queries >= 256)
        return partition_mma(storage, kv_heads, queries, batch, envelope, sms,
                             visible <= kMaxSelectedTokens
                                 ? SparseGqaPlan{64, 1, SparseGqaInstance::FullPrefix, false}
                                 : SparseGqaPlan{32, 1, SparseGqaInstance::Mma, false});
    // Budget parallel partitions independently of history. The envelope only bounds the
    // selected support and admits a short-context arithmetic schedule.
    const auto target =
        std::max<std::int64_t>(1, (std::int64_t(sms) * 3 / 4 + queries - 1) / queries);
    int splits = 1;
    while (splits < target && splits < 32) splits *= 2;
    const bool simt = visible <= 64;
    if (simt) splits = std::max(1, splits / 12);
    const int tiles = std::max(1, (int(std::min(visible, kMaxSelectedTokens)) + 31) / 32);
    return partition_mma(storage, kv_heads, queries, batch, envelope, sms,
                         {32, std::min(splits, tiles),
                          simt ? SparseGqaInstance::Simt : SparseGqaInstance::Mma, true});
}

} // namespace

SparseGqaPlan select_sparse_gqa_plan(KvCacheStorage storage, int tokens, int batch, int kv_heads,
                                     QsaSparseGqaExecutionEnvelope envelope, int sms) {
    auto plan           = select_plan(storage, tokens, batch, kv_heads, envelope, sms);
    const auto queries  = std::int64_t(tokens) * batch;
    plan.visible_tokens = envelope.max_visible_tokens;
    // Capacity has no view argument: reserve both aligned and natural-load schedules.
    // The launcher selects the actual count without allocating or repacking.
    if (storage == KvCacheStorage::BFloat16 && kv_heads == 2 && queries <= 16 &&
        plan.instance == SparseGqaInstance::Mma && plan.partial &&
        envelope.max_visible_tokens > kMaxSelectedTokens)
        plan.splits = std::max(plan.splits, native_natural_partitions(queries));
    // The public contract keeps Graph topology compatible across envelope updates.
    plan.merge_node = kv_heads == 2 ? queries <= kSplitQueries
                      : kv_heads == 4
                          ? (storage == KvCacheStorage::BFloat16 ? queries < std::int64_t(sms) * 4
                                                                 : queries <= kSplitQueries)
                          : plan.partial;
    return plan;
}

SparseGqaLaunchPlan select_sparse_gqa_launch(const ConstPagedKVBatchLayerView& cache,
                                             SparseGqaPlan plan, int queries, int batch, int sms) {
    const auto aligned = [](const ConstPagedPlaneView& t) {
        return common::operand::aligned(t.data(), 16) && t.row_stride_bytes() % 16 == 0 &&
               t.head_stride_bytes() % 16 == 0 && t.plane.tensor.region.alignment % 16 == 0;
    };
    const bool vector = aligned(cache.k_pages) && aligned(cache.v_pages);
    if (cache.num_kv_heads == 2 && plan.instance != SparseGqaInstance::Simt) {
        auto selected = plan;
        if (cache.storage == KvCacheStorage::BFloat16 && queries <= 16 && plan.partial &&
            plan.instance == SparseGqaInstance::Mma && plan.visible_tokens > kMaxSelectedTokens) {
            if (vector)
                selected = select_plan(cache.storage, queries / batch, batch, 2,
                                       {plan.visible_tokens}, sms);
            else {
                selected.splits        = native_natural_partitions(queries);
                selected.delay_softmax = true;
            }
        }
        return {cache.storage == KvCacheStorage::BFloat16 || plan.reuse_buffer
                    ? SparseGqaLoad::Reused
                    : SparseGqaLoad::Aliased,
                vector, selected.splits,
                selected.delay_softmax ? SparseGqaKernel::CompactLateSoftmax
                                       : SparseGqaKernel::Sequence};
    }
    if (cache.num_kv_heads == 4) {
        const auto load = plan.instance == SparseGqaInstance::Simt ? SparseGqaLoad::Direct
                          : cache.storage == KvCacheStorage::BFloat16
                              ? (plan.instance == SparseGqaInstance::FullPrefix &&
                                         (queries <= 8 || plan.key_rows == 32)
                                     ? SparseGqaLoad::Direct
                                     : SparseGqaLoad::Reused)
                          : plan.reuse_buffer ? SparseGqaLoad::Reused
                                              : SparseGqaLoad::Aliased;
        return {load, vector, plan.splits};
    }
    if (cache.storage == KvCacheStorage::BFloat16 || !vector ||
        plan.instance == SparseGqaInstance::Simt)
        return {SparseGqaLoad::Direct, vector, plan.splits};
    if (plan.instance == SparseGqaInstance::FullPrefix)
        return {SparseGqaLoad::InPlace, vector, plan.splits};
    // In-place expansion adds barriers but retains three CTAs/SM. Choose it only when
    // that residency removes a wave; otherwise use a separate packed staging slot.
    const auto blocks  = std::int64_t(queries) * cache.num_kv_heads * plan.splits;
    const bool inplace = (blocks + 3 * sms - 1) / (3 * sms) < (blocks + 2 * sms - 1) / (2 * sms);
    return {inplace ? SparseGqaLoad::InPlace : SparseGqaLoad::Staged, vector, plan.splits};
}

} // namespace ninfer::ops::detail::qsa
