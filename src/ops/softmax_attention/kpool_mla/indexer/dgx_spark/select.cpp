#include "ops/softmax_attention/kpool_mla/indexer/plan.h"
#include "ops/softmax_attention/kpool_mla/validation.h"

namespace ninfer::ops::detail::kpool {
namespace {
constexpr int kScoreWindowBytes = 16 << 20;
constexpr int kRequestGroup     = 16;
} // namespace

void indexer_geometry(const KpoolIndexerProfile& p, KvCacheVectorFormat format, int batch, int lo,
                      int hi, int sms) {
    require(p.index_heads == 32 && p.index_dim == 128 && p.pool_size == 4 &&
                p.selected_pools == 512,
            "unsupported Indexer profile");
    require(format == KvCacheVectorFormat::BFloat16 ||
                format == KvCacheVectorFormat::Fp8E4m3fnGroup128Ue8m0,
            "unsupported Indexer format");
    require(lo > 0 && hi >= lo && batch >= 1 && sms > 0, "Indexer T/B/device");
    require(std::int64_t(hi) * batch <= std::numeric_limits<int>::max(), "selection row extent");
}

int indexer_group_size(int tokens, int batch) {
    // Decode groups expose enough rows for score and selection parallelism.
    // Wider windows retain bounded request groups and reuse each history's K.
    const int capacity = tokens <= 4 ? std::min(64, 128 / tokens) : kRequestGroup;
    return std::min(batch, capacity);
}

int indexer_selection_items(int columns) {
    return columns <= 1024 ? 4 : columns <= 2048 ? 8 : kIndexPartition / 256;
}

IndexerSchedule select_indexer(const IndexerArguments& a, int sms) {
    const int q      = a.width >= 3 ? 4 : a.width >= 2 ? 2 : 1;
    const int groups = ((a.width + q - 1) / q) * a.batch;
    // Round down: a few persistent CTAs beyond one resident wave would scan a full
    // history after the rest finish. Header publication loops over any extra rows.
    const int stages  = a.columns <= 4096 ? 1 : 2;
    const int blocks  = q < 4 && stages == 2 ? 3 : 4;
    const int workers = std::max(1, blocks * sms / groups) * groups;
    const int select  = std::max(1, std::min(4 * sms, a.width * a.batch * a.partitions));
    return {q, stages, workers, select, select > 2 * sms};
}

IndexerWindows indexer_windows(int tokens, int batch, int columns, int prepared_batch) {
    const bool fp8 = prepared_batch > 0;
    const int tile = fp8 ? 8 : 4;
    if (tokens <= 16 || columns <= kIndexSelected) return {batch, 0, tile};

    const int width = fp8 ? std::min(tokens, kIndexPreparedQueries) : tokens;
    int group_size  = batch;
    // Long histories benefit from reusing keys across a request's queries.
    if (width >= kIndexLongRequestQueries && columns >= 65536 && batch > (fp8 ? 1 : 2))
        group_size = 1;
    else if (batch > 2) {
        if (columns >= 131072 || (columns >= 65536 && width > 32))
            group_size = 1;
        else if (columns >= 65536)
            group_size = 2;
        else if (batch > 4 && columns >= 32768)
            group_size = 4;
    }

    constexpr int columns_per_slot = kScoreWindowBytes / (4 * kRequestGroup);
    int slots                      = 8;
    while (slots > 1 && columns > columns_per_slot / slots) slots /= 2;
    const int target =
        std::min(kIndexMaxWindowQueries,
                 (2 * std::max(16, slots * (kRequestGroup / batch)) + tile - 1) / tile * tile);
    return {group_size, target * group_size, tile};
}

int indexer_graph_slots(int tokens, int width, int batch, int prepared_batch) {
    if (tokens <= 16) return 1;
    int slots = 1;
    for (const auto bucket : kAllContextBuckets) {
        const auto windows =
            indexer_windows(tokens, batch, indexer_dimensions(bucket).columns, prepared_batch);
        const int groups = batch / windows.group_size;
        const int tail   = batch % windows.group_size;
        int count        = groups * windows.count(width, windows.group_size);
        if (tail) count += windows.count(width, tail);
        slots = std::max(slots, count);
    }
    return slots;
}

Fp8IndexerSchedule select_fp8_indexer(const IndexerArguments& a, int sms) {
    const int q = a.width >= 3 ? 4 : a.width >= 2 ? 2 : 1;
    auto route  = q == 1   ? Fp8IndexerRoute::Q1K64S2
                  : q == 2 ? Fp8IndexerRoute::Q2K64S2
                           : Fp8IndexerRoute::Q4K32S2;
    // Wide query windows with smaller histories favor the lower staging overhead.
    // Long streams and medium windows use a larger key tile with extra lookahead.
    if (a.width >= 32 && a.columns <= 65536) {
        // Two queries per warp reuse K fragments. Keep the four-query route
        // when Q8 would introduce another half tile, or history is too short
        // to amortize its extra registers and epilogue-prefetch barrier.
        const bool amortized_tail = a.batch == 1 && a.columns < 32768 && a.width >= 40;
        route                     = a.columns >= (a.batch == 1 ? 16384 : 32768) &&
                        (((a.width + 3) / 4) % 2 == 0 || amortized_tail)
                                        ? Fp8IndexerRoute::Q8K16Prefetch
                                        : Fp8IndexerRoute::Q4K32S1;
    } else if (a.width >= 16 && a.columns >= 32768)
        route = Fp8IndexerRoute::Q4K64S3;
    const auto config = fp8_indexer_config(route);
    const int groups  = (a.width + config.queries - 1) / config.queries * a.batch;
    const int select  = std::max(1, std::min(4 * sms, a.width * a.batch * a.partitions));
    // The unconstrained radix kernel fits two CTAs per SM. Beyond that wave,
    // bound registers for four CTAs to hide the score loads and histogram barriers.
    const bool balance =
        a.batch == 1 && a.columns < 32768 && route == Fp8IndexerRoute::Q8K16Prefetch;
    const int workers = balance ? std::max(config.blocks * sms, groups)
                                : std::max(1, config.blocks * sms / groups) * groups;
    return {route, workers, select, select > 2 * sms};
}
} // namespace ninfer::ops::detail::kpool
