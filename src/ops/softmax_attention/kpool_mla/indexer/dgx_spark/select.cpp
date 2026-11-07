#include "ops/softmax_attention/kpool_mla/indexer/plan.h"
#include "ops/softmax_attention/kpool_mla/validation.h"

namespace ninfer::ops::detail::kpool {
void indexer_geometry(const KpoolIndexerProfile& p, KvCacheVectorFormat format,
                      KpoolExecutionEnvelope envelope, int batch, int lo, int hi, int sms) {
    require(p.index_heads == 32 && p.index_dim == 128 && p.pool_size == 4 &&
                p.selected_pools == 512,
            "unsupported Indexer profile");
    require(format == KvCacheVectorFormat::BFloat16 ||
                format == KvCacheVectorFormat::Fp8E4m3fnGroup128Ue8m0,
            "unsupported Indexer format");
    require(lo > 0 && hi >= lo && batch >= 1 && batch <= 8 && sms > 0, "Indexer T/B/device");
    require(envelope.max_visible_tokens <= kKpoolMaxVisibleTokens, "Indexer context limit");
}

IndexerWindows indexer_windows(int lo, int hi, int batch, int columns) {
    // Cap windows around a 32 MiB score working set. Long histories retain
    // at least sixteen queries per request: dividing that reuse by batch would
    // repeatedly stream every request's keys. The larger score set scales with B,
    // and balanced windows avoid a short tail scanning alone. Graph slots remain
    // fixed independently of this execution-window choice.
    constexpr int columns_per_slot = (16 << 20) / (4 * kIndexGraphRows);
    int slots                      = 8;
    while (slots > 1 && columns > columns_per_slot / slots) slots /= 2;
    const int target = 2 * std::max(16, slots * (kIndexGraphRows / batch));
    // Cap windows at the old route's maximum width (plus one row), then balance
    // query tiles. A nearly-full window no longer changes every other window's
    // tile count or persistent-worker occupancy at T+1. The interval peak is at
    // its first complete window bucket, or hi when that bucket is not reached.
    const auto bucket = (std::int64_t(lo) + target - 1) / target;
    const auto peak   = std::min<std::int64_t>(hi, bucket * target);
    const int windows = int((peak + target - 1) / target);
    return {windows, indexer_window_width(int(peak), windows)};
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

bool separate_indexer_requests(const IndexerArguments& a) {
    // A long request's keys already occupy most or all of L2. Give its queries
    // the whole planned row budget instead of interleaving several such histories.
    // Short query extents retain cross-request parallelism.
    return a.batch > 2 && a.tokens >= 64 && a.columns >= 65536;
}

Fp8IndexerWindows fp8_indexer_windows(int tokens, int batch, int columns) {
    // Scores remain FP32. Keep their window bounded independently of prepared Q.
    constexpr int score_bytes      = 16 << 20;
    constexpr int columns_per_slot = score_bytes / (4 * kIndexGraphRows);
    int slots                      = 8;
    while (slots > 1 && columns > columns_per_slot / slots) slots /= 2;
    const int target  = 2 * std::max(16, slots * (kIndexGraphRows / batch));
    const int windows = (tokens + target - 1) / target;
    // Round the planning budget to complete tiles. A partial first window must
    // not reduce the number of windows abruptly when its last row becomes live.
    const int budget = (indexer_window_width(tokens, windows, 8) + 7) / 8 * 8;
    Fp8IndexerWindows result{batch, budget * batch, 0};
    // Group concurrent histories so their FP32 scores remain in L2 for selection.
    // The score row budget is shared by the requests in each group.
    if (batch > 1 && tokens >= 64 && columns >= 65536)
        result.group_size = 1;
    else if (batch > 2 && tokens >= 8) {
        if (columns >= 131072 || (columns >= 65536 && tokens > 32)) result.group_size = 1;
        // Small batches whose score rows already fit L2 retain cross-request
        // reuse, including shared prefixes. Larger batches also bound K pressure.
        else if (columns >= 65536 &&
                 (batch > 4 || std::int64_t(tokens) * batch * columns * 4 > score_bytes))
            result.group_size = 2;
        else if (batch > 4 && columns >= 32768)
            result.group_size = 4;
    }
    const int stride      = indexer_graph_stride(batch);
    const int graph_slots = (tokens + stride - 1) / stride;
    result.group_size     = std::max(result.group_size, (batch + graph_slots - 1) / graph_slots);
    for (int b = 0; b < batch; b += result.group_size) {
        const int take  = std::min(result.group_size, batch - b);
        const int count = result.count(tokens, take);
        result.rows     = std::max(result.rows, indexer_window_width(tokens, count, 8) * take);
    }
    return result;
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
