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
    // Choose windows around a 16 MiB score working set. Long histories retain
    // at least sixteen queries per request: dividing that reuse by batch would
    // repeatedly stream every request's keys. The larger score set scales with B,
    // and balanced windows avoid a short tail scanning alone. Graph slots remain
    // fixed independently of this execution-window choice.
    constexpr int columns_per_slot = (16 << 20) / (4 * kIndexGraphRows);
    int slots                      = 8;
    while (slots > 1 && columns > columns_per_slot / slots) slots /= 2;
    const int target = std::max(16, slots * (kIndexGraphRows / batch));
    // For n=floor(T/target), width=ceil(T/n). The interval maximum is at the end of
    // its first n>=1 bucket: later bucket maxima cannot grow. lo==hi is execution.
    const int bucket  = std::max(1, lo / target);
    const auto peak   = std::min<std::int64_t>(hi, (std::int64_t(bucket) + 1) * target - 1);
    const int windows = std::max(1, int(peak) / target);
    return {windows, int((peak + windows - 1) / windows)};
}

IndexerSchedule select_indexer(const IndexerArguments& a, int sms) {
    const int q      = a.width >= 4 ? 4 : a.width >= 2 ? 2 : 1;
    const int groups = ((a.width + q - 1) / q) * a.batch;
    // Round down: a few persistent CTAs beyond one resident wave would scan a full
    // history after the rest finish. Header publication loops over any extra rows.
    const int workers = std::max(1, 4 * sms / groups) * groups;
    const int stages  = a.columns <= 4096 ? 1 : 2;
    return {q, stages, workers, std::max(1, std::min(4 * sms, a.width * a.batch * a.partitions))};
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
    const int target  = std::max(16, slots * (kIndexGraphRows / batch));
    const int windows = std::max(1, tokens / target);
    Fp8IndexerWindows result{batch, (tokens + windows - 1) / windows * batch, 0};
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
    const int graph_slots = (tokens + kIndexGraphRows / batch - 1) / (kIndexGraphRows / batch);
    result.group_size     = std::max(result.group_size, (batch + graph_slots - 1) / graph_slots);
    for (int b = 0; b < batch; b += result.group_size) {
        const int take  = std::min(result.group_size, batch - b);
        const int count = result.count(tokens, take);
        result.rows     = std::max(result.rows, (tokens + count - 1) / count * take);
    }
    return result;
}

Fp8IndexerSchedule select_fp8_indexer(const IndexerArguments& a, int sms) {
    const int q = a.width >= 4 ? 4 : a.width >= 2 ? 2 : 1;
    auto route  = q == 1   ? Fp8IndexerRoute::Q1K64S2
                  : q == 2 ? Fp8IndexerRoute::Q2K64S2
                           : Fp8IndexerRoute::Q4K32S2;
    // Wide query windows with smaller histories favor the lower staging overhead.
    // Long streams and medium windows use a larger key tile with extra lookahead.
    if (a.width >= 32 && a.columns <= 65536) {
        // Two queries per warp reuse K fragments. Keep the four-query route
        // when Q8 would introduce another half tile, or history is too short
        // to amortize its extra registers and epilogue-prefetch barrier.
        route = a.columns >= 32768 && ((a.width + 3) / 4) % 2 == 0 ? Fp8IndexerRoute::Q8K16Prefetch
                                                                   : Fp8IndexerRoute::Q4K32S1;
    } else if (a.width >= 16 && a.columns >= 32768)
        route = Fp8IndexerRoute::Q4K64S3;
    const auto config = fp8_indexer_config(route);
    const int groups  = (a.width + config.queries - 1) / config.queries * a.batch;
    const int select  = std::max(1, std::min(4 * sms, a.width * a.batch * a.partitions));
    // The unconstrained radix kernel fits two CTAs per SM. Beyond that wave,
    // bound registers for four CTAs to hide the score loads and histogram barriers.
    return {route, std::max(1, config.blocks * sms / groups) * groups, select, select > 2 * sms};
}
} // namespace ninfer::ops::detail::kpool
