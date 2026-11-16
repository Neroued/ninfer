#include "ops/softmax_attention/kpool_mla/indexer/plan.h"
#include "ops/softmax_attention/kpool_mla/validation.h"

namespace ninfer::ops::detail::kpool {
namespace {
constexpr int kScoreWindowBytes = 16 << 20;
constexpr int kRequestGroup     = 16;
} // namespace

void indexer_geometry(const KpoolIndexerProfile& p, KvCacheVectorFormat format,
                      KpoolExecutionEnvelope envelope, int batch, int lo, int hi, int sms) {
    require(p.index_heads == 32 && p.index_dim == 128 && p.pool_size == 4 &&
                p.selected_pools == 512,
            "unsupported Indexer profile");
    require(format == KvCacheVectorFormat::BFloat16 ||
                format == KvCacheVectorFormat::Fp8E4m3fnGroup128Ue8m0,
            "unsupported Indexer format");
    require(lo > 0 && hi >= lo && batch >= 1 && sms > 0, "Indexer T/B/device");
    require(std::int64_t(hi) * batch <= std::numeric_limits<int>::max(), "selection row extent");
    require(envelope.max_visible_tokens <= kKpoolMaxVisibleTokens, "Indexer context limit");
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

IndexerWindows indexer_windows(int lo, int hi, int batch, int columns) {
    // Bound scratch with request groups and query windows while preserving K reuse.
    // Increasing batch does not keep shrinking the per-request query target.
    // Graph slots are independent of this execution-window choice.
    constexpr int columns_per_slot = kScoreWindowBytes / (4 * kRequestGroup);
    int slots                      = 8;
    while (slots > 1 && columns > columns_per_slot / slots) slots /= 2;
    const int target = 2 * std::max(16, slots * (kRequestGroup / batch));
    // Balance query tiles so a partial tail does not change the other windows'
    // tile count or persistent-worker occupancy. The interval peak is at its first
    // complete window bucket, or hi when that bucket is not reached.
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
    constexpr int columns_per_slot = kScoreWindowBytes / (4 * kRequestGroup);
    int slots                      = 8;
    while (slots > 1 && columns > columns_per_slot / slots) slots /= 2;
    const int target  = 2 * std::max(16, slots * (kRequestGroup / batch));
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
                 (batch > 4 || std::int64_t(tokens) * batch * columns * 4 > kScoreWindowBytes))
            result.group_size = 2;
        else if (batch > 4 && columns >= 32768)
            result.group_size = 4;
    }
    if (result.group_size < batch) {
        // Keep enough query rows per group to amortize score and selection dispatch.
        const int query_stride = kRequestGroup / batch;
        const int max_groups   = (tokens + query_stride - 1) / query_stride;
        result.group_size      = std::max(result.group_size, (batch + max_groups - 1) / max_groups);
    }
    for (int b = 0; b < batch; b += result.group_size) {
        const int take  = std::min(result.group_size, batch - b);
        const int count = result.count(tokens, take);
        result.rows     = std::max(result.rows, indexer_window_width(tokens, count, 8) * take);
    }
    return result;
}

int indexer_graph_slots(int tokens, int batch) {
    // Every envelope admits at least 32 queries per request window. The largest
    // context also bounds the separately submitted long-request route.
    const int windows = int((std::int64_t(tokens) + 31) / 32);
    if (batch <= 2 || tokens < 64) return windows;
    const int rows = indexer_window_width(tokens, windows) * batch;
    return std::max(windows, batch * int((std::int64_t(tokens) + rows - 1) / rows));
}

int fp8_indexer_graph_slots(int tokens, int batch) {
    if (tokens < 8) return 1;
    int slots = 0;
    // Window budgets and request grouping change only at these padded-column
    // boundaries. Capture reserves their maximum independently of the envelope.
    const int pressure = int(kScoreWindowBytes / (4LL * tokens * batch) / 64 + 1) * 64;
    for (const int columns : {512, 32768, 32832, 65536, 65600, 131072, 131136, 262144, pressure}) {
        if (columns > kKpoolMaxVisibleTokens / 4) continue;
        const auto windows = fp8_indexer_windows(tokens, batch, columns);
        int count          = 0;
        for (int first = 0; first < batch; first += windows.group_size)
            count += windows.count(tokens, std::min(windows.group_size, batch - first));
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
