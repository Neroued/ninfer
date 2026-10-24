#include "ops/softmax_attention/kpool_mla/indexer/plan.h"
#include "ops/softmax_attention/kpool_mla/validation.h"

namespace ninfer::ops::detail::kpool {
void indexer_geometry(const KpoolIndexerProfile& p, KpoolExecutionEnvelope envelope, int batch,
                      int lo, int hi, int sms) {
    require(p.index_heads == 32 && p.index_dim == 128 && p.pool_size == 4 &&
                p.selected_pools == 512,
            "unsupported Indexer profile");
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
} // namespace ninfer::ops::detail::kpool
