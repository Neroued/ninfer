#include "ops/softmax_attention/qsa/indexer/plan.h"

namespace ninfer::ops::detail::qsa {
int indexer_window_capacity(int batch) { return batch <= 8 ? 64 : kIndexerMaxWindowQueries; }

IndexerWindowPlan select_indexer_window(IndexerPlan plan, int width, int batch, int sms, bool) {
    const int slots = width * batch;
    if (plan.partitions == 0) return {0, 0, indexer_workers(2, sms, slots)};
    const int tile = width <= 4 ? 4 : width <= 8 ? 8 : width <= 16 ? 16 : 32;
    const auto tasks =
        std::int64_t(batch) * ((width + tile - 1) / tile) * (plan.columns / kIndexerKeyTile);
    const int score_waves  = width <= 2 || width > 32 || batch > 8 ? 4 : 2;
    const int select_waves = width > 32 ? 4 : 2;
    return {tile, indexer_workers(score_waves, sms, tasks),
            indexer_workers(select_waves, sms, std::int64_t(slots) * plan.partitions)};
}

} // namespace ninfer::ops::detail::qsa
