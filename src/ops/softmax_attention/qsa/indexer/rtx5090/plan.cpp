#include "ops/softmax_attention/qsa/indexer/plan.h"

namespace ninfer::ops::detail::qsa {
int select_indexer_window_tokens(int tokens, int batch) { return std::min(tokens, 64 / batch); }

IndexerWindowPlan select_indexer_window(IndexerPlan plan, int width, int batch, int sms) {
    const int tile         = width <= 4 ? 4 : width <= 8 ? 8 : width <= 16 ? 16 : 32;
    const int tasks        = batch * ((width + tile - 1) / tile) * (plan.columns / kIndexerKeyTile);
    const int score_waves  = width <= 2 || width > 32 ? 4 : 2;
    const int select_waves = width > 32 ? 4 : 2;
    return {tile, std::max(1, std::min(score_waves * sms, tasks)),
            std::max(1, std::min(select_waves * sms, width * batch * plan.partitions))};
}

} // namespace ninfer::ops::detail::qsa
