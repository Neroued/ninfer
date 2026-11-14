#include "ops/softmax_attention/qsa/indexer/plan.h"

namespace ninfer::ops::detail::qsa {
int indexer_window_capacity(int) { return kIndexerMaxWindowQueries; }

IndexerWindowPlan select_indexer_window(IndexerPlan plan, int width, int batch, int sms,
                                        bool tma_cache) {
    const int tile            = width <= 4    ? 4
                                : width <= 8  ? 8
                                : width <= 16 ? 16
                                : width <= 32 ? 32
                                : width <= 64 ? 64
                                              : 128;
    const bool batched_single = width == 1 && batch > 1 && plan.columns >= 2048;
    auto score                = IndexerScore::Mma;
    if (tma_cache && width == 4 && batch > 8 && plan.columns >= 2048)
        score = IndexerScore::Tma;
    else if (batched_single || (width == 1 && batch == 1 && plan.columns >= 16384))
        score = IndexerScore::Simt16;
    else if (batch == 1 && width == 2 && plan.columns >= 32768)
        score = IndexerScore::Simt8;
    const int stages       = score == IndexerScore::Tma || (tile >= 32 && plan.columns <= 8192) ||
                               (tile <= 16 && plan.columns >= 8192) || tile == 128
                                 ? 2
                                 : 1;
    const int resident     = score == IndexerScore::Tma      ? 2
                             : score == IndexerScore::Simt16 ? (batched_single ? 8 : 6)
                             : score == IndexerScore::Simt8  ? 12
                             : tile == 128                   ? 1
                             : stages == 2 || tile >= 32     ? 2
                             : tile == 16                    ? 4
                                                             : 5;
    const int rows         = score == IndexerScore::Mma || score == IndexerScore::Tma
                                 ? (width + tile - 1) / tile
                                 : width;
    const int tasks        = batch * rows * (plan.columns / kIndexerKeyTile);
    const int select_waves = width > 32 ? 8 : 4;
    // Retain reused keys where the wider score window competes with them for L2 capacity.
    const bool retain_keys = tile == 128 && plan.partitions >= 6 && plan.partitions <= 9;
    return {tile,
            std::max(1, std::min(resident * sms, tasks)),
            std::max(1, std::min(select_waves * sms, width * batch * plan.partitions)),
            stages,
            score,
            width * batch > 64,
            retain_keys};
}
} // namespace ninfer::ops::detail::qsa
