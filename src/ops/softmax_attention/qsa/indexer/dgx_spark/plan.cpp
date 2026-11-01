#include "ops/softmax_attention/qsa/indexer/plan.h"

namespace ninfer::ops::detail::qsa {
int select_indexer_window_tokens(int tokens, int batch) {
    return std::min(tokens, kIndexerMaxWindowQueries / batch);
}

IndexerWindowPlan select_indexer_window(IndexerPlan plan, int width, int batch, int sms) {
    const int tile   = width <= 4    ? 4
                       : width <= 8  ? 8
                       : width <= 16 ? 16
                       : width <= 32 ? 32
                       : width <= 64 ? 64
                                     : 128;
    const auto score = width * batch == 1 && plan.columns >= 16384         ? IndexerScore::Simt16
                       : batch == 1 && width == 2 && plan.columns >= 32768 ? IndexerScore::Simt8
                                                                           : IndexerScore::Mma;
    const int stages =
        (tile >= 32 && plan.columns <= 8192) || (tile <= 16 && plan.columns >= 8192) || tile == 128
            ? 2
            : 1;
    const int resident     = score == IndexerScore::Simt16  ? 6
                             : score == IndexerScore::Simt8 ? 12
                             : tile == 128                  ? 1
                             : stages == 2 || tile >= 32    ? 2
                             : tile == 16                   ? 4
                                                            : 5;
    const int rows         = score == IndexerScore::Mma ? (width + tile - 1) / tile : width;
    const int tasks        = batch * rows * (plan.columns / kIndexerKeyTile);
    const int select_waves = width > 32 ? 8 : 2;
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
