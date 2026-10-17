#include "ops/softmax_attention/qsa/indexer/plan.h"

namespace ninfer::ops::detail::qsa {
IndexerWindowPlan select_indexer_window(IndexerPlan plan, int width, int batch, int sms) {
    const int tile  = width <= 4 ? 4 : width <= 8 ? 8 : width <= 16 ? 16 : width <= 32 ? 32 : 64;
    const bool simt = width * batch == 1 && plan.columns >= 16384;
    // Persistent Q fragments leave the shared-memory budget to K. Medium contexts
    // benefit from two K buffers; long scans favor the single-buffer worker count.
    const bool multi_row_scan = batch > 1 && tile <= 16 && plan.columns >= 8192;
    const int stages          = multi_row_scan || (tile >= 32 && plan.columns <= 8192) ? 2 : 1;
    const int resident        = simt             ? 6
                                : multi_row_scan ? 2
                                : tile == 64     ? 1
                                : tile == 32     ? 2
                                : tile == 16     ? 4
                                                 : 5;
    const int tasks        = batch * ((width + tile - 1) / tile) * (plan.columns / kIndexerKeyTile);
    const int select_waves = width > 32 ? 4 : 2;
    return {tile,
            std::max(1, std::min(resident * sms, tasks)),
            std::max(1, std::min(select_waves * sms, width * batch * plan.partitions)),
            stages,
            simt,
            plan.columns > 2048 && (plan.columns <= 8192 || width * batch <= 8)};
}
} // namespace ninfer::ops::detail::qsa
