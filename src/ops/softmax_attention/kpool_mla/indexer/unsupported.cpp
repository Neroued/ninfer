#include "ops/softmax_attention/kpool_mla/indexer/plan.h"

#include <stdexcept>

namespace ninfer::ops::detail::kpool {
void indexer_geometry(const KpoolIndexerProfile&, KpoolExecutionEnvelope, int, int, int, int) {
    throw std::invalid_argument("Kpool Indexer: no implementation for this target");
}

void launch_indexer(IndexerArguments, int, int, cudaStream_t) {
    throw std::invalid_argument("Kpool Indexer: no implementation for this target");
}

IndexerWindows indexer_windows(int, int, int, int) {
    throw std::invalid_argument("Kpool Indexer: no implementation for this target");
}
} // namespace ninfer::ops::detail::kpool
