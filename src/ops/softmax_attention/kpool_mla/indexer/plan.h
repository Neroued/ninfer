#pragma once

#include "ninfer/ops/kpool_mla.h"
#include "core/layout.h"
#include "ops/softmax_attention/kpool_mla/indexer/arguments.h"

#include <algorithm>

namespace ninfer::ops::detail::kpool {

void indexer_geometry(const KpoolIndexerProfile&, KpoolExecutionEnvelope, int batch, int min_tokens,
                      int max_tokens, int sms);

struct IndexerWindows {
    int count, width;
};

IndexerWindows indexer_windows(int min_tokens, int max_tokens, int batch, int columns);

struct IndexerPlan {
    int windows, columns, partitions, merge_groups;
    TensorRegion scores, candidates, merged;
    std::size_t bytes = 0;
};

// A singleton interval plans execution; a wider interval plans its peak storage.
inline IndexerPlan indexer_plan(int min_tokens, int max_tokens, int batch,
                                KpoolExecutionEnvelope envelope) {
    const int count = envelope.max_visible_tokens / 4;
    IndexerPlan p{};
    p.columns          = (count + 63) / 64 * 64;
    const auto windows = indexer_windows(min_tokens, max_tokens, batch, p.columns);
    p.windows          = windows.count;
    p.partitions       = (count + kIndexPartition - 1) / kIndexPartition;
    p.merge_groups     = (p.partitions + kIndexFanIn - 1) / kIndexFanIn;
    if (count <= kIndexSelected) return p;
    const int rows = windows.width * batch;
    LayoutBuilder builder;
    if (p.partitions > 1)
        p.candidates = builder.add_tensor(DType::I64, {kIndexSelected, p.partitions, rows}, 256,
                                          "Kpool Indexer candidates");
    {
        auto scope = builder.scope();
        p.scores = builder.add_tensor(DType::FP32, {p.columns, rows}, 256, "Kpool Indexer scores");
    }
    if (p.partitions > kIndexFanIn) {
        auto scope = builder.scope();
        p.merged   = builder.add_tensor(DType::I64, {kIndexSelected, p.merge_groups, rows}, 256,
                                        "Kpool Indexer merge");
    }
    p.bytes = builder.finish(1);
    return p;
}

struct IndexerSchedule {
    int queries, stages, score_workers, select_workers;
};

IndexerSchedule select_indexer(const IndexerArguments&, int sms);
bool separate_indexer_requests(const IndexerArguments&);
void launch_indexer(IndexerArguments, int windows, int sms, cudaStream_t);

} // namespace ninfer::ops::detail::kpool
