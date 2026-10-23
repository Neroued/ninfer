#pragma once

#include "core/layout.h"
#include "ops/softmax_attention/qsa/indexer/arguments.h"
#include "ninfer/ops/qsa.h"

#include <algorithm>

namespace ninfer::ops::detail::qsa {

inline constexpr int kIndexerPartition = 4096;

struct IndexerPlan {
    int window_tokens, columns, partitions;
};

int select_indexer_window_tokens(int tokens, int batch);

inline IndexerPlan select_indexer_plan(int tokens, int batch, std::uint32_t visible_tokens) {
    const int blocks = visible_tokens / 4;
    return {select_indexer_window_tokens(tokens, batch),
            (blocks + kIndexerKeyTile - 1) / kIndexerKeyTile * kIndexerKeyTile,
            (blocks + kIndexerPartition - 1) / kIndexerPartition};
}

enum class IndexerScore { Mma, Simt16, Simt8 };

struct IndexerWindowPlan {
    int query_tile, score_workers, select_workers;
    int stages             = 1;
    IndexerScore score     = IndexerScore::Mma;
    bool reverse_selection = false;
    bool retain_keys       = false;
};

IndexerWindowPlan select_indexer_window(IndexerPlan plan, int width, int batch, int sms);

void launch_indexer_window(IndexerArguments args, IndexerWindowPlan plan, bool vector_cache,
                           CudaExecutionView execution);

struct IndexerWorkspace {
    Tensor scores, candidates;
};

template <class Allocator>
IndexerWorkspace allocate_indexer_workspace(Allocator& allocator, IndexerPlan plan, int batch) {
    IndexerWorkspace out;
    if (plan.columns <= kIndexerSelected) return out;
    const int rows = plan.window_tokens * batch;
    out.scores     = allocator.alloc(DType::FP32, {plan.columns, rows}, 256);
    out.candidates = allocator.alloc(DType::I64, {kIndexerSelected, plan.partitions, rows}, 256);
    return out;
}

void launch_indexer(const Tensor& query, const ConstQsaIndexCacheView& cache,
                    const QsaBatchView& batch, IndexerPlan plan, const IndexerWorkspace& scratch,
                    QsaBlockSelection& out, CudaExecutionView execution);

} // namespace ninfer::ops::detail::qsa
