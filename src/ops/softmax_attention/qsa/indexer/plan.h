#pragma once

#include "core/layout.h"
#include "ops/softmax_attention/qsa/indexer/arguments.h"
#include "ninfer/ops/qsa.h"

#include <algorithm>

namespace ninfer::ops::detail::qsa {

inline constexpr int kIndexerPartition = 4096;

struct IndexerPlan {
    int window_tokens, request_rows, columns, partitions;
};

int indexer_window_capacity(int batch);

inline IndexerPlan select_indexer_plan(int tokens, int batch, std::uint32_t visible_tokens) {
    const int capacity = indexer_window_capacity(batch);
    // Retain tuned small batches. Larger calls give each request a whole query window
    // where possible, reusing its keys before the next group uses the same score storage.
    const int requests =
        batch <= 8 ? batch : std::min(batch, capacity / std::min(tokens, capacity));
    const int blocks = visible_tokens / 4;
    return {std::min(tokens, capacity / requests), requests,
            (blocks + kIndexerKeyTile - 1) / kIndexerKeyTile * kIndexerKeyTile,
            (blocks + kIndexerPartition - 1) / kIndexerPartition};
}

enum class IndexerScore { Mma, Simt16, Simt8, Tma };

struct IndexerWindowPlan {
    int query_tile, score_workers, select_workers;
    int stages             = 1;
    IndexerScore score     = IndexerScore::Mma;
    bool reverse_selection = false;
    bool retain_keys       = false;
};

IndexerWindowPlan select_indexer_window(IndexerPlan plan, int width, int batch, int sms,
                                        bool tma_cache);

template <class Launch>
void for_each_indexer_window(IndexerArguments input, IndexerPlan plan, Launch&& launch) {
    for (std::int64_t first = 0; first < input.batch; first += plan.request_rows) {
        auto a            = input;
        const auto column = first * input.tokens;
        a.query += column * 512;
        a.positions += column;
        a.valid += first;
        a.table_rows += first;
        a.ids += column * kIndexerSelected;
        a.counts += column;
        a.batch = std::min<std::int64_t>(plan.request_rows, input.batch - first);
        for (std::int64_t begin = 0; begin < a.tokens; begin += plan.window_tokens) {
            a.begin = int(begin);
            a.width = std::min(plan.window_tokens, a.tokens - a.begin);
            launch(a);
        }
    }
}

void launch_indexer_windows(IndexerArguments input, IndexerPlan plan, std::size_t key_bytes,
                            std::size_t key_alignment, CudaExecutionView execution);

struct IndexerWorkspace {
    Tensor scores, candidates;
};

template <class Allocator>
IndexerWorkspace allocate_indexer_workspace(Allocator& allocator, IndexerPlan plan) {
    IndexerWorkspace out;
    if (plan.columns <= kIndexerSelected) return out;
    const int rows = plan.window_tokens * plan.request_rows;
    out.scores     = allocator.alloc(DType::FP32, {plan.columns, rows}, 256);
    out.candidates = allocator.alloc(DType::I64, {kIndexerSelected, plan.partitions, rows}, 256);
    return out;
}

void launch_indexer(const Tensor& query, const ConstQsaIndexCacheView& cache,
                    const QsaBatchView& batch, IndexerPlan plan, const IndexerWorkspace& scratch,
                    QsaBlockSelection& out, CudaExecutionView execution);

} // namespace ninfer::ops::detail::qsa
