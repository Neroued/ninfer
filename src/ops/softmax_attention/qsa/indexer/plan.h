#pragma once

#include "core/layout.h"
#include "ops/softmax_attention/qsa/indexer/arguments.h"
#include "ninfer/ops/qsa.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <span>

namespace ninfer::ops::detail::qsa {

inline constexpr int kIndexerPartition = 4096;

struct IndexerPlan {
    int window_tokens, window_queries, columns, partitions;
};

int indexer_window_capacity(int batch);

inline IndexerPlan select_indexer_plan(int tokens, int batch, ContextBucket bucket) {
    const int window_tokens = std::max(1, indexer_window_capacity(batch) / batch);
    const int columns       = context_bucket_tokens(bucket) / 4;
    const int partitions =
        columns <= kIndexerSelected ? 0 : std::max(1, columns / kIndexerPartition);
    return {window_tokens, std::min(tokens, window_tokens) * batch, columns, partitions};
}

inline int indexer_workers(int waves, int sms, std::int64_t tasks) {
    return static_cast<int>(std::clamp<std::int64_t>(std::min(std::int64_t(waves) * sms, tasks), 1,
                                                     std::numeric_limits<int>::max()));
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

template <class Arguments, class Launch>
void for_each_indexer_window(Arguments input, IndexerPlan plan, Launch&& launch) {
    for (std::int64_t begin = 0; begin < input.tokens; begin += plan.window_tokens) {
        input.begin = int(begin);
        input.width = std::min(plan.window_tokens, input.tokens - input.begin);
        launch(input);
    }
}

// Segment membership is fixed by the validated host partition. A whole window
// inside one segment can use a rectangular score/select chain with live Device
// positions, lengths and table selectors. The scratch indices still start at zero.
inline std::optional<IndexerArguments>
single_segment_indexer_window(const RaggedIndexerArguments& a,
                              std::span<const std::int32_t> offsets) {
    if (a.partitions <= 1) return std::nullopt;
    const auto after = std::upper_bound(offsets.begin(), offsets.end(), a.begin);
    if (a.width > *after - a.begin) return std::nullopt;
    const auto segment       = after - offsets.begin() - 1;
    const std::int64_t first = offsets[segment];
    IndexerArguments row     = a;
    row.query += first * 512;
    row.positions += first;
    row.valid += segment;
    row.table_rows += segment;
    row.ids += first * kIndexerSelected;
    row.counts += first;
    row.tokens = *after - first;
    row.begin -= first;
    return row;
}

template <class Arguments>
void launch_indexer_windows(Arguments input, IndexerPlan plan, std::size_t key_bytes,
                            std::size_t key_alignment, CudaExecutionView execution,
                            std::span<const std::int32_t> offsets = {});

struct IndexerWorkspace {
    Tensor scores, candidates;
};

template <class Allocator>
IndexerWorkspace allocate_indexer_workspace(Allocator& allocator, IndexerPlan plan) {
    IndexerWorkspace out;
    if (plan.partitions == 0) return out;
    out.scores = allocator.alloc(DType::FP32, {plan.columns, plan.window_queries}, 256);
    if (plan.partitions > 1)
        out.candidates = allocator.alloc(
            DType::I64, {kIndexerSelected, plan.partitions, plan.window_queries}, 256);
    return out;
}

void launch_indexer(const Tensor& query, const ConstQsaIndexCacheView& cache,
                    const QsaBatchView& batch, IndexerPlan plan, const IndexerWorkspace& scratch,
                    QsaBlockSelection& out, CudaExecutionView execution,
                    const RaggedSequenceView* ragged = nullptr);

} // namespace ninfer::ops::detail::qsa
