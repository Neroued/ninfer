#pragma once

#include "ninfer/ops/kpool_mla.h"
#include "core/layout.h"
#include "ops/softmax_attention/kpool_mla/indexer/arguments.h"

#include <algorithm>

namespace ninfer::ops::detail::kpool {

void indexer_geometry(const KpoolIndexerProfile&, KvCacheVectorFormat, KpoolExecutionEnvelope,
                      int batch, int min_tokens, int max_tokens, int sms);

struct IndexerWindows {
    int count, width;
};

struct IndexerWindow {
    int begin, width;
};

// Balance query tiles and put the single partial tile in the first, largest
// window. This keeps a short tail from selecting a smaller pipeline in isolation.
inline IndexerWindow indexer_window(int tokens, int windows, int slot, int tile = 4) {
    const auto tiles = (std::int64_t(tokens) + tile - 1) / tile;
    const auto width = tiles / windows, larger = tiles % windows;
    const int padding = int(tiles * tile - tokens);
    const int begin =
        int((slot * width + std::min<std::int64_t>(slot, larger)) * tile) - (slot ? padding : 0);
    return {begin, int((width + (slot < larger)) * tile) - (slot ? 0 : padding)};
}

inline int indexer_window_width(int tokens, int windows, int tile = 4) {
    const int first = indexer_window(tokens, windows, 0, tile).width;
    return windows == 1 ? first : std::max(first, indexer_window(tokens, windows, 1, tile).width);
}

IndexerWindows indexer_windows(int min_tokens, int max_tokens, int batch, int columns);
inline constexpr int kIndexPreparedQueries = 128;

inline int indexer_graph_stride(int batch) {
    // Single-request windows never need more than one score/select slot per
    // 32 queries. Batched decode retains its finer request-grouping budget.
    return batch == 1 ? 32 : kIndexGraphRows / batch;
}

struct Fp8IndexerWindows {
    int group_size, row_budget, rows;

    int count(int tokens, int batch) const {
        const int width = row_budget / batch;
        return (tokens + width - 1) / width;
    }
};

Fp8IndexerWindows fp8_indexer_windows(int tokens, int batch, int columns);

struct IndexerPlan {
    int windows, columns, partitions, merge_groups;
    TensorRegion scores, candidates, merged;
    TensorRegion query_fp8, weighted_coefficients;
    int query_width   = 0;
    std::size_t bytes = 0;
};

inline IndexerPlan indexer_dimensions(KpoolExecutionEnvelope envelope) {
    const int count = envelope.max_visible_tokens / 4;
    IndexerPlan p{};
    p.columns      = (count + 63) / 64 * 64;
    p.partitions   = (count + kIndexPartition - 1) / kIndexPartition;
    p.merge_groups = (p.partitions + kIndexFanIn - 1) / kIndexFanIn;
    return p;
}

inline void indexer_scratch(IndexerPlan& p, int rows, LayoutBuilder& builder) {
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
}

// A singleton interval plans execution; a wider interval plans its peak storage.
inline IndexerPlan indexer_plan(int min_tokens, int max_tokens, int batch,
                                KpoolExecutionEnvelope envelope) {
    auto p             = indexer_dimensions(envelope);
    const auto windows = indexer_windows(min_tokens, max_tokens, batch, p.columns);
    p.windows          = windows.count;
    if (p.columns <= kIndexSelected) return p;
    LayoutBuilder builder;
    indexer_scratch(p, windows.width * batch, builder);
    return p;
}

inline IndexerPlan fp8_indexer_plan(int tokens, int batch, KpoolExecutionEnvelope envelope) {
    auto p        = indexer_dimensions(envelope);
    p.query_width = std::min(tokens, kIndexPreparedQueries);
    if (p.columns <= kIndexSelected) return p;
    int rows       = fp8_indexer_windows(p.query_width, batch, p.columns).rows;
    const int tail = tokens % kIndexPreparedQueries;
    if (tokens > kIndexPreparedQueries && tail)
        rows = std::max(rows, fp8_indexer_windows(tail, batch, p.columns).rows);
    LayoutBuilder builder;
    // These operands survive every score/select window inside the prepared window.
    p.query_fp8 = builder.add_tensor(DType::FP8_E4M3FN, {128, 32, p.query_width, batch}, 256,
                                     "Kpool FP8 query");
    p.weighted_coefficients = builder.add_tensor(DType::FP32, {32, p.query_width, batch}, 256,
                                                 "Kpool scaled coefficients");
    indexer_scratch(p, rows, builder);
    return p;
}

inline std::size_t fp8_indexer_capacity(int lo, int hi, int batch,
                                        KpoolExecutionEnvelope envelope) {
    std::size_t result = 0;
    // Above one full prepared window, scratch repeats with the final window's residue.
    const auto end = std::min<std::int64_t>(hi, std::int64_t(lo) + 2 * kIndexPreparedQueries - 1);
    for (std::int64_t t = lo; t <= end; ++t)
        result = std::max(result, fp8_indexer_plan(int(t), batch, envelope).bytes);
    return result;
}

struct IndexerSchedule {
    int queries, stages, score_workers, select_workers;
    bool resident_selection;
};

IndexerSchedule select_indexer(const IndexerArguments&, int sms);
enum class Fp8IndexerRoute { Q1K64S2, Q2K64S2, Q4K32S1, Q4K32S2, Q4K64S3, Q8K16Prefetch };

struct Fp8IndexerConfig {
    int queries, keys, stages, blocks, warp_queries = 1;
    bool epilogue_prefetch = false;
};

constexpr Fp8IndexerConfig fp8_indexer_config(Fp8IndexerRoute route) {
    switch (route) {
    case Fp8IndexerRoute::Q1K64S2:
        return {1, 64, 2, 4};
    case Fp8IndexerRoute::Q2K64S2:
        return {2, 64, 2, 4};
    case Fp8IndexerRoute::Q4K32S1:
        return {4, 32, 1, 4};
    case Fp8IndexerRoute::Q4K32S2:
        return {4, 32, 2, 4};
    case Fp8IndexerRoute::Q4K64S3:
        return {4, 64, 3, 3};
    case Fp8IndexerRoute::Q8K16Prefetch:
        return {8, 16, 1, 3, 2, true};
    }
    return {};
}

struct Fp8IndexerSchedule {
    Fp8IndexerRoute route;
    int score_workers, select_workers;
    bool resident_selection;
};

Fp8IndexerSchedule select_fp8_indexer(const IndexerArguments&, int sms);
bool separate_indexer_requests(const IndexerArguments&);
void launch_indexer(IndexerArguments, int windows, int sms, cudaStream_t);
void launch_fp8_indexer(Fp8IndexerArguments, int sms, cudaStream_t);

} // namespace ninfer::ops::detail::kpool
