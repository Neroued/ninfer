#pragma once

#include "ninfer/ops/kpool_mla.h"
#include "core/layout.h"
#include "ops/softmax_attention/kpool_mla/indexer/arguments.h"

#include <algorithm>

namespace ninfer::ops::detail::kpool {

void indexer_geometry(const KpoolIndexerProfile&, KvCacheVectorFormat, int batch, int min_tokens,
                      int max_tokens, int sms);

struct IndexerWindow {
    int begin, width;
};

// Balance query tiles and put the single partial tile in the first, largest
// window. This keeps a short tail from selecting a smaller pipeline in isolation.
inline IndexerWindow indexer_window(int tokens, int windows, int slot, int tile = 4) {
    const auto tiles = (std::int64_t(tokens) + tile - 1) / tile;
    const auto width = tiles / windows, larger = tiles % windows;
    const int padding = int(tiles * tile - tokens);
    const auto begin =
        (slot * width + std::min<std::int64_t>(slot, larger)) * tile - (slot ? padding : 0);
    return {int(begin), int((width + (slot < larger)) * tile - (slot ? 0 : padding))};
}

inline int indexer_window_width(int tokens, int windows, int tile = 4) {
    const int first = indexer_window(tokens, windows, 0, tile).width;
    return windows == 1 ? first : std::max(first, indexer_window(tokens, windows, 1, tile).width);
}

inline constexpr int kIndexPreparedQueries    = 128;
inline constexpr int kIndexLongRequestQueries = 64;

int indexer_group_size(int tokens, int batch);

struct IndexerWindows {
    int group_size, row_budget, tile;

    int count(int tokens, int batch) const {
        if (!row_budget) return 1;
        const int width = std::min(kIndexMaxWindowQueries, row_budget / batch / tile * tile);
        return int((std::int64_t(tokens) + width - 1) / width);
    }
};

// FP8 retains the largest outer group's prepared queries in the scratch layout.
// BF16 passes zero prepared_batch.
IndexerWindows indexer_windows(int tokens, int batch, int columns, int prepared_batch = 0);
int indexer_graph_slots(int tokens, int width, int batch, int prepared_batch = 0);
int indexer_selection_items(int columns);

struct IndexerPlan {
    int columns, partitions, merge_groups;
    TensorRegion scores, candidates, merged;
    TensorRegion query_fp8, weighted_coefficients;
    int query_width   = 0;
    std::size_t bytes = 0;
};

inline IndexerPlan indexer_dimensions(ContextBucket bucket) {
    IndexerPlan p{};
    p.columns = context_bucket_tokens(bucket) / 4;
    if (p.columns <= kIndexSelected) return p;
    p.partitions   = (p.columns + kIndexPartition - 1) / kIndexPartition;
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

inline IndexerPlan indexer_plan(int tokens, int batch, ContextBucket bucket, bool fp8) {
    auto p        = indexer_dimensions(bucket);
    p.query_width = fp8 ? std::min(tokens, kIndexPreparedQueries) : tokens;
    if (p.columns <= kIndexSelected) return p;
    const int group = indexer_group_size(tokens, batch);
    int rows        = 0;
    for (const int outer : {group, batch % group}) {
        if (!outer) continue;
        const auto windows = indexer_windows(tokens, outer, p.columns, fp8 ? group : 0);
        for (const int take : {windows.group_size, outer % windows.group_size}) {
            if (!take) continue;
            const int tail = fp8 ? tokens % p.query_width : 0;
            for (const int width : {p.query_width, tail}) {
                if (!width) continue;
                const int window_width =
                    indexer_window_width(width, windows.count(width, take), windows.tile);
                rows = std::max(rows, window_width * take);
            }
        }
    }
    LayoutBuilder builder;
    if (fp8) {
        // Prepared operands survive all score/select windows in their chunk.
        p.query_fp8 = builder.add_tensor(DType::FP8_E4M3FN, {128, 32, p.query_width, group}, 256,
                                         "Kpool FP8 query");
        p.weighted_coefficients = builder.add_tensor(DType::FP32, {32, p.query_width, group}, 256,
                                                     "Kpool scaled coefficients");
    }
    indexer_scratch(p, rows, builder);
    return p;
}

inline IndexerPlan packed_indexer_plan(int tokens, ContextBucket bucket, bool fp8) {
    auto p = indexer_dimensions(bucket);
    if (p.columns <= kIndexSelected) return p;
    const int limit = indexer_windows(kIndexMaxWindowQueries, 1, p.columns, fp8 ? 1 : 0).row_budget;
    p.query_width   = std::min(tokens, fp8 ? std::min(limit, kIndexPreparedQueries) : limit);
    LayoutBuilder builder;
    if (fp8) {
        p.query_fp8 =
            builder.add_tensor(DType::FP8_E4M3FN, {128, 32, p.query_width}, 256, "Kpool FP8 query");
        p.weighted_coefficients =
            builder.add_tensor(DType::FP32, {32, p.query_width}, 256, "Kpool scaled coefficients");
    }
    indexer_scratch(p, p.query_width, builder);
    return p;
}

inline std::size_t indexer_capacity(int lo, int hi, int batch, ContextBucket bucket, bool fp8) {
    if (bucket == ContextBucket::K2) return 0;
    // BF16 reaches every group's maximum window width within one full window
    // after the decode range. FP8 repeats with the residue after a full chunk.
    const auto limit =
        fp8 ? std::max(lo, kIndexPreparedQueries) + std::int64_t(kIndexPreparedQueries) - 1
            : std::max(lo, kIndexLongRequestQueries) + std::int64_t(kIndexMaxWindowQueries) - 1;
    const auto end     = std::min<std::int64_t>(hi, limit);
    std::size_t result = 0;
    for (std::int64_t t = lo; t <= end; ++t)
        result = std::max(result, indexer_plan(int(t), batch, bucket, fp8).bytes);
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
void launch_indexer(IndexerArguments, int sms, cudaStream_t);
void launch_fp8_indexer(Fp8IndexerArguments, int sms, cudaStream_t);

void launch_indexer_segment(IndexerArguments, int sms, cudaStream_t);
void launch_fp8_indexer_segment(Fp8IndexerArguments, int sms, cudaStream_t);
void launch_packed_fp8_indexer(Fp8IndexerArguments, const RaggedSequenceView&, int window, int sms,
                               cudaStream_t);
void launch_packed_indexer(IndexerArguments, const RaggedSequenceView&, int window, int sms,
                           cudaStream_t);

} // namespace ninfer::ops::detail::kpool
