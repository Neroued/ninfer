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

int indexer_group_size(int tokens, int batch);

struct Fp8IndexerWindows {
    int group_size, row_budget, rows;

    int count(int tokens, int batch) const {
        const int width = row_budget / batch;
        return (tokens + width - 1) / width;
    }
};

Fp8IndexerWindows fp8_indexer_windows(int tokens, int batch, int columns);
int indexer_graph_slots(int tokens, int batch);
int fp8_indexer_graph_slots(int tokens, int batch);
int indexer_selection_items(int columns);

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
    auto p = indexer_dimensions(envelope);
    if (p.columns <= kIndexSelected) return p;
    int rows           = 0;
    const auto account = [&](int lo, int hi) {
        const int group   = indexer_group_size(hi, batch);
        const auto window = indexer_windows(lo, hi, group, p.columns);
        p.windows         = std::max(p.windows, window.count);
        rows              = std::max(rows, window.width * group);
        if (const int tail = batch % group)
            rows = std::max(rows, indexer_windows(lo, hi, tail, p.columns).width * tail);
    };
    for (int t = min_tokens; t <= std::min(max_tokens, 4); ++t) account(t, t);
    if (max_tokens > 4) account(std::max(min_tokens, 5), max_tokens);
    LayoutBuilder builder;
    indexer_scratch(p, rows, builder);
    return p;
}

inline IndexerPlan fp8_indexer_plan(int tokens, int batch, KpoolExecutionEnvelope envelope) {
    auto p        = indexer_dimensions(envelope);
    p.query_width = std::min(tokens, kIndexPreparedQueries);
    if (p.columns <= kIndexSelected) return p;
    const int group = indexer_group_size(tokens, batch);
    int rows        = 0;
    for (const int count : {group, batch % group}) {
        if (!count) continue;
        rows           = std::max(rows, fp8_indexer_windows(p.query_width, count, p.columns).rows);
        const int tail = tokens % kIndexPreparedQueries;
        if (tokens > kIndexPreparedQueries && tail)
            rows = std::max(rows, fp8_indexer_windows(tail, count, p.columns).rows);
    }
    LayoutBuilder builder;
    // These operands survive every score/select window inside the prepared window.
    p.query_fp8 = builder.add_tensor(DType::FP8_E4M3FN, {128, 32, p.query_width, group}, 256,
                                     "Kpool FP8 query");
    p.weighted_coefficients = builder.add_tensor(DType::FP32, {32, p.query_width, group}, 256,
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
void launch_indexer(IndexerArguments, int sms, cudaStream_t);
void launch_fp8_indexer(Fp8IndexerArguments, int sms, cudaStream_t);

} // namespace ninfer::ops::detail::kpool
