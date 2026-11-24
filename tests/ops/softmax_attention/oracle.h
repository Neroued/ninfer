#pragma once

#include "ninfer/ops/attention_geometry.h"
#include "ops/suite.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ninfer::test {

// Independent logical Softmax Attention oracle. Callbacks expose represented public values and
// the entry-specific visible set; no production staging cast, tile, cache address, or reduction
// tree is reproduced here. Parallel workers own separate query/head rows; value callbacks must
// be read-only and store must write only the addressed output. The default remains serial for
// callers that already parallelize batches.
template <typename QueryValue, typename KeyValue, typename ValueValue, typename Visible,
          typename Store>
void naive_dense_softmax_attention(ops::AttentionHeadGeometry geometry, int query_tokens,
                                   int key_tokens, double scale, QueryValue query_value,
                                   KeyValue key_value, ValueValue value_value, Visible visible,
                                   Store store, bool parallel = false) {
    if (!ops::valid_attention_head_geometry(geometry) || query_tokens < 0 || key_tokens < 0) {
        throw std::invalid_argument("invalid naive Softmax Attention geometry");
    }
    const int group         = geometry.query_heads / geometry.kv_heads;
    const std::int64_t rows = static_cast<std::int64_t>(query_tokens) * geometry.query_heads;
    if (rows == 0) return;
    const auto run_rows = [&](std::int64_t first, std::int64_t last) {
        std::vector<double> scores(static_cast<std::size_t>(key_tokens));
        std::vector<double> numerators(static_cast<std::size_t>(geometry.head_dim));
        for (auto row = first; row < last; ++row) {
            const int query      = row / geometry.query_heads;
            const int query_head = row % geometry.query_heads;
            const int kv_head    = query_head / group;
            double maximum       = -std::numeric_limits<double>::infinity();
            for (int key = 0; key < key_tokens; ++key) {
                if (!visible(query, key)) {
                    scores[static_cast<std::size_t>(key)] =
                        -std::numeric_limits<double>::infinity();
                    continue;
                }
                double dot = 0.0;
                for (int d = 0; d < geometry.head_dim; ++d) {
                    dot += query_value(d, query_head, query) * key_value(d, kv_head, key);
                }
                const double score                    = dot * scale;
                scores[static_cast<std::size_t>(key)] = score;
                maximum                               = std::max(maximum, score);
            }

            double denominator = 0.0;
            if (maximum != -std::numeric_limits<double>::infinity()) {
                for (int key = 0; key < key_tokens; ++key) {
                    double& score = scores[static_cast<std::size_t>(key)];
                    if (score == -std::numeric_limits<double>::infinity()) continue;
                    score = std::exp(score - maximum);
                    denominator += score;
                }
            }
            // Visit each contiguous value row once. Each output still sums keys in ascending
            // order in FP64; only the interleaving of independent feature sums changes.
            std::fill(numerators.begin(), numerators.end(), 0.0);
            for (int key = 0; key < key_tokens; ++key) {
                const double weight = scores[static_cast<std::size_t>(key)];
                if (weight == -std::numeric_limits<double>::infinity()) continue;
                for (int d = 0; d < geometry.head_dim; ++d) {
                    numerators[static_cast<std::size_t>(d)] +=
                        weight * value_value(d, kv_head, key);
                }
            }
            for (int d = 0; d < geometry.head_dim; ++d) {
                store(d, query_head, query,
                      denominator > 0.0 ? numerators[static_cast<std::size_t>(d)] / denominator
                                        : 0.0);
            }
        }
    };
    if (!parallel) return run_rows(0, rows);
    parallel_rows(static_cast<int>(rows), std::uint64_t(rows) * key_tokens * geometry.head_dim,
                  run_rows);
}

} // namespace ninfer::test
