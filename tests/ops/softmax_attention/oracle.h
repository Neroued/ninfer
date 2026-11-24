#pragma once

#include "ninfer/ops/attention_geometry.h"
#include "ops/suite.h"

#include <algorithm>
#include <array>
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
    const int group = geometry.query_heads / geometry.kv_heads;
    if (query_tokens == 0) return;

    // Batch heads that share the same represented K/V row. Each score still adds
    // features in ascending order, and each numerator/denominator adds keys in ascending order.
    const auto evaluate = [&]<int Heads>() {
        using HeadValues          = std::array<double, Heads>;
        const int tiles_per_kv    = group / Heads;
        const int tiles_per_query = geometry.kv_heads * tiles_per_kv;
        const int rows            = query_tokens * tiles_per_query;
        const auto run_rows       = [&](int first, int last) {
            std::vector<HeadValues> scores(key_tokens);
            std::vector<HeadValues> query_values(geometry.head_dim), numerators(geometry.head_dim);
            for (int row = first; row < last; ++row) {
                const int query      = row / tiles_per_query;
                const int kv_head    = (row % tiles_per_query) / tiles_per_kv;
                const int query_head = (row % tiles_per_query) * Heads;
                for (int d = 0; d < geometry.head_dim; ++d)
                    for (int h = 0; h < Heads; ++h)
                        query_values[d][h] = query_value(d, query_head + h, query);

                constexpr double masked = -std::numeric_limits<double>::infinity();
                HeadValues maximum;
                maximum.fill(masked);
                for (int key = 0; key < key_tokens; ++key) {
                    if (!visible(query, key)) {
                        scores[key].fill(masked);
                        continue;
                    }
                    HeadValues dot{};
                    for (int d = 0; d < geometry.head_dim; ++d) {
                        const double value = key_value(d, kv_head, key);
                        for (int h = 0; h < Heads; ++h) dot[h] += query_values[d][h] * value;
                    }
                    for (int h = 0; h < Heads; ++h) {
                        const double score = dot[h] * scale;
                        scores[key][h]     = score;
                        maximum[h]         = std::max(maximum[h], score);
                    }
                }

                HeadValues denominator{};
                for (auto& row_scores : scores)
                    for (int h = 0; h < Heads; ++h) {
                        double& score = row_scores[h];
                        if (maximum[h] == masked || score == masked) continue;
                        score = std::exp(score - maximum[h]);
                        denominator[h] += score;
                    }
                std::fill(numerators.begin(), numerators.end(), HeadValues{});
                for (int key = 0; key < key_tokens; ++key) {
                    const auto& weights = scores[key];
                    if (std::all_of(weights.begin(), weights.end(),
                                          [&](double weight) { return weight == masked; }))
                        continue;
                    for (int d = 0; d < geometry.head_dim; ++d) {
                        const double value = value_value(d, kv_head, key);
                        for (int h = 0; h < Heads; ++h)
                            if (weights[h] != masked) numerators[d][h] += weights[h] * value;
                    }
                }
                for (int d = 0; d < geometry.head_dim; ++d)
                    for (int h = 0; h < Heads; ++h)
                        store(d, query_head + h, query,
                              denominator[h] > 0.0 ? numerators[d][h] / denominator[h] : 0.0);
            }
        };
        if (parallel)
            parallel_rows(rows, std::uint64_t(rows) * Heads * key_tokens * geometry.head_dim,
                          run_rows);
        else
            run_rows(0, rows);
    };
    if (group % 4 == 0 && key_tokens >= 64)
        evaluate.template operator()<4>();
    else
        evaluate.template operator()<1>();
}

} // namespace ninfer::test
