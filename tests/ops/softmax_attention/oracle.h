#pragma once

#include "ninfer/ops/attention_geometry.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ninfer::test {

// Independent logical Softmax Attention oracle. Callbacks expose represented public values and
// the entry-specific visible set; no production staging cast, tile, cache address, or reduction
// tree is reproduced here.
template <typename QueryValue, typename KeyValue, typename ValueValue, typename Visible,
          typename Store>
void naive_dense_softmax_attention(ops::AttentionHeadGeometry geometry, int query_tokens,
                                   int key_tokens, double scale, QueryValue query_value,
                                   KeyValue key_value, ValueValue value_value, Visible visible,
                                   Store store) {
    if (!ops::valid_attention_head_geometry(geometry) || query_tokens < 0 || key_tokens < 0) {
        throw std::invalid_argument("invalid naive Softmax Attention geometry");
    }
    const int group = geometry.query_heads / geometry.kv_heads;
    std::vector<double> scores(static_cast<std::size_t>(key_tokens));
    for (int query = 0; query < query_tokens; ++query) {
        for (int query_head = 0; query_head < geometry.query_heads; ++query_head) {
            const int kv_head = query_head / group;
            double maximum    = -std::numeric_limits<double>::infinity();
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
            for (int d = 0; d < geometry.head_dim; ++d) {
                double numerator = 0.0;
                for (int key = 0; key < key_tokens; ++key) {
                    const double weight = scores[static_cast<std::size_t>(key)];
                    if (weight == -std::numeric_limits<double>::infinity()) continue;
                    numerator += weight * value_value(d, kv_head, key);
                }
                store(d, query_head, query, denominator > 0.0 ? numerator / denominator : 0.0);
            }
        }
    }
}

} // namespace ninfer::test
