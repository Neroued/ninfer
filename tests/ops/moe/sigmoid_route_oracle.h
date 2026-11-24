#pragma once
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"
#include "ops/suite.h"
#include <array>
#include <numeric>

namespace ninfer::test::sigmoid_route {
struct Reference {
    std::array<int, 8> ids;
    std::array<double, 8> coefficients;
};

// Independent represented-weight FP64 dot and stable real sigmoid; no production routing helpers.
inline Reference reference(const quantized_weight::PackedWeight& weight,
                           std::span<const std::uint16_t> input, std::span<const float> bias,
                           double factor, double epsilon) {
    std::vector<std::pair<int, double>> active;
    for (int k = 0; k < weight.weight.k; ++k) {
        const double v = bf16_to_f32(input[k]);
        if (v != 0) active.emplace_back(k, v);
    }
    std::array<double, 288> score{}, sigmoid{};
    parallel_rows(288, std::uint64_t(288) * active.size(), [&](int begin, int end) {
        for (int e = begin; e < end; ++e) {
            double logit = 0;
            for (auto [k, v] : active)
                logit += quantized_weight::logical_weight_fp64(weight, e, k) * v;
            sigmoid[e] =
                logit >= 0 ? 1 / (1 + std::exp(-logit)) : std::exp(logit) / (1 + std::exp(logit));
            score[e] = sigmoid[e] + bias[e];
        }
    });
    std::array<int, 288> ids;
    std::iota(ids.begin(), ids.end(), 0);
    std::partial_sort(ids.begin(), ids.begin() + 8, ids.end(), [&](int a, int b) {
        return score[a] > score[b] || (score[a] == score[b] && a < b);
    });
    Reference result;
    double denominator = epsilon;
    for (int j = 0; j < 8; ++j) {
        result.ids[j] = ids[j];
        denominator += sigmoid[ids[j]];
    }
    for (int j = 0; j < 8; ++j) result.coefficients[j] = factor * sigmoid[ids[j]] / denominator;
    return result;
}
} // namespace ninfer::test::sigmoid_route
