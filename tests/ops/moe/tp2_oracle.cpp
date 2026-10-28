#include "ops/moe/tp2_fixture.h"
#include "ops/quantized_weight.h"
#include "ops/suite.h"

#include <array>
#include <cmath>
#include <map>
#include <numeric>

namespace ninfer::test::moe_tp2 {
namespace {
namespace codec = quantized_weight::detail;

std::vector<double> decode(const Fixture::MatrixSnapshot& m) {
    std::vector<double> values(std::size_t(m.n) * m.k);
    parallel_rows(m.n, values.size(), [&](int begin, int end) {
        for (int row = begin; row < end; ++row)
            for (int k = 0; k < m.k; ++k) {
                const auto index = std::size_t(row) * m.k + k;
                double value;
                if (m.format == QType::BF16) {
                    value = codec::bf16_to_f32(codec::load_u16_le(m.codes, index * 2));
                } else if (m.format == QType::Q8_G32_FP16) {
                    value = std::int8_t(m.codes[index]) *
                            double(codec::f16_to_f32(codec::load_u16_le(
                                m.scales, (std::size_t(row) * (m.k / 32) + k / 32) * 2)));
                } else if (m.format == QType::NVFP4) {
                    const auto code   = (m.codes[index / 2] >> (4 * (k & 1))) & 15;
                    const int group   = k / 16;
                    const auto offset = (row / 128 * (m.k / 64) + group / 4) * 512 + row % 32 * 16 +
                                        (row % 128) / 32 * 4 + group % 4;
                    value = codec::decode_e2m1(code) * codec::decode_e4m3fn(m.scales[offset]) /
                            double(m.divisor);
                } else {
                    const auto offset = ((row / m.block) * (m.k / m.block) + k / m.block) * 2;
                    value             = codec::decode_e4m3fn(m.codes[index]) *
                            double(codec::bf16_to_f32(codec::load_u16_le(m.scales, offset)));
                }
                values[index] = value;
            }
    });
    return values;
}

double sigmoid(double x) {
    return x >= 0 ? 1 / (1 + std::exp(-x)) : std::exp(x) / (1 + std::exp(x));
}
} // namespace

struct Oracle::Impl {
    const Fixture& fixture;
    std::vector<std::uint16_t> control;
    std::vector<float> bias;
    std::map<std::pair<int, bool>, std::vector<double>> matrices;
    std::map<std::pair<int, int>, std::vector<double>> products;
    std::map<int, std::pair<std::vector<int>, std::vector<double>>> selected;
    std::map<int, double> shared_coefficient;

    explicit Impl(const Fixture& f)
        : fixture(f), control(f.control_words()), bias(f.bias_words()) {}

    const std::vector<double>& matrix(int expert, bool gate) {
        const auto key = std::pair{expert, gate};
        auto found     = matrices.find(key);
        if (found == matrices.end())
            found = matrices
                        .emplace(key, decode(expert < 0 ? fixture.shared_snapshot(gate)
                                                        : fixture.snapshot(gate, expert)))
                        .first;
        return found->second;
    }

    const std::pair<std::vector<int>, std::vector<double>>& routing(int token) {
        if (auto found = selected.find(token); found != selected.end()) return found->second;
        const auto s      = fixture.shape;
        const auto& input = fixture.input_words();
        std::vector<double> logits(s.experts + !s.clamped), scores(s.experts);
        for (int row = 0; row < int(logits.size()); ++row)
            for (int k = 0; k < s.hidden; ++k)
                logits[row] +=
                    double(codec::bf16_to_f32(control[std::size_t(row) * s.hidden + k])) *
                    codec::bf16_to_f32(input[std::size_t(token) * s.hidden + k]);
        for (int e = 0; e < s.experts; ++e)
            scores[e] = s.clamped ? sigmoid(logits[e]) + bias[e] : logits[e];
        std::vector<int> ids(s.experts);
        std::iota(ids.begin(), ids.end(), 0);
        std::sort(ids.begin(), ids.end(), [&](int a, int b) {
            return scores[a] == scores[b] ? a < b : scores[a] > scores[b];
        });
        ids.resize(s.top_k);
        std::vector<double> coefficients(s.top_k);
        double denominator = s.clamped ? 1e-20 : 0;
        for (int j = 0; j < s.top_k; ++j) {
            coefficients[j] =
                s.clamped ? sigmoid(logits[ids[j]]) : std::exp(logits[ids[j]] - logits[ids[0]]);
            denominator += coefficients[j];
        }
        for (auto& c : coefficients) c *= (s.clamped ? 2.5 : 1) / denominator;
        shared_coefficient[token] = s.clamped ? 1 : sigmoid(logits.back());
        return selected.emplace(token, std::pair{std::move(ids), std::move(coefficients)})
            .first->second;
    }

    const std::vector<double>& product(int expert, int token) {
        const auto key = std::pair{expert, token};
        if (auto found = products.find(key); found != products.end()) return found->second;
        const auto s      = fixture.shape;
        const auto& gu    = matrix(expert, true);
        const auto& input = fixture.input_words();
        std::vector<double> result(s.intermediate);
        parallel_rows(s.intermediate, std::uint64_t(s.intermediate) * s.hidden * 2,
                      [&](int begin, int end) {
                          for (int row = begin; row < end; ++row) {
                              double gate = 0, up = 0;
                              for (int k = 0; k < s.hidden; ++k) {
                                  const double x =
                                      codec::bf16_to_f32(input[std::size_t(token) * s.hidden + k]);
                                  gate += gu[std::size_t(row) * s.hidden + k] * x;
                                  up += gu[std::size_t(row + s.intermediate) * s.hidden + k] * x;
                              }
                              if (s.clamped) {
                                  gate = std::min(gate, 10.0);
                                  up   = std::clamp(up, -10.0, 10.0);
                              }
                              result[row] = gate * sigmoid(gate) * up;
                          }
                      });
        return products.emplace(key, std::move(result)).first->second;
    }

    double expert_output(int expert, int token, int row) {
        const auto& values = product(expert, token);
        const auto& down   = matrix(expert, false);
        double result      = 0;
        for (int k = 0; k < fixture.shape.intermediate; ++k)
            result += down[std::size_t(row) * fixture.shape.intermediate + k] * values[k];
        return result;
    }
};

Oracle::Oracle(const Fixture& fixture) : impl_(std::make_unique<Impl>(fixture)) {}

Oracle::~Oracle() = default;

std::pair<std::vector<int>, std::vector<double>> Oracle::routing(int token) {
    return impl_->routing(token);
}

double Oracle::shared_coefficient(int token) {
    (void)impl_->routing(token);
    return impl_->shared_coefficient.at(token);
}

double Oracle::output(int row, int token, bool complete, bool signed_coefficients,
                      std::span<const int> expert_mask) {
    auto& f       = *impl_;
    const auto s  = f.fixture.shape;
    double result = 0;
    if (complete) {
        const auto& [ids, coefficients] = f.routing(token);
        for (int j = 0; j < s.top_k; ++j)
            if (expert_mask.empty() || expert_mask[ids[j]])
                result += coefficients[j] * f.expert_output(ids[j], token, row);
        result += f.shared_coefficient[token] * f.expert_output(-1, token, row);
    } else {
        for (int j = 0; j < s.top_k; ++j) {
            const int e    = f.fixture.expected_ids()[std::size_t(token) * s.top_k + j];
            const double c = (signed_coefficients && j % 3 == 0 ? -.375 : .75) * (1 + j * .125);
            if (expert_mask.empty() || expert_mask[e]) result += c * f.expert_output(e, token, row);
        }
    }
    return result;
}
} // namespace ninfer::test::moe_tp2
