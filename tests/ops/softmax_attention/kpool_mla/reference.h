#pragma once

#include "ops/suite.h"
#include "ops/op_tester.h"
#include "ops/float_codec.h"
#include "ops/fp8_cache_reference.h"
#include "ninfer/kv_cache_format.h"
#include "../../../../tools/support/kpool_fixture.h"

#include <cmath>
#include <iostream>
#include <numeric>
#include <set>

namespace ninfer::test {
namespace kp = ninfer::tool::kpool;

inline constexpr ReductionCriterion kPreparedBf16Criterion{.0041, 1e-4, .012};
inline constexpr ReductionCriterion kPooledBf16Criterion{.006, 1e-4, .018};
inline constexpr ReductionCriterion kCoefficientCriterion{1e-4, 1e-6, 3e-4};
// E4M3 normal rounding contributes at most 1/16 relative error. The remaining
// budget covers BF16 raw-input production and pooling before the orthogonal map.
inline constexpr ReductionCriterion kPooledFp8Criterion{.07, 1e-4, .10};

inline std::array<double, 128> index_hadamard_reference(const std::array<double, 128>& values) {
    std::array<double, 128> result{};
    for (unsigned i = 0; i < 128; ++i) {
        for (unsigned j = 0; j < 128; ++j)
            result[i] += (std::popcount(i & j) & 1) ? -values[j] : values[j];
        result[i] /= std::sqrt(128.0);
    }
    return result;
}

struct KpoolError {
    std::vector<double> actual, reference;

    void add(double got, double expected) {
        actual.push_back(got);
        reference.push_back(expected);
    }

    int check(const char* name, const ReductionCriterion& criterion) const {
        return actual.empty() ? 0 : verify_reduction(name, actual, reference, criterion);
    }
};

// Full factor/softmax/factor budget, including private A16 operands and final
// BF16 output. The independent oracle has no private intermediate casts.
inline constexpr ReductionCriterion kMlaCriterion{8e-3, 1e-4, 1.2e-2};

// Only the selected set is observable. Bound the smallest score correction that
// makes that set a valid Top512 of the independent mathematical scores. This
// measures ranking error without reproducing private query quantization/MMA.
inline constexpr ReductionCriterion kFp8IndexRankingCriterion{.01, 1e-6, .05};

inline int verify_fp8_ranking(const std::vector<double>& scores, const std::set<int>& selected) {
    std::vector<bool> inside(scores.size(), false);
    for (int id : selected) inside[id] = true;
    double lowest = INFINITY, excluded = -INFINITY;
    for (int g = 0; g < int(scores.size()); ++g)
        if (inside[g])
            lowest = std::min(lowest, scores[g]);
        else
            excluded = std::max(excluded, scores[g]);
    if (lowest >= excluded) return 0;
    // Projection onto min(selected)>=max(excluded) is clipping both sides to a
    // common threshold. Its convex squared-error derivative is monotone.
    double lo = lowest, hi = excluded;
    for (int iteration = 0; iteration < 48; ++iteration) {
        const double pivot = (lo + hi) * .5;
        double derivative  = 0;
        for (int g = 0; g < int(scores.size()); ++g)
            if ((inside[g] && scores[g] < pivot) || (!inside[g] && scores[g] > pivot))
                derivative += pivot - scores[g];
        if (derivative < 0)
            lo = pivot;
        else
            hi = pivot;
    }
    auto closest       = scores;
    const double pivot = (lo + hi) * .5;
    for (int g = 0; g < int(scores.size()); ++g)
        closest[g] = inside[g] ? std::max(scores[g], pivot) : std::min(scores[g], pivot);
    return verify_reduction("FP8 selection correction", closest, scores, kFp8IndexRankingCriterion);
}

template <class Fixture>
int verify_selection(Fixture& f, bool exact = false) {
    const bool fp8 = f.cache.format == KvCacheVectorFormat::Fp8E4m3fnGroup128Ue8m0;
    const auto ids = f.ids.download(), counts = f.counts.download(), ends = f.ends.download();
    for (int b = 0; b < f.B; ++b) {
        for (int t = 0; t < f.T; ++t) {
            const int row = t + b * f.T;
            const int E = t < f.valid.at(b) ? f.positions.at(t, b) + 1 : 0, C = E / 4;
            const int count = std::min(C, 512);
            if (counts[row] != count || ends[row] != E) {
                std::cerr << "count/end mismatch row=" << row << '\n';
                return 1;
            }
            std::set<int> selected;
            for (int i = 0; i < count; ++i) {
                const int id = ids[row * 512 + i];
                if (id < 0 || id >= C || !selected.insert(id).second) {
                    std::cerr << "invalid/duplicate ID row=" << row << " id=" << id << '\n';
                    return 1;
                }
            }
            if (C <= 512) continue;
            // Sample query coordinates for large T, but evaluate their entire causal history.
            if (f.T > 16 && t != 0 && t != 7 && t != 15 && t != 16 && t != f.valid.at(b) - 1)
                continue;
            std::array<double, 4096> q;
            for (int h = 0; h < 32; ++h)
                for (int d = 0; d < 128; ++d) q[h * 128 + d] = kp::decode(f.query.at(d, h, t, b));
            if (fp8)
                for (int h = 0; h < 32; ++h) {
                    std::array<double, 128> input;
                    std::copy_n(q.begin() + h * 128, 128, input.begin());
                    const auto rotated = index_hadamard_reference(input);
                    std::copy(rotated.begin(), rotated.end(), q.begin() + h * 128);
                }
            std::vector<double> scores(C), errors(C);
            parallel_rows(C, std::uint64_t(C) * 4096, [&](int lo, int hi) {
                for (int g = lo; g < hi; ++g) {
                    std::array<double, 128> k;
                    const auto* raw = f.key(b, g);
                    if (fp8) {
                        const auto key_row = (raw - f.keys.data()) / 128;
                        const double scale = fp8_cache_multiplier(f.key_scales[key_row]);
                        for (int d = 0; d < 128; ++d)
                            k[d] =
                                float_codec::decode_e4m3fn(f.key_codes[key_row * 128 + d]) * scale;
                    } else {
                        for (int d = 0; d < 128; ++d) k[d] = kp::decode(raw[d]);
                    }
                    double score = 0, absolute = 0, weighted = 0;
                    for (int h = 0; h < 32; ++h) {
                        double dot = 0, absdot = 0;
                        for (int d = 0; d < 128; ++d) {
                            const double v = q[h * 128 + d] * k[d];
                            dot += v;
                            absdot += std::abs(v);
                        }
                        const double w = f.coeff.at(h, t, b);
                        score += w * std::max(0., dot);
                        absolute += std::abs(w) * absdot;
                        weighted += std::abs(w * dot);
                    }
                    // Independent forward error bound: 128-term dots, Lipschitz ReLU,
                    // then signed FP32 products and a 32-head reduction. No relative-to-score
                    // tolerance: positive and negative heads may nearly cancel.
                    constexpr double u = 0x1p-24, gd = 128 * u / (1 - 128 * u),
                                     gh = 64 * u / (1 - 64 * u);
                    scores[g]           = score;
                    errors[g] = exact ? 0 : gd * absolute + gh * (weighted + gd * absolute);
                }
            });
            if (exact) {
                std::vector<int> order(C);
                std::iota(order.begin(), order.end(), 0);
                std::partial_sort(order.begin(), order.begin() + 512, order.end(),
                                  [&](int a, int b) {
                                      return scores[a] != scores[b] ? scores[a] > scores[b] : a < b;
                                  });
                if (selected != std::set<int>(order.begin(), order.begin() + 512)) {
                    std::cerr << "exact Top512 mismatch row=" << row << '\n';
                    return 1;
                }
            } else if (fp8) {
                if (const auto failures = verify_fp8_ranking(scores, selected)) return failures;
            } else {
                double lowest = INFINITY, excluded = -INFINITY;
                for (int g = 0; g < C; ++g) {
                    if (selected.contains(g))
                        lowest = std::min(lowest, scores[g] + errors[g]);
                    else
                        excluded = std::max(excluded, scores[g] - errors[g]);
                }
                if (!(excluded <= lowest)) {
                    std::cerr << "ranking outside error bound row=" << row << " selected=" << lowest
                              << " excluded=" << excluded << '\n';
                    return 1;
                }
            }
        }
    }
    return 0;
}

// FP64 formula from represented public inputs; no private A16 intermediate seams.
inline std::vector<double> mla_reference(const std::uint16_t* query, const std::uint16_t* key_up,
                                         const std::uint16_t* value_up,
                                         const std::vector<const std::uint16_t*>& rows,
                                         float scale) {
    std::vector<double> c(rows.size() * 512), result(64 * 256);
    if (rows.empty()) return result;
    for (std::size_t j = 0; j < rows.size(); ++j)
        for (int d = 0; d < 512; ++d) c[j * 512 + d] = kp::decode(rows[j][d]);
    parallel_rows(64, std::uint64_t(rows.size()) * 64 * 1024, [&](int lo, int hi) {
        for (int h = lo; h < hi; ++h) {
            std::array<double, 512> q{}, u{};
            for (int k = 0; k < 256; ++k) {
                const double x = kp::decode(query[h * 256 + k]);
                for (int d = 0; d < 512; ++d)
                    q[d] += x * kp::decode(key_up[(h * 256 + k) * 512 + d]);
            }
            std::vector<double> logits(rows.size());
            double maximum = -INFINITY;
            for (std::size_t j = 0; j < rows.size(); ++j) {
                double dot = 0;
                for (int d = 0; d < 512; ++d) dot += q[d] * c[j * 512 + d];
                logits[j] = scale * dot;
                maximum   = std::max(maximum, logits[j]);
            }
            double denominator = 0;
            for (double& x : logits) {
                x = std::exp(x - maximum);
                denominator += x;
            }
            for (std::size_t j = 0; j < rows.size(); ++j)
                for (int d = 0; d < 512; ++d) u[d] += logits[j] / denominator * c[j * 512 + d];
            for (int v = 0; v < 256; ++v)
                for (int d = 0; d < 512; ++d)
                    result[h * 256 + v] += kp::decode(value_up[(h * 256 + v) * 512 + d]) * u[d];
        }
    });
    return result;
}
} // namespace ninfer::test
