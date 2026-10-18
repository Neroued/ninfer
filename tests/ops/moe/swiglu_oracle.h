#pragma once
#include "ops/quantized_weight.h"
#include "ops/op_tester.h"
#include "ops/suite.h"
#include <map>

namespace ninfer::test::swiglu {
namespace qw = quantized_weight;

struct Oracle {
    const int H, I;
    const qw::PackedWeight& gu;
    const qw::PackedWeight& dn;
    const std::vector<std::uint16_t>& input;
    std::vector<double> down;
    double limit;
    std::map<std::pair<int, int>, std::vector<double>> products;

    Oracle(const qw::PackedWeight& a, const qw::PackedWeight& b,
           const std::vector<std::uint16_t>& x, double clamp_limit = 10.0)
        : H(a.weight.k), I(b.weight.k), gu(a), dn(b), input(x), down(std::size_t(H) * I),
          limit(clamp_limit) {
        parallel_rows(H, std::uint64_t(H) * I, [&](int begin, int end) {
            for (int r = begin; r < end; ++r)
                for (int k = 0; k < I; ++k)
                    down[std::size_t(r) * I + k] = qw::logical_weight_fp64(dn, r, k);
        });
    }

    const std::vector<double>& product(int token, int phase) {
        const auto key = std::pair{token, phase};
        auto found     = products.find(key);
        if (found != products.end()) return found->second;
        std::vector<std::pair<int, double>> active;
        for (int k = 0; k < H; ++k) {
            const double x = bf16_to_f32(input[std::size_t(token) * H + k]) * (phase ? -1.0 : 1.0);
            if (x != 0) active.emplace_back(k, x);
        }
        std::vector<double> values(I);
        parallel_rows(I, std::uint64_t(I) * active.size(), [&](int begin, int end) {
            for (int r = begin; r < end; ++r) {
                double g = 0, u = 0;
                for (auto [k, x] : active) {
                    g += qw::logical_weight_fp64(gu, r, k) * x;
                    u += qw::logical_weight_fp64(gu, r + I, k) * x;
                }
                g = std::min(g, limit);
                u = std::clamp(u, -limit, limit);
                const double silu =
                    g >= 0 ? g / (1 + std::exp(-g)) : g * std::exp(g) / (1 + std::exp(g));
                values[r] = silu * u;
            }
        });
        return products.emplace(key, std::move(values)).first->second;
    }

    double output(int row, int token, int phase) {
        const auto& p = product(token, phase);
        double sum    = 0;
        for (int k = 0; k < I; ++k) sum += down[std::size_t(row) * I + k] * p[k];
        return sum;
    }
};

void add_bf16_cases(OpSuite&);
} // namespace ninfer::test::swiglu
