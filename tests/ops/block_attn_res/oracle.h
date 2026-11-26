#pragma once

#include "ninfer/ops/block_attn_res.h"
#include "ops/op_tester.h"

namespace ninfer::test::attn_res {

inline std::vector<float> values(std::size_t size, unsigned seed, float extent = 1.0f) {
    std::vector<float> result(size);
    fill_uniform(result, seed, -extent, extent);
    return result;
}

inline std::vector<double>
oracle(int d, int m, int q, int stacked, const std::vector<double>& stack,
       const std::vector<double>& partial, const std::vector<double>& addend,
       const std::vector<float>& queries, const std::vector<double>& weights,
       const ops::BlockAttnResParameters& params, const std::vector<int>& active = {}) {
    const std::size_t plane = std::size_t(d) * m;
    std::vector<double> out(plane * q);
    parallel_rows(m, plane * q * (stacked + 1), [&](int first, int last) {
        std::vector<double> dots(stacked + 1), rms(stacked + 1);
        std::vector<double> sources(std::size_t(d) * (stacked + 1));
        for (int t = first; t < last; ++t) {
            if (!active.empty() && !active[t]) continue;
            const auto col = std::size_t(t) * d;
            for (int s = 0; s <= stacked; ++s) {
                double squares = 0;
                for (int c = 0; c < d; ++c) {
                    const double x =
                        s < stacked ? stack[s * plane + col + c]
                                    : partial[col + c] + (addend.empty() ? 0.0 : addend[col + c]);
                    squares += x * x;
                    sources[std::size_t(s) * d + c] = x;
                }
                rms[s] = 1.0 / std::sqrt(squares / d + params.rms_eps);
            }
            for (int query = 0; query < q; ++query) {
                const auto query_base = std::size_t(query) * d;
                for (int s = 0; s <= stacked; ++s) {
                    double dot = 0;
                    for (int c = 0; c < d; ++c)
                        dot += queries[query_base + c] * sources[std::size_t(s) * d + c];
                    dots[s] = params.query_scale * dot * rms[s];
                }
                const double max   = *std::max_element(dots.begin(), dots.end());
                double denominator = 0;
                for (auto& p : dots) {
                    p = std::exp(p - max);
                    denominator += p;
                }
                const auto dst = query * plane + col;
                double squares = 0;
                for (int c = 0; c < d; ++c) {
                    double sum = 0;
                    for (int s = 0; s <= stacked; ++s) {
                        const double p = dots[s] / denominator;
                        sum += (params.normalize_values ? p * rms[s] : p) *
                               sources[std::size_t(s) * d + c];
                    }
                    out[dst + c] = sum;
                    squares += sum * sum;
                }
                if (!weights.empty()) {
                    const double norm = 1.0 / std::sqrt(squares / d + params.norm_eps);
                    for (int c = 0; c < d; ++c) out[dst + c] *= norm * weights[query_base + c];
                }
            }
        }
    });
    return out;
}

inline int check(std::string_view name, const DeviceBuffer& out, std::span<const double> ref,
                 bool norm) {
    constexpr ReductionCriterion bf16{3e-3, 1.e-5, 7.85e-3};
    constexpr ReductionCriterion fp32{1e-5, 1.e-6, 3.e-5};
    return verify_reduction(
        name, norm ? from_device_bf16(out, ref.size()) : from_device_f32(out, ref.size()), ref,
        norm ? bf16 : fp32);
}

template <class F>
int rejects(std::string_view name, F&& f) {
    try {
        f();
    } catch (const std::invalid_argument&) { return 0; }
    std::cerr << name << ": expected invalid_argument\n";
    return 1;
}

} // namespace ninfer::test::attn_res
