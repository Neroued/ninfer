#pragma once
#include "ops/suite.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace vda_ref {
constexpr int D = 128, M = D * D;

struct Input {
    int frames, per_frame, heads, batch, chunk = 5, radius = 1, offset = 1;
    std::vector<float> q, k, v, alpha, beta, initial, weight, gate;

    std::size_t vi(int b, int t, int h, int d = 0) const {
        return ((std::size_t(b) * frames * per_frame + t) * heads + h) * D + d;
    }

    std::size_t ai(int b, int f, int h, int d = 0) const {
        return ((std::size_t(b) * frames + f) * heads + h) * D + d;
    }

    std::size_t bi(int b, int t, int h) const {
        return (std::size_t(b) * frames * per_frame + t) * heads + h;
    }

    std::size_t si(int b, int h) const { return (std::size_t(b) * heads + h) * M; }
};

// Independent FP64 Cholesky and direct normal-equation solve; no production factors,
// inverse, tensor-core casts, affine composition or boundary scan reused by this oracle.
inline void factor(std::vector<double>& a) {
    for (int i = 0; i < D; ++i)
        for (int j = 0; j <= i; ++j) {
            double z = a[i * D + j];
            for (int p = 0; p < j; ++p) z -= a[i * D + p] * a[j * D + p];
            if (i == j) {
                if (!(z > 0)) throw std::runtime_error("FP64 oracle lost SPD");
                a[i * D + j] = std::sqrt(z);
            } else
                a[i * D + j] = z / a[j * D + j];
        }
}

inline void solve(const double* l, double* rhs) {
    for (int r = 0; r < D; ++r) {
        auto* x = rhs + r * D;
        for (int i = 0; i < D; ++i) {
            double v = x[i];
            for (int j = 0; j < i; ++j) v -= l[i * D + j] * x[j];
            x[i] = v / l[i * D + i];
        }
        for (int i = D - 1; i >= 0; --i) {
            double v = x[i];
            for (int j = i + 1; j < D; ++j) v -= l[j * D + i] * x[j];
            x[i] = v / l[i * D + i];
        }
    }
}

inline void statistics(const Input& in, int b, int h, int f, std::vector<double>& a,
                       std::vector<double>& write) {
    a.assign(M, 0);
    write.assign(M, 0);
    for (int u = 0; u < in.per_frame; ++u) {
        const int t       = f * in.per_frame + u;
        const auto at     = in.vi(b, t, h);
        const double beta = in.beta[in.bi(b, t, h)];
        for (int i = 0; i < D; ++i) {
            const double ki = in.k[at + i] * beta, vi = in.v[at + i] * beta;
            for (int j = 0; j <= i; ++j) a[i * D + j] += ki * in.k[at + j];
            for (int j = 0; j < D; ++j) write[i * D + j] += vi * in.k[at + j];
        }
    }
    for (int i = 0; i < D; ++i) {
        a[i * D + i] += 1;
        for (int j = 0; j < i; ++j) a[j * D + i] = a[i * D + j];
    }
}

inline std::vector<double> initial(const Input& in, double scale) {
    std::vector<double> out(std::size_t(in.batch) * in.heads * M);
    ninfer::test::parallel_rows(in.batch * in.heads, std::uint64_t(in.per_frame) * D * D,
                                [&](int first, int end) {
                                    for (int bh = first; bh < end; ++bh) {
                                        std::vector<double> a, w;
                                        statistics(in, bh / in.heads, bh % in.heads, 0, a, w);
                                        factor(a);
                                        solve(a.data(), w.data());
                                        for (int i = 0; i < M; ++i)
                                            out[std::size_t(bh) * M + i] = scale * w[i];
                                    }
                                });
    return out;
}

inline std::vector<double> evaluate(const Input& in, bool gated) {
    std::vector<double> out(in.q.size());
    if (!in.frames) return out;
    ninfer::test::parallel_rows(
        in.batch * in.heads, std::uint64_t(in.frames) * in.per_frame * D * D,
        [&](int first, int end) {
            for (int bh = first; bh < end; ++bh) {
                const int b = bh / in.heads, h = bh % in.heads, F = in.frames;
                std::vector<double> factors(std::size_t(F) * M), writes(factors.size()),
                    forward(factors.size()), reverse(factors.size());
                for (int f = 0; f < F; ++f) {
                    std::vector<double> a, w;
                    statistics(in, b, h, f, a, w);
                    factor(a);
                    std::copy(a.begin(), a.end(), factors.begin() + std::size_t(f) * M);
                    std::copy(w.begin(), w.end(), writes.begin() + std::size_t(f) * M);
                }
                for (int direction = 0; direction < 2; ++direction) {
                    std::vector<double> s(in.initial.begin() + in.si(b, h),
                                          in.initial.begin() + in.si(b, h) + M);
                    auto& states = direction ? reverse : forward;
                    for (int step = 0; step < F; ++step) {
                        int f         = direction ? F - 1 - step : step;
                        const auto at = std::size_t(f) * M;
                        for (int v = 0; v < D; ++v)
                            for (int k = 0; k < D; ++k)
                                s[v * D + k] = s[v * D + k] * in.alpha[in.ai(b, f, h, k)] +
                                               writes[at + v * D + k];
                        solve(factors.data() + at, s.data());
                        std::copy(s.begin(), s.end(), states.begin() + at);
                    }
                }
                for (int f = 0; f < F; ++f) {
                    const auto c = (std::int64_t(f) + in.offset) / in.chunk;
                    const auto lo =
                        std::max<std::int64_t>(0, (c - in.radius) * in.chunk - in.offset);
                    const auto hi = std::min<std::int64_t>(F - 1, (c + in.radius + 1) * in.chunk -
                                                                      in.offset - 1);
                    double dl[D], dr[D];
                    for (int k = 0; k < D; ++k) {
                        dl[k] = dr[k] = 1;
                        for (int j = lo; j <= f; ++j)
                            dl[k] *= std::max<double>(in.alpha[in.ai(b, j, h, k)], 1e-12);
                        for (int j = f; j <= hi; ++j)
                            dr[k] *= std::max<double>(in.alpha[in.ai(b, j, h, k)], 1e-12);
                    }
                    std::vector<double> s(M);
                    for (int i = 0; i < M; ++i) {
                        double l = lo ? forward[(lo - 1) * M + i] : in.initial[in.si(b, h) + i];
                        double r =
                            hi < F - 1 ? reverse[(hi + 1) * M + i] : in.initial[in.si(b, h) + i];
                        s[i] = l * dl[i % D] + r * dr[i % D];
                    }
                    for (int u = 0; u < in.per_frame; ++u) {
                        const auto at = in.vi(b, f * in.per_frame + u, h);
                        double y[D], sum = 0;
                        for (int v = 0; v < D; ++v) {
                            double x = 0;
                            for (int k = 0; k < D; ++k) x += s[v * D + k] * in.q[at + k];
                            y[v] = x;
                            sum += x * x;
                        }
                        const double inv = 1 / std::sqrt(sum / D + 1e-6);
                        for (int d = 0; d < D; ++d)
                            out[at + d] = gated ? y[d] * inv * in.weight[d] /
                                                      (1 + std::exp(-double(in.gate[at + d])))
                                                : y[d];
                    }
                }
            }
        });
    return out;
}
} // namespace vda_ref
