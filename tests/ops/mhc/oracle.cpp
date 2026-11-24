#include "support.h"

namespace ninfer::test::mhc {
double round_bf16(double value) {
    // Round directly from FP64 to the BF16 grid, avoiding an FP32 double-rounding seam.
    if (value == 0) return value;
    int exponent;
    std::frexp(std::abs(value), &exponent);
    const double unit   = std::ldexp(1., std::max(exponent - 8, -133));
    const double scaled = std::abs(value) / unit, lower = std::floor(scaled), tail = scaled - lower;
    const double rounded = lower + (tail > 0.5 || (tail == 0.5 && std::fmod(lower, 2.) != 0.));
    return std::copysign(rounded * unit, value);
}

PreReference pre_reference(const Fixture& f, std::span<const double> x) {
    PreReference out{std::vector<double>(D), std::vector<double>(S), std::vector<double>(S * S)};
    double ss = 0;
    for (double v : x) ss += v * v;
    const double inverse = 1 / std::sqrt(ss / K + f.parameters.coefficient_epsilon);
    std::array<double, N> z{};
    for (int a = 0; a < N; ++a)
        for (int k = 0; k < K; ++k) z[a] += double(f.f[a * K + k]) * x[k];
    for (int a = 0; a < N; ++a)
        z[a] = double(f.scale[a < 4 ? 0 : a < 8 ? 1 : 2]) * z[a] * inverse + f.base[a];
    std::array<double, S> p{};
    for (int i = 0; i < S; ++i) {
        p[i]     = 1 / (1 + std::exp(-z[i])) + f.parameters.pre_epsilon;
        out.q[i] = f.parameters.post_multiplier / (1 + std::exp(-z[4 + i]));
    }
    const double eps = f.parameters.sinkhorn_epsilon;
    for (int i = 0; i < S; ++i) {
        double max = z[8 + i * S], sum = 0;
        for (int j = 1; j < S; ++j) max = std::max(max, z[8 + i * S + j]);
        for (int j = 0; j < S; ++j) sum += out.c[i * S + j] = std::exp(z[8 + i * S + j] - max);
        for (int j = 0; j < S; ++j) out.c[i * S + j] = out.c[i * S + j] / sum + eps;
    }
    for (int iteration = 0; iteration < f.parameters.sinkhorn_iterations; ++iteration) {
        if (iteration)
            for (int i = 0; i < S; ++i) {
                double sum = eps;
                for (int j = 0; j < S; ++j) sum += out.c[i * S + j];
                for (int j = 0; j < S; ++j) out.c[i * S + j] /= sum;
            }
        for (int j = 0; j < S; ++j) {
            double sum = eps;
            for (int i = 0; i < S; ++i) sum += out.c[i * S + j];
            for (int i = 0; i < S; ++i) out.c[i * S + j] /= sum;
        }
    }
    double norm = 0;
    for (int d = 0; d < D; ++d) {
        for (int i = 0; i < S; ++i) out.u[d] += p[i] * x[i * D + d];
        norm += out.u[d] * out.u[d];
    }
    norm = 1 / std::sqrt(norm / D + f.parameters.branch_epsilon);
    for (int d = 0; d < D; ++d) out.u[d] *= norm * double(f.gamma[d]);
    return out;
}

std::vector<double> post_reference(const Fixture& f, int m) {
    std::vector<double> result(K);
    for (int j = 0; j < S; ++j)
        for (int d = 0; d < D; ++d) {
            double v = 0;
            for (int i = 0; i < S; ++i)
                v += double(f.c[m * 16 + i * S + j]) * double(f.x[std::size_t(m) * K + i * D + d]);
            result[j * D + d] = v + double(f.q[m * S + j]) * double(f.y[std::size_t(m) * D + d]);
        }
    return result;
}
} // namespace ninfer::test::mhc
