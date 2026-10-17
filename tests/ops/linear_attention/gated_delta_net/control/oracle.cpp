#include "support.h"

#include <algorithm>
#include <cmath>

namespace ninfer::test::gdn_control {

double softplus(double value) {
    return std::max(value, 0.0) + std::log1p(std::exp(-std::abs(value)));
}

double sigmoid(double value) {
    if (value >= 0) return 1.0 / (1.0 + std::exp(-value));
    const double e = std::exp(value);
    return e / (1.0 + e);
}

Reference projection_reference(const Fixture& f) {
    const int h = f.profile.heads, k = f.profile.input_rows, m = f.test_case.columns;
    Reference result;
    result.g.resize(std::size_t(h) * m);
    result.beta.resize(result.g.size());
    if (f.normalize_input) result.normalized.resize(std::size_t(k) * m);
    std::vector<double> column(k);
    for (int token = 0; token < m; ++token) {
        double ss = 0;
        for (int row = 0; row < k; ++row) {
            const double x = f.x[std::size_t(token) * k + row];
            ss += x * x;
            column[row] = x;
        }
        if (f.normalize_input) {
            const double inv = 1.0 / std::sqrt(ss / k + double(kEps));
            for (int row = 0; row < k; ++row) {
                column[row] *= inv * (1.0 + double(f.norm_gain[row]));
                result.normalized[std::size_t(token) * k + row] = column[row];
            }
        }
        for (int head = 0; head < h; ++head) {
            double a = 0, b = 0;
            for (int row = 0; row < k; ++row) {
                a += double(f.weights.a[std::size_t(head) * k + row]) * column[row];
                b += double(f.weights.b[std::size_t(head) * k + row]) * column[row];
            }
            const auto i = std::size_t(token) * h + head;
            result.g[i]  = -std::exp(double(f.a_log[head])) * softplus(a + double(f.dt_bias[head]));
            result.beta[i] = sigmoid(b);
        }
    }
    return result;
}

int verify_projection(std::string_view name, const Fixture& f, const Reference& expected) {
    const auto read = [](const GuardedDeviceBuffer& buffer, std::size_t count) {
        const auto floats = from_device<float>(buffer.data(), count);
        return std::vector<double>(floats.begin(), floats.end());
    };
    const std::string prefix(name);
    const auto criterion = f.normalize_input ? kNormControl : kProjection;
    int failures         = verify_reduction((prefix + " g").c_str(), read(f.g, expected.g.size()),
                                            expected.g, criterion);
    failures += verify_reduction((prefix + " beta").c_str(), read(f.beta, expected.beta.size()),
                                 expected.beta, criterion);
    if (f.normalize_input)
        failures +=
            verify_reduction((prefix + " normalized").c_str(),
                             from_device_bf16(f.normalized.data(), expected.normalized.size()),
                             expected.normalized, kNormalized);
    return failures;
}

} // namespace ninfer::test::gdn_control
