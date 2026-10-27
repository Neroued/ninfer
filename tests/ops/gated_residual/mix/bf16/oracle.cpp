#include "support.h"
#include "../../support.h"

#include "ops/suite.h"

namespace ninfer::test::hc::mix::bf16 {
namespace {
double sigmoid(double x) {
    if (x >= 0) return 1.0 / (1.0 + std::exp(-x));
    const double e = std::exp(x);
    return e / (1.0 + e);
}

double dot(std::span<const float> w, std::span<const float> x) {
    double sum = 0;
    for (std::size_t k = 0; k < x.size(); ++k) sum += double(w[k]) * double(x[k]);
    return sum;
}
} // namespace

Column reference_column(const Weights& w, std::span<const float> z) {
    std::vector<double> u(kRank);
    for (int r = 0; r < kRank; ++r) {
        const double a =
            dot(std::span{w.down_values.data() + std::size_t(r) * kTotal, std::size_t(kTotal)}, z) /
            4.0;
        u[r] = a * sigmoid(a);
    }
    Column result{std::vector<double>(kWidth), {}};
    for (int s = 0; s < kStreams; ++s)
        for (int d = 0; d < kWidth; ++d) {
            const auto row = s * kWidth + d;
            double g       = 0;
            for (int r = 0; r < kRank; ++r)
                g += double(w.up_values[std::size_t(row) * kRank + r]) * u[r];
            result.output[d] += sigmoid(g) * double(z[row]) / 4.0;
        }
    if (w.kind == Kind::Block) {
        result.injection.resize(kStreams);
        for (int s = 0; s < kStreams; ++s)
            result.injection[s] =
                2.0 * sigmoid(dot(std::span{w.down_values.data() + std::size_t(kRank + s) * kTotal,
                                            std::size_t(kTotal)},
                                  z) /
                              4.0);
    }
    return result;
}

std::vector<Column> reference(const Weights& weights, std::span<const float> catalog) {
    const int columns = int(catalog.size() / kTotal);
    std::vector<Column> result(columns);
    parallel_rows(columns, std::uint64_t(columns) * kTotal * kRank, [&](int first, int last) {
        for (int m = first; m < last; ++m)
            result[m] = reference_column(weights, catalog.subspan(std::size_t(m) * kTotal, kTotal));
    });
    return result;
}

int verify(const Fixture& f, std::span<const Column> ref, const std::string& label) {
    const auto output          = from_device_bf16(f.x.data, std::size_t(kWidth) * f.columns);
    const auto injection_words = from_device<float>(
        f.j.data, f.weights.kind == Kind::Block ? std::size_t(kStreams) * f.columns : 0);
    const std::vector<double> injection(injection_words.begin(), injection_words.end());
    int failed = 0;
    for (int m = 0; m < f.columns; ++m) {
        const auto& expected = ref[m % ref.size()];
        const auto tag       = label + " column=" + std::to_string(m);
        failed += verify_reduction(tag, std::span{output}.subspan(std::size_t(m) * kWidth, kWidth),
                                   expected.output, kMixOutput);
        if (f.weights.kind == Kind::Block) {
            failed +=
                verify_pointwise(tag + " injection",
                                 std::span{injection}.subspan(std::size_t(m) * kStreams, kStreams),
                                 expected.injection, kMixInjection);
            for (int s = 0; s < kStreams; ++s)
                if (injection[m * kStreams + s] < 0 || injection[m * kStreams + s] > 2) ++failed;
        }
    }
    return failed;
}
} // namespace ninfer::test::hc::mix
