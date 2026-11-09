#include "ops/suite.h"
#include "ops/normalization/support.h"
#include "cuda_fixture.h"
#include "core/decode_graph.h"
#include "ninfer/ops/rmsnorm_gathered_features.h"
#include <cmath>
#include <limits>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::norm;

namespace {
constexpr ReductionCriterion kCriterion{1.85e-3, 1.e-5, 3.4e-3};

int run(int columns, bool graph, float magnitude = 1.F) {
    constexpr float epsilon = 1.e-5F;
    std::vector<float> x(std::size_t(4096) * columns), gamma(4096);
    fill_uniform(x, 51101U, -magnitude, magnitude);
    fill_uniform(gamma, 51103U, -1.F, 2.F);
    // Unequal rank energies distinguish a full-feature norm from two local norms.
    for (std::size_t i = std::size_t(2048) * columns; i < x.size(); ++i) x[i] *= 7.F;
    round_to_bf16(x);
    round_to_bf16(gamma);
    auto dx = make_input(x, false), dg = make_input(gamma, false);
    GuardedDeviceBuffer dy(x.size() * 2);
    Tensor tx(dx.memory, DType::BF16, {2048, columns, 2}), tg(dg.memory, DType::BF16, {4096}),
        ty(dy.span(), DType::BF16, {4096, columns});
    const auto e = cuda_environment().context.execution_view();
    DecodeGraphDefinition definition;
    DecodeGraphExecutable executable;
    auto launch = [&] { ops::rmsnorm_gathered_features(tx, tg, epsilon, ty, e); };
    cuda_synchronize();
    if (graph) {
        definition.capture(e.stream(), launch);
        executable.instantiate(definition);
    }
    int failures = 0;
    for (int replay = 0; replay < (graph ? 2 : 1); ++replay) {
        if (replay) {
            for (std::size_t i = 0; i < x.size(); ++i) {
                x[i]           = -x[i];
                dx.expected[i] = f32_to_bf16(x[i]);
            }
            dx.storage.copy_from_host(dx.expected.data(), dx.expected.size() * 2);
        }
        cuda_check(cudaMemsetAsync(ty.data, 0xff, x.size() * 2, e.stream()),
                   "poison gathered output");
        if (graph)
            executable.launch(e.stream());
        else
            launch();
        cuda_synchronize(e.stream());
        std::vector<double> expected(x.size());
        for (int m = 0; m < columns; ++m) {
            double square = 0;
            for (int h = 0; h < 4096; ++h) {
                const auto index = std::size_t(h / 2048) * 2048 * columns + m * 2048 + h % 2048;
                square += double(x[index]) * x[index];
            }
            const double inverse = 1 / std::sqrt(square / 4096 + epsilon);
            for (int h = 0; h < 4096; ++h) {
                const auto index = std::size_t(h / 2048) * 2048 * columns + m * 2048 + h % 2048;
                expected[std::size_t(m) * 4096 + h] = x[index] * inverse * gamma[h];
            }
        }
        failures += verify_reduction("gathered features", from_device_bf16(dy.data(), x.size()),
                                     expected, kCriterion);
        failures += dy.verify_guards("gathered output");
        failures += verify_preserved("gathered input", dx) + verify_preserved("gathered gamma", dg);
    }
    return failures;
}

int contract() {
    const auto e = cuda_environment().default_execution;
    DeviceBuffer data(e.device(), 65536);
    Tensor x(data.span().subspan(0, 16384), DType::BF16, {2048, 2, 2}),
        gamma(data.span().subspan(16384, 8192), DType::BF16, {4096}),
        out(data.span().subspan(24576, 16384), DType::BF16, {4096, 2});
    int failures      = 0;
    const auto reject = [&](float epsilon) {
        try {
            ops::rmsnorm_gathered_features(x, gamma, epsilon, out, e);
            ++failures;
        } catch (const std::invalid_argument&) {}
    };
    for (float value : {0.F, -1.F, std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::quiet_NaN()})
        reject(value);
    x.ne[2] = 1;
    reject(1.e-5F);
    x.ne[2]  = 2;
    out.data = x.data;
    reject(1.e-5F);
    return failures;
}
} // namespace

void add_gathered_features_cases(OpSuite& suite) {
    for (int columns : {1, 7, 32, 33, 128, 256, 1024, 2048})
        suite.add("gathered-features/m" + std::to_string(columns),
                  [=] { return run(columns, columns == 7 || columns == 2048); },
                  {{"execution", columns == 7 || columns == 2048 ? "graph" : "eager"}});
    suite.add("gathered-features/epsilon", [] { return run(2, false, 1.e-4F); });
    suite.add("gathered-features/contract", contract);
}
