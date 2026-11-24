#include "ops/suite.h"
#include "ops/op_tester.h"
#include "ninfer/ops/rmsnorm_rope.h"
#include "cuda_fixture.h"
#include "core/decode_graph.h"
#include <cmath>

using namespace ninfer;
using namespace ninfer::test;

namespace {
constexpr ReductionCriterion kAxialNormBf16{2.e-3, 2.e-5, 6.9e-3};

std::vector<double> reference(const std::vector<float>& x, const std::vector<float>& gain,
                              const std::vector<int>& positions, float epsilon, float theta) {
    std::vector<double> result(x.size());
    const int tokens = positions.size() / 2;
    for (int t = 0; t < tokens; ++t)
        for (int h = 0; h < 16; ++h) {
            const auto base = (std::size_t(t) * 16 + h) * 64;
            double square   = 0;
            for (int d = 0; d < 64; ++d) square += double(x[base + d]) * x[base + d];
            const double inverse = 1 / std::sqrt(square / 64 + epsilon);
            for (int i = 0; i < 32; ++i) {
                const double angle = positions[t + (i / 16) * tokens] *
                                     std::pow(double(theta), -double(i % 16) / 16);
                const double first    = x[base + i] * inverse * gain[i],
                             second   = x[base + i + 32] * inverse * gain[i + 32];
                result[base + i]      = first * std::cos(angle) - second * std::sin(angle);
                result[base + i + 32] = second * std::cos(angle) + first * std::sin(angle);
            }
        }
    return result;
}

int run(int tokens, bool graph, bool natural_gain, float theta) {
    constexpr float epsilon = 1.e-5F;
    std::vector<float> q(std::size_t(tokens) * 1024), k(q.size()), qw(64), kw(64);
    fill_uniform(q, 6301, -3.F, 3.F);
    fill_uniform(k, 6307, -2.F, 2.F);
    fill_uniform(qw, 6311, -1.F, 1.F);
    fill_uniform(kw, 6317, -1.F, 1.F);
    for (auto* x : {&q, &k, &qw, &kw}) round_to_bf16(*x);
    std::vector<int> positions(tokens * 2);
    for (int t = 0; t < tokens; ++t) {
        positions[t]          = t % 61 - 2;
        positions[t + tokens] = t / 61 + 3;
    }
    const int prefix = natural_gain ? 1 : 0;
    auto gain_bits   = [&](const auto& values) {
        std::vector<uint16_t> bits(64 + prefix, 0x5a5a);
        for (int d = 0; d < 64; ++d) bits[d + prefix] = f32_to_bf16(values[d]);
        return bits;
    };
    auto qbits = gain_bits(qw), kbits = gain_bits(kw);
    auto qgain = to_device(qbits), kgain = to_device(kbits), pos = to_device(positions);
    GuardedDeviceBuffer qdata(q.size() * 2), kdata(k.size() * 2);
    Tensor tq(qdata.span(), DType::BF16, {64, 16, tokens}),
        tk(kdata.span(), DType::BF16, {64, 16, tokens}),
        wq(qgain.span().subspan(prefix * 2), DType::BF16, {64}),
        wk(kgain.span().subspan(prefix * 2), DType::BF16, {64}),
        p(pos.span(), DType::I32, {tokens, 2});
    const auto e = cuda_environment().context.execution_view();
    DecodeGraphDefinition definition;
    DecodeGraphExecutable executable;
    auto launch  = [&] { ops::rmsnorm_rope_2d(p, wq, wk, epsilon, theta, tq, tk, e); };
    int failures = 0;
    for (int phase = 0; phase < (graph ? 2 : 1); ++phase) {
        if (phase) {
            for (auto& x : q) x = -x;
            for (auto& x : qw) x = -x;
            for (auto& x : positions) x += 5;
        }
        auto upload = [](void* dst, const std::vector<float>& x) {
            std::vector<uint16_t> bits(x.size());
            for (size_t i = 0; i < x.size(); ++i) bits[i] = f32_to_bf16(x[i]);
            CUDA_CHECK(cudaMemcpy(dst, bits.data(), bits.size() * 2, cudaMemcpyHostToDevice));
        };
        upload(tq.data, q);
        upload(tk.data, k);
        upload(wq.data, qw);
        CUDA_CHECK(
            cudaMemcpy(pos.data(), positions.data(), positions.size() * 4, cudaMemcpyHostToDevice));
        if (graph && phase == 0) {
            definition.capture(e.stream(), launch);
            executable.instantiate(definition);
        }
        if (graph)
            executable.launch(e.stream());
        else
            launch();
        cuda_synchronize(e.stream());
        failures += verify_reduction("axial q norm", from_device_bf16(tq.data, q.size()),
                                     reference(q, qw, positions, epsilon, theta), kAxialNormBf16);
        failures += verify_reduction("axial k norm", from_device_bf16(tk.data, k.size()),
                                     reference(k, kw, positions, epsilon, theta), kAxialNormBf16);
        failures +=
            verify_exact("q gain", from_device<uint16_t>(qgain, 64 + prefix), gain_bits(qw));
        failures += verify_exact("k gain", from_device<uint16_t>(kgain, 64 + prefix), kbits);
        failures += verify_exact("positions", from_device<int>(pos, positions.size()), positions);
        failures += qdata.verify_guards("q guards") + kdata.verify_guards("k guards");
    }
    return failures;
}

int contract() {
    const auto e = cuda_environment().context.execution_view();
    DeviceBuffer data(e.device(), 16384);
    Tensor q(data.span().subspan(0, 4096), DType::BF16, {64, 16, 2}),
        k(data.span().subspan(4096, 4096), DType::BF16, {64, 16, 2}),
        w(data.span().subspan(8192, 128), DType::BF16, {64}),
        p(data.span().subspan(8320, 16), DType::I32, {2, 2});
    int failures = 0;
    auto launch  = [&] { ops::rmsnorm_rope_2d(p, w, w, 1.e-5F, 10000.F, q, k, e); };
    auto reject  = [&] {
        try {
            launch();
            ++failures;
        } catch (const std::invalid_argument&) {}
    };
    auto saved = k;
    k.data     = q.data;
    reject();
    k       = saved;
    q.ne[0] = 72;
    reject();
    q.ne[0] = 64;
    p.ne[1] = 1;
    reject();
    p.ne[1]           = 2;
    auto saved_weight = w;
    w.data            = q.data;
    reject();
    w       = saved_weight;
    q.ne[2] = k.ne[2] = p.ne[0] = 0;
    launch();
    return failures;
}
} // namespace

void add_rmsnorm_rope_2d_cases(OpSuite& suite) {
    for (int tokens : {1, 2, 7, 32, 127, 1024, 4097})
        suite.add("rmsnorm-rope-2d/N=" + std::to_string(tokens),
                  [=] { return run(tokens, tokens == 7, false, 10000.F); },
                  {{"execution", tokens == 7 ? "graph" : "eager"}});
    suite.add("rmsnorm-rope-2d/natural-gain", [] { return run(3, true, true, 5700.F); },
              {{"execution", "graph"}});
    suite.add("rmsnorm-rope-2d/contract", contract);
}
