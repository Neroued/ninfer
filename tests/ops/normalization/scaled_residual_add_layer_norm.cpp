#include "ninfer/ops/scaled_residual_add_layer_norm.h"
#include "core/decode_graph.h"
#include "ops/normalization/support.h"

#include <limits>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr float kEps = 1.0e-5f;
constexpr ReductionCriterion kCriterion{2.05e-3, 3.e-6, 4.1e-3};

enum class Values { Random, LargeMean, Rounding, Overflow, LargeEpsilon };

std::vector<double> norm_reference(const std::vector<std::uint16_t>& r,
                                   const std::vector<float>& weight, const std::vector<float>& bias,
                                   int d, int m, float eps) {
    std::vector<double> result(r.size());
    parallel_rows(m, r.size(), [&](int first, int last) {
        for (int row = first; row < last; ++row) {
            const auto base = std::size_t(row) * d;
            double mean     = 0;
            for (int i = 0; i < d; ++i) mean += bf16_to_f32(r[base + i]);
            mean /= d;
            double variance = 0;
            for (int i = 0; i < d; ++i) {
                const double centered = double(bf16_to_f32(r[base + i])) - mean;
                variance += centered * centered;
            }
            const double inverse = 1. / std::sqrt(variance / d + eps);
            for (int i = 0; i < d; ++i)
                result[base + i] =
                    (double(bf16_to_f32(r[base + i])) - mean) * inverse * weight[i] + bias[i];
        }
    });
    return result;
}

int run(int d, int m, bool bf, bool fused, bool alias, bool natural, bool graph, Values values) {
    const std::size_t count = std::size_t(d) * m, lead = natural ? 1 : 0;
    std::vector<float> residual(count), update(count), gamma(d), weight(d), bias(d);
    fill_uniform(residual, 3931, -3.f, 3.f);
    fill_uniform(update, 3937, -2.f, 2.f);
    fill_uniform(gamma, 3943, -.75f, .75f);
    fill_uniform(weight, 3953, -.5f, 1.5f);
    fill_uniform(bias, 3967, -.25f, .25f);
    const float eps = values == Values::LargeEpsilon ? std::numeric_limits<float>::max() : kEps;
    if (values == Values::LargeMean) {
        fill_uniform(residual, 3931, 999.99f, 1000.01f);
        fill_uniform(update, 3937, -.01f, .01f);
        // Some rows round to a constant; others have only one BF16 step of variation.
        for (int row = 1; row < m; row += 2) residual[std::size_t(row) * d + d - 1] = 1004.f;
        std::fill(bias.begin(), bias.end(), 0.f);
    } else if (values == Values::Overflow || values == Values::LargeEpsilon) {
        constexpr float magnitudes[]{0x1p60f, 0x1p126f, 0x1.fep127f};
        for (int row = 0; row < m; ++row)
            for (int i = 0; i < d; ++i)
                residual[std::size_t(row) * d + i] =
                    i % 2 == 0 ? magnitudes[row] : -magnitudes[row];
        std::fill(update.begin(), update.end(), 0.f);
        std::fill(bias.begin(), bias.end(), 0.f);
    } else if (values == Values::Rounding) {
        const float x[]{-0.f, 0.f, 0x1p-133f, -0x1p-133f, 0x1p-126f, 1.f, -1.f, 0.9921875f};
        for (std::size_t i = 0; i < count; ++i) {
            residual[i] = x[i % 8];
            update[i]   = x[(i / 8) % 8];
        }
        for (int i = 0; i < d; ++i) gamma[i] = i & 1 ? -.5f : 0.50390625f;
    }
    for (auto* v : {&residual, &update, &weight, &bias}) round_to_bf16(*v);
    if (bf) round_to_bf16(gamma);
    auto dr = norm::make_input(residual, lead), du = norm::make_input(update, lead);
    auto dw = norm::make_input(weight, lead), db = norm::make_input(bias, lead);
    auto gamma_bf16 = norm::make_input(bf ? gamma : std::vector<float>{}, lead);
    if (!bf && natural) gamma.insert(gamma.begin(), -8.f);
    DeviceBuffer gamma_fp32;
    if (!bf) gamma_fp32 = to_device(gamma);
    if (!bf && natural) gamma.erase(gamma.begin());
    GuardedDeviceBuffer output((count + lead) * 2), normalized(fused ? (count + lead) * 2 : 0);
    Tensor r(dr.memory, DType::BF16, {d, m}), u(du.memory, DType::BF16, {d, m});
    Tensor g(bf ? gamma_bf16.memory : gamma_fp32.span().subspan(lead * sizeof(float)),
             bf ? DType::BF16 : DType::FP32, {d});
    Tensor w(dw.memory, DType::BF16, {d}), b(db.memory, DType::BF16, {d});
    Tensor out(alias ? dr.memory : output.span().subspan(lead * 2), DType::BF16, {d, m});
    Tensor norm;
    if (fused) norm = Tensor(normalized.span().subspan(lead * 2), DType::BF16, {d, m});
    auto& context        = cuda_environment().context;
    const auto execution = context.execution_view();
    const auto launch    = [&] {
        if (fused)
            ops::scaled_residual_add_layer_norm(r, u, g, w, b, eps, out, norm, execution);
        else
            ops::scaled_residual_add(r, u, g, out, execution);
    };
    DecodeGraphDefinition definition;
    DecodeGraphExecutable executable;
    int errors = 0;
    for (int phase = 0; phase < (graph ? 2 : 1); ++phase) {
        if (phase) {
            for (auto& value : residual) value = -value;
            for (std::size_t i = 0; i < count; ++i)
                dr.expected[lead + i] = f32_to_bf16(residual[i]);
            dr.storage.copy_from_host(dr.expected.data(), dr.storage.size());
        }
        output.fill(0xff);
        if (fused) normalized.fill(0xff);
        if (graph) {
            if (phase == 0) {
                definition.capture(context.stream(), launch);
                executable.instantiate(definition);
            }
            executable.launch(context.stream());
        } else
            launch();
        cuda_synchronize(context.stream());

        std::vector<std::uint16_t> exact(count);
        parallel_elements(count, [&](std::size_t i) {
            const float product  = float(double(update[i]) * gamma[i % d]);
            const double rounded = bf16_to_f32(f32_to_bf16(product));
            exact[i]             = f32_to_bf16(float(double(residual[i]) + rounded));
        });
        errors += verify_exact("scaled residual BF16 boundaries",
                               from_device<std::uint16_t>(out.data, count), exact);
        if (fused) {
            errors +=
                verify_reduction("scaled residual LayerNorm", from_device_bf16(norm.data, count),
                                 norm_reference(exact, weight, bias, d, m, eps), kCriterion);
            errors += norm::verify_output_storage("normalized output", normalized, lead);
            errors += norm::verify_preserved("LayerNorm weight", dw);
            errors += norm::verify_preserved("LayerNorm bias", db);
        }
        if (!alias) {
            errors += norm::verify_preserved("residual input", dr);
            errors += norm::verify_output_storage("published residual", output, lead);
        } else if (lead) {
            const std::vector<std::uint16_t> prefix(lead, 0x5a5a);
            errors += verify_exact("in-place residual prefix",
                                   from_device<std::uint16_t>(dr.storage, lead), prefix);
        }
        errors += norm::verify_preserved("scaled residual update", du);
        if (bf)
            errors += norm::verify_preserved("BF16 gamma", gamma_bf16);
        else
            errors += verify_exact("FP32 gamma", from_device<float>(g.data, d), gamma);
    }
    return errors;
}

int contract() {
    auto& context        = cuda_environment().context;
    const auto execution = context.execution_view();
    DeviceBuffer buffer(execution.device(), 4096);
    Tensor r(buffer.span(), DType::BF16, {17, 2});
    Tensor u(buffer.span().subspan(256), DType::BF16, {17, 2});
    Tensor g(buffer.span().subspan(512), DType::FP32, {17});
    Tensor w(buffer.span().subspan(768), DType::BF16, {17});
    Tensor b(buffer.span().subspan(1024), DType::BF16, {17});
    Tensor out(buffer.span().subspan(1280), DType::BF16, {17, 2});
    Tensor y(buffer.span().subspan(1536), DType::BF16, {17, 2});
    int errors        = 0;
    const auto reject = [&](const auto& launch) {
        try {
            launch();
            ++errors;
        } catch (const std::invalid_argument&) {}
    };
    for (std::size_t offset : {2, 256, 512, 1281}) {
        Tensor invalid(buffer.span().subspan(offset), DType::BF16, {17, 2});
        reject([&] { ops::scaled_residual_add(r, u, g, invalid, execution); });
        reject([&] {
            ops::scaled_residual_add_layer_norm(r, u, g, w, b, kEps, invalid, y, execution);
        });
    }
    reject([&] { ops::scaled_residual_add(r, r, g, r, execution); });
    for (std::size_t offset : {0, 2, 256, 512, 768, 1024, 1280, 1282}) {
        Tensor invalid(buffer.span().subspan(offset), DType::BF16, {17, 2});
        reject([&] {
            ops::scaled_residual_add_layer_norm(r, u, g, w, b, kEps, out, invalid, execution);
        });
    }
    auto bad_w = w;
    bad_w.data = out.data;
    reject(
        [&] { ops::scaled_residual_add_layer_norm(r, u, g, bad_w, b, kEps, out, y, execution); });
    auto invalid = u;
    invalid.nb[1] += 2;
    reject([&] { ops::scaled_residual_add(r, invalid, g, out, execution); });
    invalid        = g;
    invalid.device = Device::cpu();
    reject([&] { ops::scaled_residual_add(r, u, invalid, out, execution); });
    invalid       = g;
    invalid.dtype = DType::I32;
    reject([&] { ops::scaled_residual_add(r, u, invalid, out, execution); });
    invalid = y.view({34});
    reject(
        [&] { ops::scaled_residual_add_layer_norm(r, u, g, w, b, kEps, out, invalid, execution); });
    for (float eps : {0.f, -1.f, std::numeric_limits<float>::infinity(),
                      std::numeric_limits<float>::quiet_NaN()})
        reject([&] { ops::scaled_residual_add_layer_norm(r, u, g, w, b, eps, out, y, execution); });

    r.ne[1] = u.ne[1] = out.ne[1] = y.ne[1] = 0;
    r.data = u.data = out.data = y.data = nullptr;
    for (const auto* param : {&g, &w, &b}) {
        auto bad_device = *param, bad_pointer = *param;
        bad_device.device = Device::cpu();
        bad_pointer.data  = nullptr;
        for (auto& bad : {bad_device, bad_pointer}) {
            if (param == &g) reject([&] { ops::scaled_residual_add(r, u, bad, out, execution); });
            reject([&] {
                ops::scaled_residual_add_layer_norm(r, u, param == &g ? bad : g,
                                                    param == &w ? bad : w, param == &b ? bad : b,
                                                    kEps, out, y, execution);
            });
        }
    }
    cudaGraph_t graph = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(context.stream(), cudaStreamCaptureModeThreadLocal));
    ops::scaled_residual_add(r, u, g, out, execution);
    ops::scaled_residual_add_layer_norm(r, u, g, w, b, kEps, out, y, execution);
    CUDA_CHECK(cudaStreamEndCapture(context.stream(), &graph));
    std::size_t nodes = 0;
    CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &nodes));
    CUDA_CHECK(cudaGraphDestroy(graph));
    return errors + (nodes != 0);
}

} // namespace

void add_scaled_residual_add_cases(OpSuite& suite) {
    struct Case {
        int d, m;
        bool fused, alias = false, natural = false, graph = false;
        Values values = Values::Random;
    };

    constexpr Case cases[]{
        {1, 1, true},
        {17, 3, true, true, true, true},
        {256, 4093, true},
        {1024, 257, true},
        {1024, 1021, true, true},
        {2304, 17, true, false, false, true},
        {4096, 3, true, true, false, false, Values::LargeMean},
        {4093, 17, true, false, true, false, Values::LargeMean},
        {4099, 3, true, true, false, true, Values::LargeMean},
        {256, 17, true, false, false, false, Values::Rounding},
        {1024, 17, false, true, false, true},
        {17, 257, false, false, true},
        {256, 17, false, false, false, false, Values::Rounding},
    };
    for (const auto c : cases)
        for (bool bf : {false, true}) {
            const auto id =
                std::string(c.fused ? "scaled-residual-add-layer-norm/" : "scaled-residual-add/") +
                (bf ? "bf16/" : "fp32/") + std::to_string(c.d) + "/" + std::to_string(c.m) +
                (c.alias ? "/inplace" : "/separate") +
                (c.values == Values::LargeMean ? "/large-mean" : "") +
                (c.values == Values::Rounding ? "/rounding" : "");
            suite.add(
                id,
                [=] { return run(c.d, c.m, bf, c.fused, c.alias, c.natural, c.graph, c.values); },
                {{"execution", c.graph ? "graph" : "eager"}});
        }
    for (int d : {1024, 4608}) {
        suite.add("scaled-residual-add-layer-norm/large-finite/" + std::to_string(d),
                  [=] { return run(d, 3, false, true, d > 4096, false, true, Values::Overflow); },
                  {{"execution", "graph"}});
    }
    suite.add("scaled-residual-add-layer-norm/large-finite-epsilon",
              [] { return run(2, 1, false, true, false, false, false, Values::LargeEpsilon); });
    suite.add("scaled-residual-add/contract", contract);
}
