#include "ninfer/ops/weighted_sum_rmsnorm.h"
#include "core/decode_graph.h"
#include "ops/normalization/support.h"

#include <limits>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr float kEps = 1.0e-5f;
// The FP64 oracle is unrounded. Even a single correct BF16 result can incur
// nearly 2^-8 relative error from its final rounding.
constexpr ReductionCriterion kCriterion{4.e-3, 3.e-6, 4.1e-3};

std::vector<double> reference(const std::vector<float>& input, const std::vector<float>& alphas,
                              const std::vector<float>& gain, int d, int k, int m, float eps) {
    std::vector<double> result(std::size_t(d) * m);
    parallel_rows(m, result.size(), [&](int first, int last) {
        for (int row = first; row < last; ++row) {
            const auto base = std::size_t(row) * d;
            double sum      = 0;
            for (int channel = 0; channel < d; ++channel) {
                double mixed = 0;
                for (int tap = 0; tap < k; ++tap)
                    mixed += double(alphas[tap]) *
                             double(input[(std::size_t(row) * k + tap) * d + channel]);
                result[base + channel] = mixed;
                sum += mixed * mixed;
            }
            const double inverse = 1. / std::sqrt(sum / d + eps);
            for (int channel = 0; channel < d; ++channel)
                result[base + channel] *= inverse * gain[channel];
        }
    });
    return result;
}

int run(int d, int k, int m, bool bf, bool natural, bool graph, bool cancellation = false) {
    const auto count       = std::size_t(d) * m;
    const std::size_t lead = natural ? 1 : 0;
    const float eps        = cancellation ? 1.e-8f : kEps;
    std::vector<float> inputs(count * k), alphas(k), weight(d);
    fill_uniform(inputs, 2411, -2.f, 2.f);
    fill_uniform(alphas, 2417, -1.f, 1.f);
    fill_uniform(weight, 2423, -.75f, 1.5f);
    if (cancellation) {
        alphas = {1.f, -1.f, 0x1p-13f};
        for (int row = 0; row < m; ++row) {
            const auto base = std::size_t(row) * k * d;
            std::copy_n(inputs.begin() + base, d, inputs.begin() + base + d);
        }
    }
    for (auto* v : {&inputs, &weight}) round_to_bf16(*v);
    if (bf) round_to_bf16(alphas);
    auto x_dev   = norm::make_input(inputs, lead);
    auto w_dev   = norm::make_input(weight, lead);
    auto a_words = bf16_words(alphas);
    auto a_dev   = bf ? to_device(a_words) : to_device(alphas);
    GuardedDeviceBuffer y_dev((count + lead) * 2);
    Tensor x(x_dev.memory, DType::BF16, {d, k, m});
    Tensor a(a_dev.span(), bf ? DType::BF16 : DType::FP32, {k});
    Tensor w(w_dev.memory, DType::BF16, {d});
    Tensor y(y_dev.span().subspan(lead * 2), DType::BF16, {d, m});

    auto& context        = cuda_environment().context;
    const auto execution = context.execution_view();
    const auto launch    = [&] { ops::weighted_sum_rmsnorm(x, a, w, eps, y, execution); };
    DecodeGraphDefinition definition;
    DecodeGraphExecutable executable;
    int errors = 0;
    for (int phase = 0; phase < (graph ? 2 : 1); ++phase) {
        if (phase) {
            std::reverse(alphas.begin(), alphas.end());
            for (auto& alpha : alphas) alpha = -alpha;
            a_words = bf16_words(alphas);
            a_dev.copy_from_host(bf ? static_cast<const void*>(a_words.data()) : alphas.data(),
                                 a_dev.size());
        }
        y_dev.fill(0xff);
        if (graph) {
            if (phase == 0) {
                definition.capture(context.stream(), launch);
                executable.instantiate(definition);
            }
            executable.launch(context.stream());
        } else
            launch();
        cuda_synchronize(context.stream());
        errors += verify_reduction("weighted RMSNorm", from_device_bf16(y.data, count),
                                   reference(inputs, alphas, weight, d, k, m, eps), kCriterion);
        errors += norm::verify_preserved("weighted RMSNorm input", x_dev);
        errors += norm::verify_preserved("weighted RMSNorm gain", w_dev);
        if (bf)
            errors += verify_exact("weighted RMSNorm alphas", from_device<std::uint16_t>(a_dev, k),
                                   a_words);
        else
            errors += verify_exact("weighted RMSNorm alphas", from_device<float>(a_dev, k), alphas);
        errors += norm::verify_output_storage("weighted RMSNorm output", y_dev, lead);
    }
    return errors;
}

int extreme_input(int d, float value, bool cancel, float eps = kEps) {
    const float alpha = 0x1.000002p0f;
    const auto alphas = cancel ? std::vector<float>{alpha, -alpha} : std::vector<float>{1.f};
    const int k       = static_cast<int>(alphas.size());
    const std::vector<float> input(std::size_t(d) * k, value), gain(d, 1.f);
    auto dx = to_device_bf16(input);
    auto da = to_device(alphas);
    auto dw = to_device_bf16(gain);
    GuardedDeviceBuffer dy(std::size_t(d) * 2);
    dy.fill(0xff);
    Tensor x(dx.span(), DType::BF16, {d, k, 1}), a(da.span(), DType::FP32, {k});
    Tensor w(dw.span(), DType::BF16, {d}), y(dy.span(), DType::BF16, {d, 1});
    auto& context = cuda_environment().context;
    ops::weighted_sum_rmsnorm(x, a, w, eps, y, context.execution_view());
    cuda_synchronize(context.stream());
    const auto expected = reference(input, alphas, gain, d, k, 1, eps);
    int failures;
    if (cancel) {
        std::vector<std::uint16_t> words(d);
        std::transform(expected.begin(), expected.end(), words.begin(),
                       [](double v) { return f32_to_bf16(float(v)); });
        failures = verify_exact("opposite taps cancel to zero",
                                from_device<std::uint16_t>(y.data, d), words);
    } else {
        failures =
            verify_reduction("finite large RMS", from_device_bf16(y.data, d), expected, kCriterion);
    }
    return failures + dy.verify_guards("extreme RMS output");
}

int contract() {
    auto& context        = cuda_environment().context;
    const auto execution = context.execution_view();
    DeviceBuffer buffer(execution.device(), 4096);
    Tensor x(buffer.span(), DType::BF16, {17, 3, 2});
    Tensor a(buffer.span().subspan(256), DType::FP32, {3});
    Tensor w(buffer.span().subspan(512), DType::BF16, {17});
    Tensor y(buffer.span().subspan(768), DType::BF16, {17, 2});
    int errors        = 0;
    const auto reject = [&](const Tensor& xx, const Tensor& aa, const Tensor& ww, Tensor yy,
                            float eps = kEps) {
        try {
            ops::weighted_sum_rmsnorm(xx, aa, ww, eps, yy, execution);
            ++errors;
        } catch (const std::invalid_argument&) {}
    };
    for (std::size_t offset : {0, 2, 256, 512})
        reject(x, a, w, Tensor(buffer.span().subspan(offset), DType::BF16, {17, 2}));
    reject(x, a, w, Tensor(buffer.span().subspan(769), DType::BF16, {17, 2}));
    reject(x, a, w, y.view({34}));
    reject(x, w, w, y);
    reject(x, a, Tensor(w.data, w.device, DType::FP32, {17}), y);
    auto invalid = x;
    invalid.nb[2] += 2;
    reject(invalid, a, w, y);
    invalid       = x;
    invalid.ne[1] = 0;
    reject(invalid, a, w, y);
    invalid.ne[0] = 0;
    reject(invalid, a, w, y);
    invalid        = a;
    invalid.device = Device::cpu();
    reject(x, invalid, w, y);
    for (float eps : {0.f, -1.f, std::numeric_limits<float>::infinity(),
                      std::numeric_limits<float>::quiet_NaN()})
        reject(x, a, w, y, eps);

    x.ne[2] = y.ne[1] = 0;
    x.data = y.data = nullptr;
    for (const auto* param : {&a, &w}) {
        auto bad_device = *param, bad_pointer = *param, bad_stride = *param;
        bad_device.device = Device::cpu();
        bad_pointer.data  = nullptr;
        bad_stride.nb[0] *= 2;
        for (const auto& bad : {bad_device, bad_pointer, bad_stride})
            if (param == &a)
                reject(x, bad, w, y);
            else
                reject(x, a, bad, y);
    }
    cudaGraph_t graph = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(context.stream(), cudaStreamCaptureModeThreadLocal));
    ops::weighted_sum_rmsnorm(x, a, w, kEps, y, execution);
    CUDA_CHECK(cudaStreamEndCapture(context.stream(), &graph));
    std::size_t nodes = 0;
    CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &nodes));
    CUDA_CHECK(cudaGraphDestroy(graph));
    return errors + (nodes != 0);
}

} // namespace

void add_weighted_sum_rmsnorm_cases(OpSuite& suite) {
    struct Case {
        int d, k, m;
        bool natural = false, graph = false, cancellation = false;
    };

    constexpr Case cases[]{
        {1, 1, 1},
        {17, 2, 3, true, true},
        {256, 3, 4093},
        {1024, 5, 1021},
        {2304, 4, 1},
        {2304, 4, 7, false, true},
        {4096, 8, 63},
        {256, 13, 7},
        {4099, 13, 3},
        {1024, 3, 7, true},
        {256, 3, 3, false, false, true},
    };
    for (const auto c : cases)
        for (bool bf : {false, true}) {
            const auto id = std::string("weighted-sum-rmsnorm/") + (bf ? "bf16/" : "fp32/") +
                            std::to_string(c.d) + "/" + std::to_string(c.k) + "/" +
                            std::to_string(c.m) + (c.natural ? "/natural" : "") +
                            (c.cancellation ? "/cancellation" : "");
            suite.add(id,
                      [=] { return run(c.d, c.k, c.m, bf, c.natural, c.graph, c.cancellation); },
                      {{"execution", c.graph ? "graph" : "eager"}});
        }
    for (int d : {1, 256, 4608}) {
        suite.add("weighted-sum-rmsnorm/exact-cancellation/" + std::to_string(d),
                  [=] { return extreme_input(d, 96.f, true); });
    }
    for (int d : {1024, 4608})
        for (int exponent : {60, 120}) {
            suite.add("weighted-sum-rmsnorm/large-finite/" + std::to_string(d) + "/" +
                          std::to_string(exponent),
                      [=] { return extreme_input(d, std::ldexp(1.f, exponent), false); });
        }
    suite.add("weighted-sum-rmsnorm/large-finite-epsilon",
              [] { return extreme_input(1, 0x1p63f, false, std::numeric_limits<float>::max()); });
    suite.add("weighted-sum-rmsnorm/contract", contract);
}
