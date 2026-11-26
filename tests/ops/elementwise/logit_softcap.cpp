#include "ninfer/ops/logit_softcap.h"
#include "core/decode_graph.h"
#include "ops/op_tester.h"

#include <array>
#include <bit>
#include <limits>

using namespace ninfer;
using namespace ninfer::test;

namespace {

using bf16 = std::uint16_t;

template <class Word>
Word encode(float value) {
    if constexpr (std::is_same_v<Word, bf16>)
        return f32_to_bf16(value);
    else
        return std::bit_cast<std::uint32_t>(value);
}

template <class Word>
float decode(Word word) {
    if constexpr (std::is_same_v<Word, bf16>)
        return bf16_to_f32(word);
    else
        return std::bit_cast<float>(word);
}

template <class Word>
int check(const std::vector<Word>& input, const std::vector<Word>& actual, float cap) {
    constexpr unsigned limit = std::is_same_v<Word, bf16> ? 1 : 2;
    constexpr auto sign      = Word(Word(1) << (sizeof(Word) * 8 - 1));
    unsigned max_error       = 0;
    int bad                  = 0;
    for (std::size_t i = 0; i < input.size(); ++i) {
        const float x        = decode(input[i]);
        const float observed = decode(actual[i]);
        if (std::isnan(x)) {
            if (!std::isnan(observed)) {
                if (bad < 4) std::cerr << "softcap returned a number for NaN input\n";
                ++bad;
            }
            continue;
        }
        const Word expected  = encode<Word>(float(double(cap) * std::tanh(x / double(cap))));
        const unsigned error = expected >= actual[i] ? expected - actual[i] : actual[i] - expected;
        max_error            = std::max(max_error, error);
        if (std::isnan(observed) || std::isinf(observed) != std::isinf(decode(expected)) ||
            ((expected ^ actual[i]) & sign) || error > limit) {
            if (bad < 4)
                std::cerr << "softcap x=" << x << " cap=" << cap << " expected=" << decode(expected)
                          << " actual=" << decode(actual[i]) << " ulp=" << error << '\n';
            ++bad;
        }
    }
    std::cout << "  maximum softcap ULP error = " << max_error << '\n';
    return bad != 0;
}

template <class Word>
std::vector<Word> input_words(int n, float cap, bool exhaustive) {
    const std::array edges{
        -0.f,
        0.f,
        -1.e30f,
        1.e30f,
        -std::numeric_limits<float>::denorm_min(),
        std::numeric_limits<float>::denorm_min(),
        -0x1p-133f,
        0x1p-133f,
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        -std::numeric_limits<float>::min(),
        std::numeric_limits<float>::min(),
        std::nextafter(cap, 0.f),
        std::nextafter(cap, std::numeric_limits<float>::infinity()),
        std::numeric_limits<float>::quiet_NaN(),
        -std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::signaling_NaN(),
    };
    std::mt19937 rng(19371);
    std::vector<Word> words(n);
    for (int i = 0; i < n; ++i) {
        if constexpr (std::is_same_v<Word, bf16>) {
            if (exhaustive) {
                words[i] = static_cast<bf16>(i);
                continue;
            }
        }
        float x;
        if (std::size_t(i) < edges.size())
            x = edges[i];
        else if (i & 1) {
            // Include finite values over the complete exponent range.
            x = std::bit_cast<float>(std::uint32_t(rng()));
            if (!std::isfinite(x)) x = 0.f;
        } else {
            // Dense coverage near the polynomial interval and the approach to saturation.
            const double q = double(int(rng() % 4000001U) - 2000000) * 0.000002;
            x              = float(q * cap);
        }
        words[i] = encode<Word>(x);
    }
    return words;
}

template <class Word>
int run(int n, float cap, bool alias, bool natural, bool graph, bool exhaustive = false) {
    constexpr DType dtype     = std::is_same_v<Word, bf16> ? DType::BF16 : DType::FP32;
    auto input                = input_words<Word>(n, cap, exhaustive);
    const std::size_t leading = natural ? sizeof(Word) : 0;
    const std::size_t bytes   = input.size() * sizeof(Word);
    GuardedDeviceBuffer x(leading + bytes), y(leading + bytes);
    auto& context        = cuda_environment().context;
    const auto execution = context.execution_view();
    Tensor src(x.span().subspan(leading), dtype, {n});
    Tensor out((alias ? x.span() : y.span()).subspan(leading), dtype, {n});
    if (n % 16 == 0) {
        src = src.view({2, 2, 2, n / 8});
        out = out.view({2, 2, 2, n / 8});
    }
    auto launch = [&] { ops::logit_softcap(src, cap, out, execution); };
    DecodeGraphDefinition definition;
    DecodeGraphExecutable executable;
    int failures = 0;
    for (int phase = 0; phase < (graph ? 2 : 1); ++phase) {
        if (phase) {
            for (auto& word : input) word ^= Word(Word(1) << (sizeof(Word) * 8 - 1));
        }
        x.fill(0x5a);
        y.fill(0x5a);
        x.copy_from_host(input.data(), bytes, leading);
        if (graph) {
            if (phase == 0) {
                definition.capture(context.stream(), launch);
                executable.instantiate(definition);
            }
            executable.launch(context.stream());
        } else
            launch();
        cuda_synchronize(context.stream());
        failures += check(input, from_device<Word>(out.data, n), cap);
        if (!alias)
            failures +=
                verify_exact("softcap preserves input", from_device<Word>(src.data, n), input);
        if (leading) {
            failures +=
                verify_exact("softcap input prefix", from_device<std::uint8_t>(x.data(), leading),
                             std::vector<std::uint8_t>(leading, 0x5a));
            failures +=
                verify_exact("softcap output prefix", from_device<std::uint8_t>(y.data(), leading),
                             std::vector<std::uint8_t>(leading, 0x5a));
        }
        failures += x.verify_guards("softcap input") + y.verify_guards("softcap output");
    }
    return failures;
}

int contract() {
    auto& context        = cuda_environment().context;
    const auto execution = context.execution_view();
    DeviceBuffer storage(execution.device(), 512);
    Tensor x(storage.span(), DType::FP32, {17, 2});
    Tensor y(storage.span().subspan(256), DType::FP32, {17, 2});
    int failures      = 0;
    const auto reject = [&](const Tensor& input, Tensor out, float cap = 30.f) {
        try {
            ops::logit_softcap(input, cap, out, execution);
            ++failures;
        } catch (const std::invalid_argument&) {}
    };
    reject(x, Tensor(storage.span().subspan(4), DType::FP32, {17, 2}));
    reject(Tensor(storage.span().subspan(4), DType::FP32, {17, 2}), x);
    reject(x, Tensor(storage.span().subspan(258), DType::FP32, {17, 2}));
    reject(x, Tensor(storage.span().subspan(256), DType::BF16, {17, 2}));
    reject(x, y.view({34}));
    auto invalid = x;
    invalid.nb[1] += 4;
    reject(invalid, y);
    invalid        = x;
    invalid.device = Device::cpu();
    reject(invalid, y);
    invalid      = x;
    invalid.data = nullptr;
    reject(invalid, y);
    for (float cap : {0.f, -1.f, std::numeric_limits<float>::infinity(),
                      std::numeric_limits<float>::quiet_NaN()})
        reject(x, y, cap);

    Tensor empty      = Tensor::metadata(DType::FP32, {17, 1, 2});
    empty.ne[1]       = 0;
    cudaGraph_t graph = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(context.stream(), cudaStreamCaptureModeThreadLocal));
    ops::logit_softcap(empty, 30.f, empty, execution);
    CUDA_CHECK(cudaStreamEndCapture(context.stream(), &graph));
    std::size_t nodes = 0;
    CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &nodes));
    CUDA_CHECK(cudaGraphDestroy(graph));
    return failures + (nodes != 0);
}
} // namespace

void add_logit_softcap_cases(OpSuite& suite) {
    constexpr int sizes[]{1, 7, 15, 16, 17, 1021, 32003, 200058, 201024, 1600513};
    constexpr float caps[]{.5f, 15.f, 30.f, 50.f};
    for (std::size_t i = 0; i < std::size(sizes); ++i) {
        const int n      = sizes[i];
        const float cap  = caps[i % std::size(caps)];
        const bool graph = n == 17 || n == 1021;
        for (bool bf : {false, true}) {
            const bool alias     = bool(i & 1) != bf;
            const bool natural   = i == 4 || i == 6;
            const std::string id = std::string("logit-softcap/") + (bf ? "bf16/" : "fp32/") +
                                   std::to_string(n) + (alias ? "/inplace" : "/separate");
            suite.add(id,
                      [=] {
                          return bf ? run<bf16>(n, cap, alias, natural, graph)
                                    : run<std::uint32_t>(n, cap, alias, natural, graph);
                      },
                      {{"execution", graph ? "graph" : "eager"}});
        }
    }
    for (float cap : caps) {
        suite.add("logit-softcap/bf16/all-encodings/cap=" + std::to_string(cap),
                  [=] { return run<bf16>(65536, cap, false, false, false, true); });
        suite.add("logit-softcap/fp32/sampled-domain/cap=" + std::to_string(cap),
                  [=] { return run<std::uint32_t>(1 << 20, cap, false, false, false); });
    }
    suite.add("logit-softcap/extreme-caps", [] {
        int errors = 0;
        for (float cap : {std::numeric_limits<float>::denorm_min(), 0x1p-133f, 0x1p-125f, 0x1p125f,
                          std::numeric_limits<float>::max()}) {
            errors += run<bf16>(65536, cap, true, false, false, true);
            errors += run<std::uint32_t>(32003, cap, false, true, false);
        }
        return errors;
    });
    suite.add("logit-softcap/contract", contract);
}
