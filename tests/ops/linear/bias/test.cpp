#include "ninfer/ops/linear_bias.h"
#include "ninfer/ops/linear_split.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ops/linear/support.h"
#include "ops/direct_bf16_weight.h"
#include "core/decode_graph.h"
#include "core/platform.h"

#include <algorithm>
#include <array>

using namespace ninfer;
using namespace ninfer::test;

namespace ninfer::test::linear_bias {
void add_native_cases(linear_suite::Suite& suite);
}

namespace {
enum class Kind { Projection, Split, SwiGlu };
enum class Recipe { Random, Cancellation, NegativeGate, AlternateLimit };

constexpr ReductionCriterion kA16{1.0 / 256, 1.0e-7, 2.0 / 256};
constexpr std::array kDefaultGraphTokens{17, 257};

int run(Kind kind, int n, int k, Recipe recipe, std::span<const std::size_t> selected,
        std::span<const linear_suite::Case> cases) {
    const bool cancellation = recipe == Recipe::Cancellation;
    const float limit       = recipe == Recipe::AlternateLimit ? 3.F : 10.F;
    int maximum             = 1;
    for (auto i : selected) maximum = std::max(maximum, cases[i].t);
    auto host = direct_bf16_weight::make_patterned(n, k, 19);
    std::vector<std::uint16_t> input(std::size_t(k) * maximum), bias(n);
    std::vector<std::vector<std::pair<int, double>>> active(maximum);
    for (int t = 0; t < maximum; ++t) {
        const int count = t == 0 ? k : 4;
        for (int j = 0; j < count; ++j) {
            const int column                   = t == 0 ? j : (t * 17 + j * 131) % k;
            input[std::size_t(t) * k + column] = f32_to_bf16(float((j * 7 + t * 3) % 31 - 15) / 16);
        }
        if (cancellation) {
            std::fill(input.begin() + std::size_t(t) * k, input.begin() + std::size_t(t + 1) * k,
                      0);
            input[std::size_t(t) * k] = input[std::size_t(t) * k + 1] = f32_to_bf16(1);
        }
        for (int j = 0; j < k; ++j) {
            const double value = bf16_to_f32(input[std::size_t(t) * k + j]);
            if (value != 0) active[t].emplace_back(j, value);
        }
    }
    for (int row = 0; row < n; ++row) {
        bias[row] = f32_to_bf16(float(row % 65 - 32) / 2);
        if (cancellation) {
            std::fill(host.bits.begin() + std::size_t(row) * k,
                      host.bits.begin() + std::size_t(row + 1) * k, 0);
            host.bits[std::size_t(row) * k]     = f32_to_bf16(1);
            host.bits[std::size_t(row) * k + 1] = f32_to_bf16(1.0F / 256);
            bias[row]                           = f32_to_bf16(-1);
        } else if (recipe == Recipe::NegativeGate) {
            std::fill(host.bits.begin() + std::size_t(row) * k,
                      host.bits.begin() + std::size_t(row + 1) * k, 0);
            bias[row] = f32_to_bf16(row < n / 2 ? -10.5F : 20.F);
        }
    }
    std::vector<double> reference(std::size_t(n) * maximum);
    parallel_rows(n, std::uint64_t(n) * k * maximum, [&](int first, int last) {
        for (int row = first; row < last; ++row)
            for (int t = 0; t < maximum; ++t) {
                double dot = 0;
                for (auto [column, value] : active[t])
                    dot += bf16_to_f32(host.bits[std::size_t(row) * k + column]) * value;
                reference[std::size_t(t) * n + row] = dot + bf16_to_f32(bias[row]);
            }
    });
    direct_bf16_weight::DeviceWeight weight(std::move(host));
    GuardedDeviceBuffer dx(input.size() * 2), db(bias.size() * 2 + 2);
    dx.copy_from_host(input.data(), input.size() * 2);
    // Bias requires natural alignment, including a two-byte-offset view.
    auto bias_span = db.span().subspan(2, bias.size() * 2);
    cuda_check(cudaMemcpy(bias_span.data, bias.data(), bias.size() * 2, cudaMemcpyHostToDevice),
               "bias upload");
    Tensor b(bias_span, DType::BF16, {n});
    DeviceContext context;
    const auto execution = context.execution_view();
    int failures         = 0;
    for (auto i : selected) {
        const auto& c         = cases[i];
        const int output_rows = kind == Kind::SwiGlu ? n / 2 : n;
        const int segments    = kind == Kind::Split ? 3 : 1;
        std::vector<GuardedDeviceBuffer> buffers;
        std::vector<Tensor> outputs;
        for (int s = 0; s < segments; ++s) {
            buffers.emplace_back(std::size_t(output_rows / segments) * c.t * 2);
            outputs.emplace_back(buffers.back().span(), DType::BF16,
                                 std::initializer_list<int>{output_rows / segments, c.t});
        }
        Tensor x(dx.span(), DType::BF16, {k, c.t});
        const auto launch = [&] {
            if (kind == Kind::Projection)
                ops::linear_bias(x, weight.view(), b, outputs[0], execution);
            else if (kind == Kind::Split)
                ops::linear_split(x, weight.view(), b, outputs, execution);
            else
                ops::linear_swiglu(x, weight.view(), b, outputs[0], limit, execution);
        };
        DecodeGraphDefinition definition;
        DecodeGraphExecutable graph;
        cuda_synchronize();
        launch();
        cuda_synchronize(execution.stream());
        if (c.execution == linear_suite::Execution::Graph) {
            definition.capture(execution.stream(), launch);
            graph.instantiate(definition);
        }
        for (int phase = 0; phase < (c.execution == linear_suite::Execution::Graph ? 2 : 1);
             ++phase) {
            auto values = bias;
            if (phase)
                for (auto& value : values) value ^= 0x8000;
            cuda_check(cudaMemcpy(bias_span.data, values.data(), values.size() * 2,
                                  cudaMemcpyHostToDevice),
                       "update bias");
            for (auto& buffer : buffers) buffer.fill(0xff);
            cuda_synchronize();
            if (c.execution == linear_suite::Execution::Graph)
                graph.launch(execution.stream());
            else
                launch();
            cuda_synchronize(execution.stream());
            for (int s = 0; s < segments; ++s) {
                const int rows = output_rows / segments;
                const auto bits =
                    from_device<std::uint16_t>(buffers[s].data(), std::size_t(rows) * c.t);
                std::vector<double> actual(bits.size()), expected(bits.size());
                for (std::size_t j = 0; j < bits.size(); ++j) {
                    const int row         = s * rows + j % rows;
                    const int token       = j / rows;
                    const auto projection = [&](int r) {
                        return reference[std::size_t(token) * n + r] +
                               (phase ? -2.0 * bf16_to_f32(bias[r]) : 0.0);
                    };
                    actual[j] = bf16_to_f32(bits[j]);
                    if (kind == Kind::SwiGlu) {
                        const double gate = std::min(projection(row), double(limit));
                        const double up =
                            std::clamp(projection(row + n / 2), -double(limit), double(limit));
                        expected[j] = gate / (1 + std::exp(-gate)) * up;
                    } else
                        expected[j] = projection(row);
                }
                failures += verify_reduction(c.id, actual, expected, kA16);
                failures += buffers[s].verify_guards(c.id);
            }
            failures += linear_common::verify_preserved(bias_span.data,
                                                        std::as_bytes(std::span(values)), "bias");
        }
        cuda_check(cudaMemcpy(bias_span.data, bias.data(), bias.size() * 2, cudaMemcpyHostToDevice),
                   "restore bias");
    }
    failures +=
        dx.verify_guards("input") + db.verify_guards("bias") + weight.verify_preserved("weight");
    failures +=
        linear_common::verify_preserved(dx.data(), std::as_bytes(std::span(input)), "input");
    return failures;
}

void add(linear_suite::Suite& suite, int n, int k, std::span<const int> tokens,
         Recipe recipe = Recipe::Random, Kind kind = Kind::Projection,
         std::span<const int> graph_tokens = kDefaultGraphTokens) {
    std::vector<linear_suite::Case> cases;
    const std::string label = std::string(kind == Kind::Split    ? "split-bias"
                                          : kind == Kind::SwiGlu ? "swiglu-bias"
                                                                 : "bias") +
                              (recipe == Recipe::Cancellation     ? "-cancellation"
                               : recipe == Recipe::NegativeGate   ? "-negative-gate"
                               : recipe == Recipe::AlternateLimit ? "-limit3"
                                                                  : "");
    for (int t : tokens)
        cases.push_back(linear_suite::make_case("bf16", n, k, t, ops::LinearPolicy::A16Only,
                                                linear_suite::Execution::Eager, label));
    for (int t : graph_tokens)
        cases.push_back(linear_suite::make_case("bf16", n, k, t, ops::LinearPolicy::A16Only,
                                                linear_suite::Execution::Graph, label));
    suite.add(std::move(cases), [=](auto selected, auto entries) {
        return run(kind, n, k, recipe, selected, entries);
    });
}

int validate(Kind kind, int n, int k) {
    const int rows = kind == Kind::SwiGlu ? n / 2 : kind == Kind::Split ? n / 3 : n;
    auto host      = direct_bf16_weight::make_patterned(n, k, 29);
    direct_bf16_weight::DeviceWeight storage(std::move(host));
    auto weight = storage.view();
    GuardedDeviceBuffer dx(k * 2), db(n * 2);
    std::vector<GuardedDeviceBuffer> buffers;
    std::vector<Tensor> outputs;
    for (int i = 0; i < (kind == Kind::Split ? 3 : 1); ++i) {
        buffers.emplace_back(rows * 2);
        outputs.emplace_back(buffers.back().span(), DType::BF16,
                             std::initializer_list<int>{rows, 1});
    }
    Tensor x(dx.span(), DType::BF16, {k, 1}), bias(db.span(), DType::BF16, {n});
    auto execution  = cuda_environment().default_execution;
    float limit     = 10.F;
    const auto call = [&] {
        if (kind == Kind::Projection)
            ops::linear_bias(x, weight, bias, outputs[0], execution);
        else if (kind == Kind::Split)
            ops::linear_split(x, weight, bias, outputs, execution);
        else
            ops::linear_swiglu(x, weight, bias, outputs[0], limit, execution);
    };
    int failures      = 0;
    const auto reject = [&](const auto& invalid_call) {
        try {
            invalid_call();
            ++failures;
        } catch (const std::invalid_argument&) {}
    };
    const auto original_bias = bias;
    bias.dtype               = DType::FP32;
    reject(call);
    bias       = original_bias;
    bias.nb[0] = 4;
    reject(call);
    bias        = original_bias;
    bias.device = Device::cpu();
    reject(call);
    bias                       = original_bias;
    const auto original_output = outputs[0];
    outputs[0].data            = bias.data;
    reject(call);
    outputs[0]    = original_output;
    weight.layout = QuantLayout::RowSplit;
    reject(call);
    weight = storage.view();
    if (kind == Kind::Projection) {
        const auto capacity = [=](int first, int last) {
            return ops::linear_bias_workspace_capacity_bytes(
                QType::BF16, n, k, ops::LinearPolicy::A16Only, first, last);
        };
        if (capacity(1, std::numeric_limits<int>::max()) != 0) {
            std::cerr << "BF16 bias: expected zero workspace\n";
            ++failures;
        }
        reject([&] { (void)capacity(0, 1); });
        reject([&] { (void)capacity(2, 1); });
    }
    if (kind == Kind::Split) {
        const auto second = outputs[1];
        outputs[1].data   = outputs[0].data;
        reject(call);
        outputs[1]      = second;
        outputs[0].data = x.data;
        reject(call);
        outputs[0].data = const_cast<void*>(weight.qdata);
        reject(call);
        outputs[0]       = original_output;
        outputs[1].ne[0] = rows - 16;
        reject(call);
        outputs[1] = second;
        x.ne[1]    = 0;
        reject(call);
        x.ne[1]    = 1;
        bias.ne[0] = n - 1;
        reject(call);
        bias = original_bias;

        constexpr auto policy = ops::LinearPolicy::A16Only;
        const std::array segments{rows, rows, rows};
        const auto capacity = [&](int first, int last) {
            return ops::linear_split_bias_workspace_capacity_bytes(QType::BF16, n, k, segments,
                                                                   policy, first, last);
        };
        if (capacity(1, std::numeric_limits<int>::max()) != 0) {
            std::cerr << "BF16 split bias: expected zero workspace capacity\n";
            ++failures;
        }
        reject([&] { (void)capacity(0, 1); });
        reject([&] { (void)capacity(2, 1); });
        reject([&] {
            (void)ops::linear_split_bias_workspace_capacity_bytes(
                QType::BF16, n, k, std::array{rows - 16, rows + 16, rows}, policy, 1, 1);
        });
#if NINFER_TARGET_RTX5090
        reject([&] {
            (void)ops::linear_split_bias_workspace_capacity_bytes(QType::MXFP8, n, k, segments,
                                                                  policy, 1, 1);
        });
#endif
        MemoryArena cpu_workspace({nullptr, 0, Device::cpu()});
        reject(
            [&] { ops::linear_split(x, weight, bias, outputs, policy, cpu_workspace, execution); });
        MemoryArena empty({nullptr, 0, execution.device()});
        ops::linear_split(x, weight, bias, outputs, policy, empty, execution);
        if (empty.used() != 0 || empty.peak_used() != 0) {
            std::cerr << "BF16 split bias: expected zero workspace use\n";
            ++failures;
        }
    }
    if (kind == Kind::SwiGlu) {
        for (float invalid : {0.F, -1.F, std::numeric_limits<float>::infinity(),
                              std::numeric_limits<float>::quiet_NaN()}) {
            limit = invalid;
            reject(call);
        }
    }
    cuda_synchronize();
    for (auto& buffer : buffers) failures += buffer.verify_guards("invalid call");
    if (failures) std::cerr << "bias operands: invalid call was accepted\n";
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    linear_suite::Suite suite;
    linear_bias::add_native_cases(suite);
#if NINFER_TARGET_RTX5090
    constexpr int n = 768, k = 2304;
    suite.add(linear_suite::make_case("bf16", n, k, 1, ops::LinearPolicy::A16Only,
                                      linear_suite::Execution::Eager, "bias-operands"),
              [] { return validate(Kind::Projection, n, k); });
    std::vector<int> tokens{1, 4, 8, 128, 512, 1023, 1024, 1025, 2048, 4096, 4097};
    for (int boundary :
         {2, 25, 49, 81, 225, 385, 641, 897, 1121, 1345, 1793, 2241, 2689, 3137, 3585}) {
        for (int t : {boundary - 1, boundary, boundary + 1}) tokens.push_back(t);
    }
    std::sort(tokens.begin(), tokens.end());
    tokens.erase(std::unique(tokens.begin(), tokens.end()), tokens.end());
    add(suite, n, k, tokens, Recipe::Random, Kind::Projection, tokens);

    const std::array cancellation{1,    8,    32,   64,   128,  256,  512, 768,
                                  1024, 1280, 1536, 2048, 2560, 3072, 4097};
    add(suite, n, k, cancellation, Recipe::Cancellation, Kind::Projection, cancellation);

    const auto add_vision = [&](Kind kind, int rows, int hidden,
                                std::initializer_list<int> boundaries) {
        suite.add(linear_suite::make_case("bf16", rows, hidden, 1, ops::LinearPolicy::A16Only,
                                          linear_suite::Execution::Eager, "bias-operands"),
                  [=] { return validate(kind, rows, hidden); });

        std::vector<int> points{1, 4, 8, 128, 512, 1023, 1024, 1025, 2048, 4096, 4097};
        for (int t : boundaries) points.insert(points.end(), {t - 1, t, t + 1});
        std::sort(points.begin(), points.end());
        points.erase(std::unique(points.begin(), points.end()), points.end());
        add(suite, rows, hidden, points, Recipe::Random, kind, points);

        std::vector<int> cancel{1};
        cancel.insert(cancel.end(), boundaries.begin(), boundaries.end());
        cancel.push_back(4097);
        add(suite, rows, hidden, cancel, Recipe::Cancellation, kind, cancel);
    };
    add_vision(Kind::Split, 3072, 1024,
               {2, 17, 49, 81, 145, 193, 337, 449, 561, 673, 785, 897, 1121, 1345, 1569, 1793, 1921, 2017, 2241});
    add_vision(Kind::Projection, 1024, 1024,
               {2, 113, 321, 641, 801, 1009, 1177, 1345, 1681, 2017, 2353, 2689, 3361, 4033});
    add_vision(Kind::Projection, 4096, 1024,
               {2, 48, 64, 97, 161, 241, 321, 401, 481, 561, 641, 801, 961, 1041, 1457, 1601});
    add_vision(
        Kind::Projection, 1024, 4096,
        {2, 35, 73, 129, 257, 481, 641, 801, 1009, 1177, 1345, 1601, 2017, 2353, 2689, 3361, 4033});
#endif
#if NINFER_TARGET_SPARK
    for (auto kind : {Kind::Projection, Kind::Split, Kind::SwiGlu}) {
        const int n = kind == Kind::Projection ? 1024 : kind == Kind::Split ? 3072 : 8192;
        suite.add(linear_suite::make_case("bf16", n, 1024, 1, ops::LinearPolicy::A16Only,
                                          linear_suite::Execution::Eager, "bias-operands"),
                  [kind, n] { return validate(kind, n, 1024); });
    }
    add(suite, 1024, 1176,
        std::array{1,   4,   8,   15,  16,  17,  31,  32,   33,   127,  128,  129,  191,  192,
                   193, 255, 256, 257, 511, 512, 513, 1535, 1536, 1537, 2303, 2304, 2305, 4096});
    add(suite, 1024, 1024, std::array{1,   4,   7,   8,    9,    15,   16,   17,   31,   32,
                                      33,  63,  64,  65,   255,  256,  257,  383,  384,  385,
                                      767, 768, 769, 1535, 1536, 1537, 2303, 2304, 2305, 4096});
    add(suite, 1024, 4096,
        std::array{1,   4,   7,   8,   9,    15,   16,   17,   31,   32,   33,
                   63,  64,  65,  127, 128,  129,  255,  256,  257,  383,  384,
                   385, 767, 768, 769, 1535, 1536, 1537, 2303, 2304, 2305, 4096});
    add(suite, 4096, 4096, std::array{1,   4,   8,   15,  16,  17,  31,  32,   33,   191,  192,
                                      193, 255, 256, 257, 511, 512, 513, 2559, 2560, 2561, 4096});
    add(suite, 1024, 1176, std::array{1, 8, 33, 257}, Recipe::Cancellation);
    add(suite, 3072, 1024, std::array{1,  4,  7,   8,   9,   15,  16,  17,  31,   32,   33,   63,
                                      64, 65, 127, 128, 129, 255, 256, 257, 2047, 2048, 2049, 4096},
        Recipe::Random, Kind::Split);
    add(suite, 3072, 1024, std::array{1, 8, 33, 257}, Recipe::Cancellation, Kind::Split);
    add(suite, 8192, 1024, std::array{1,  4,  7,  8,   9,   15,  16,  17,  31,  32,   33,
                                      63, 64, 65, 127, 128, 129, 511, 512, 513, 2048, 4096},
        Recipe::Random, Kind::SwiGlu);
    add(suite, 8192, 1024, std::array{1, 8, 33, 257}, Recipe::Cancellation, Kind::SwiGlu);
    add(suite, 8192, 1024, std::array{1, 8, 33, 513}, Recipe::NegativeGate, Kind::SwiGlu);
    add(suite, 8192, 1024, std::array{8, 33}, Recipe::AlternateLimit, Kind::SwiGlu);
#endif
    return suite.main(argc, argv);
}
