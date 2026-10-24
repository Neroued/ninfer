#include "ninfer/ops/linear_bias.h"
#include "ninfer/ops/linear_split.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ops/linear/support.h"
#include "ops/direct_bf16_weight.h"
#include "core/decode_graph.h"
#include "core/platform.h"

#include <array>

using namespace ninfer;
using namespace ninfer::test;

namespace {
enum class Kind { Projection, Split, SwiGlu };
enum class Recipe { Random, Cancellation, NegativeGate, AlternateLimit };

constexpr ReductionCriterion kA16{1.0 / 256, 1.0e-7, 2.0 / 256};

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
         Recipe recipe = Recipe::Random, Kind kind = Kind::Projection) {
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
    for (int t : {17, 257})
        cases.push_back(linear_suite::make_case("bf16", n, k, t, ops::LinearPolicy::A16Only,
                                                linear_suite::Execution::Graph, label));
    suite.add(std::move(cases), [=](auto selected, auto entries) {
        return run(kind, n, k, recipe, selected, entries);
    });
}

int validate(Kind kind) {
    const int n    = kind == Kind::Projection ? 1024 : kind == Kind::Split ? 3072 : 8192;
    const int rows = kind == Kind::SwiGlu ? n / 2 : 1024;
    auto host      = direct_bf16_weight::make_patterned(n, 1024, 29);
    direct_bf16_weight::DeviceWeight storage(std::move(host));
    auto weight = storage.view();
    GuardedDeviceBuffer dx(2048), db(n * 2);
    std::vector<GuardedDeviceBuffer> buffers;
    std::vector<Tensor> outputs;
    for (int i = 0; i < (kind == Kind::Split ? 3 : 1); ++i) {
        buffers.emplace_back(rows * 2);
        outputs.emplace_back(buffers.back().span(), DType::BF16,
                             std::initializer_list<int>{rows, 1});
    }
    Tensor x(dx.span(), DType::BF16, {1024, 1}), bias(db.span(), DType::BF16, {n});
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
    const auto reject = [&] {
        try {
            call();
            ++failures;
        } catch (const std::invalid_argument&) {}
    };
    const auto original_bias = bias;
    bias.dtype               = DType::FP32;
    reject();
    bias       = original_bias;
    bias.nb[0] = 4;
    reject();
    bias        = original_bias;
    bias.device = Device::cpu();
    reject();
    bias                       = original_bias;
    const auto original_output = outputs[0];
    outputs[0].data            = bias.data;
    reject();
    outputs[0]    = original_output;
    weight.layout = QuantLayout::RowSplit;
    reject();
    weight = storage.view();
    if (kind == Kind::Split) {
        const auto original = outputs[1];
        outputs[1].data     = outputs[0].data;
        reject();
        outputs[1] = original;
    }
    if (kind == Kind::SwiGlu) {
        for (float invalid : {0.F, -1.F, std::numeric_limits<float>::infinity(),
                              std::numeric_limits<float>::quiet_NaN()}) {
            limit = invalid;
            reject();
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
#if NINFER_TARGET_SPARK
    for (auto kind : {Kind::Projection, Kind::Split, Kind::SwiGlu}) {
        const int n = kind == Kind::Projection ? 1024 : kind == Kind::Split ? 3072 : 8192;
        suite.add(linear_suite::make_case("bf16", n, 1024, 1, ops::LinearPolicy::A16Only,
                                          linear_suite::Execution::Eager, "bias-operands"),
                  [kind] { return validate(kind); });
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
