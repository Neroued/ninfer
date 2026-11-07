#include "core/decode_graph.h"
#include "core/platform.h"
#include "ninfer/ops/linear_bias.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ops/linear/support.h"
#include "ops/quantized_weight.h"

#include <array>

namespace ninfer::test::linear_bias {
namespace {

int run(QType format, int n, int k, bool swiglu, ops::LinearPolicy policy,
        std::span<const std::size_t> selected, std::span<const linear_suite::Case> cases) {
    int maximum = 1;
    for (auto i : selected) maximum = std::max(maximum, cases[i].t);
    const int rows = swiglu ? n / 2 : n;
    quantized_weight::PatternedWeightOptions options;
    if (format == QType::NVFP4_MUL) {
        options.weight_scale = 0.031713F;
        options.input_scale  = 0.01931F;
    }
    const auto host = quantized_weight::make_patterned_weight(format, n, k, 3407U, options);
    std::vector<std::uint16_t> input(std::size_t(k) * maximum), bias(n);
    std::vector<std::vector<std::pair<int, double>>> active(maximum);
    for (int t = 0; t < maximum; ++t) {
        const int count = t == 0 ? k : 4;
        for (int j = 0; j < count; ++j) {
            const int column                   = t == 0 ? j : (t * 17 + j * 131) % k;
            input[std::size_t(t) * k + column] = f32_to_bf16(float((j * 7 + t * 3) % 31 - 15) / 16);
        }
        for (int j = 0; j < k; ++j) {
            const double value = bf16_to_f32(input[std::size_t(t) * k + j]);
            if (value != 0) active[t].emplace_back(j, value);
        }
    }
    for (int row = 0; row < n; ++row)
        bias[row] = f32_to_bf16(swiglu && row % 64 == 0 ? (row < rows ? -10.5F : 20.F)
                                                        : float(row % 33 - 16) / 32);
    std::vector<double> projection(std::size_t(n) * maximum);
    parallel_rows(n, std::uint64_t(n) * k * maximum, [&](int first, int last) {
        for (int row = first; row < last; ++row)
            for (int t = 0; t < maximum; ++t) {
                double value = 0;
                for (auto [column, x] : active[t])
                    value += quantized_weight::logical_weight_fp64(host, row, column) * x;
                projection[std::size_t(t) * n + row] = value;
            }
    });
    GuardedDeviceBuffer dw(host.payload.size()), dx(input.size() * 2), db(bias.size() * 2 + 2);
    dw.copy_from_host(host.payload.data(), host.payload.size());
    dx.copy_from_host(input.data(), input.size() * 2);
    const auto weight    = host.device_weight(dw.span());
    const auto bias_span = db.span().subspan(2, bias.size() * 2);
    Tensor b(bias_span, DType::BF16, {n});
    DeviceContext context;
    const auto execution         = context.execution_view();
    int failures                 = 0;
    bool checked_workspace_alias = false;
    if (format == QType::MXFP8 && policy == ops::LinearPolicy::AllowA8) {
        for (auto [first, last] : {std::pair{1, 64}, std::pair{33, 193}, std::pair{193, 769},
                                   std::pair{769, 1025}, std::pair{1, 4097}}) {
            const auto peak =
                ops::linear_bias_workspace_capacity_bytes(format, n, k, policy, first, last);
            for (int t = first; t <= last; ++t)
                if (ops::linear_bias_workspace_capacity_bytes(format, n, k, policy, t, t) > peak)
                    ++failures;
        }
    }
    for (auto i : selected) {
        const auto& c = cases[i];
        const auto capacity =
            swiglu ? ops::linear_swiglu_bias_workspace_capacity_bytes(format, n, k, policy, 1, c.t)
                   : ops::linear_bias_workspace_capacity_bytes(format, n, k, policy, 1, c.t);
        GuardedDeviceBuffer dy(std::size_t(rows) * c.t * 2), scratch(capacity);
        MemoryArena arena(scratch.span());
        Tensor x(dx.span(), DType::BF16, {k, c.t}), y(dy.span(), DType::BF16, {rows, c.t});
        const auto launch = [&] {
            if (swiglu)
                ops::linear_swiglu(x, weight, b, y, 10.F, policy, arena, execution);
            else
                ops::linear_bias(x, weight, b, y, policy, arena, execution);
        };
        if (format == QType::MXFP8 && capacity && !checked_workspace_alias) {
            checked_workspace_alias = true;
            MemoryArena overlapping(dx.span());
            try {
                ops::linear_bias(x, weight, b, y, policy, overlapping, execution);
                ++failures;
            } catch (const std::invalid_argument&) {}
            if (overlapping.used() != 0) ++failures;
        }
        DecodeGraphDefinition definition;
        DecodeGraphExecutable graph;
        cuda_check(cudaMemcpy(bias_span.data, bias.data(), bias.size() * 2, cudaMemcpyHostToDevice),
                   "bias upload");
        cuda_synchronize();
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
            dy.fill(0xff);
            cuda_synchronize();
            if (c.execution == linear_suite::Execution::Graph)
                graph.launch(execution.stream());
            else
                launch();
            cuda_synchronize(execution.stream());
            const auto bits = from_device<std::uint16_t>(dy.data(), std::size_t(rows) * c.t);
            std::vector<double> actual(bits.size()), expected(bits.size());
            for (std::size_t j = 0; j < bits.size(); ++j) {
                const int row = j % rows, token = j / rows;
                const double gate =
                    projection[std::size_t(token) * n + row] + bf16_to_f32(values[row]);
                actual[j] = bf16_to_f32(bits[j]);
                if (swiglu) {
                    const double g = std::min(gate, 10.0);
                    const double u = std::clamp(projection[std::size_t(token) * n + row + rows] +
                                                    bf16_to_f32(values[row + rows]),
                                                -10.0, 10.0);
                    expected[j]    = g / (1.0 + std::exp(-g)) * u;
                } else {
                    expected[j] = gate;
                }
            }
            const ReductionCriterion criterion =
                policy == ops::LinearPolicy::AllowA8
                    ? ReductionCriterion{0.04, 1.0 / 256, 0.06}
                    : ReductionCriterion{1.0 / 256, 1.0e-7, 2.0 / 256};
            failures += verify_reduction(c.id, actual, expected, criterion);
            failures += dy.verify_guards(c.id) + scratch.verify_guards(c.id);
            if (arena.used() != 0 || arena.peak_used() > capacity) ++failures;
        }
    }
    failures += dw.verify_guards("weight") + dx.verify_guards("input") + db.verify_guards("bias");
    failures += linear_common::verify_preserved(dw.data(), std::as_bytes(std::span(host.payload)),
                                                "weight");
    failures +=
        linear_common::verify_preserved(dx.data(), std::as_bytes(std::span(input)), "input");
    return failures;
}

void add(linear_suite::Suite& suite, QType format, int n, int k, bool swiglu,
         ops::LinearPolicy policy, std::span<const int> tokens) {
    std::vector<linear_suite::Case> cases;
    const auto label = swiglu ? "native-swiglu-bias" : "native-bias";
    for (int t : tokens)
        cases.push_back(linear_suite::make_case(linear_suite::format_name(format), n, k, t, policy,
                                                linear_suite::Execution::Eager, label));
    if (format == QType::MXFP8 && policy == ops::LinearPolicy::AllowA8) {
        for (int t : {1, 3, 4, 8, 9, 33, 65, 193, 385, 769, 1025, 4097})
            cases.push_back(linear_suite::make_case(linear_suite::format_name(format), n, k, t,
                                                    policy, linear_suite::Execution::Graph, label));
    } else {
        cases.push_back(linear_suite::make_case(linear_suite::format_name(format), n, k, 33, policy,
                                                linear_suite::Execution::Graph, label));
    }
    if (format == QType::NVFP4_MUL && swiglu)
        cases.push_back(linear_suite::make_case(linear_suite::format_name(format), n, k, 129,
                                                policy, linear_suite::Execution::Graph, label));
    if (format == QType::NVFP4_MUL && !swiglu)
        for (int t : {1, 97})
            cases.push_back(linear_suite::make_case(linear_suite::format_name(format), n, k, t,
                                                    policy, linear_suite::Execution::Graph, label));
    suite.add(std::move(cases), [=](auto selected, auto entries) {
        return run(format, n, k, swiglu, policy, selected, entries);
    });
}
} // namespace

void add_native_cases(linear_suite::Suite& suite) {
#if NINFER_TARGET_SPARK
    constexpr std::array tokens{1, 2, 7, 17, 129, 513};
    constexpr std::array mx_tokens{
        1,   2,   3,   4,   5,    7,    8,    9,    16,   17,   24,   31,   32,   33,  48,  63,
        64,  65,  96,  128, 129,  191,  192,  193,  255,  256,  257,  383,  384,  385, 511, 512,
        513, 767, 768, 769, 1023, 1024, 1025, 2047, 2048, 2049, 3072, 4095, 4096, 4097};
    add(suite, QType::MXFP8, 1024, 1024, false, ops::LinearPolicy::AllowA8, mx_tokens);
    add(suite, QType::MXFP8, 1024, 1024, false, ops::LinearPolicy::A16Only, std::array{17, 257});
    add(suite, QType::NVFP4_MUL, 1024, 4096, false, ops::LinearPolicy::A16Only,
        std::array{1, 2, 8, 16, 17, 32, 65, 96, 97, 192, 193, 384, 385, 768, 769, 1152, 1153,
                   4096});
    add(suite, QType::NVFP4_MUL, 8192, 1024, true, ops::LinearPolicy::A16Only,
        std::array{1, 2, 7, 8, 9, 16, 17, 64, 65, 128, 129, 513});
#endif
}

} // namespace ninfer::test::linear_bias
