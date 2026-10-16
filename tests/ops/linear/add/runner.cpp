#include "ops/linear/add/runner.h"
#include "cuda_fixture.h"

#include "core/decode_graph.h"
#include "core/device.h"
#include "ops/direct_bf16_weight.h"
#include "ops/linear_common.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

namespace ninfer::test::linear_add {
namespace {

using linear_common::all_indices;
using linear_common::checked_elements;
using linear_common::sampled_indices;
using linear_common::verify_preserved;

constexpr std::size_t kOutputScanWords = 1U << 20;
constexpr double kBf16UnitRoundoff     = 1.0 / 256.0;
constexpr ReductionCriterion kA16Tolerance{
    kBf16UnitRoundoff,
    kBf16UnitRoundoff,
    2.0 * kBf16UnitRoundoff,
};
constexpr ReductionCriterion kA8Tolerance{0.04, kBf16UnitRoundoff, 0.06};
constexpr ReductionCriterion kA4Tolerance{0.16, kBf16UnitRoundoff, 0.16};

const ReductionCriterion& criterion(ops::LinearPolicy policy) {
    switch (policy) {
    case ops::LinearPolicy::A16Only:
        return kA16Tolerance;
    case ops::LinearPolicy::AllowA8:
        return kA8Tolerance;
    case ops::LinearPolicy::AllowA4:
        return kA4Tolerance;
    }
    throw std::invalid_argument("linear_add test: unknown Linear policy");
}

std::vector<std::int32_t> select_indices(std::int32_t extent, Samples samples) {
    if (samples == Samples::All) { return all_indices(extent); }
    if (samples == Samples::BoundarySeven) { return sampled_indices(extent); }
    return linear_common::evenly_spaced_indices(extent);
}

std::vector<std::int32_t> conformance_tokens(const ShapeCase& shape) {
    std::vector<std::int32_t> result{1};
    for (const std::int32_t boundary : shape.route_starts) {
        if (boundary <= 1) {
            throw std::invalid_argument("linear_add test: route starts must be greater than one");
        }
        result.push_back(boundary - 1);
        result.push_back(boundary);
        if (boundary != std::numeric_limits<std::int32_t>::max()) {
            result.push_back(boundary + 1);
        }
    }
    for (const std::int32_t interior : shape.route_interiors) {
        if (interior <= 0) {
            throw std::invalid_argument("linear_add test: route interior must be positive");
        }
        result.push_back(interior);
    }
    if (shape.composite_offset > 0) {
        for (const std::int32_t boundary : shape.route_starts) {
            result.push_back(shape.composite_offset + boundary - 1);
            result.push_back(shape.composite_offset + boundary);
            if (boundary != std::numeric_limits<std::int32_t>::max()) {
                result.push_back(shape.composite_offset + boundary + 1);
            }
        }
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

std::vector<std::uint16_t> make_activation(const InvocationCase& shape, std::int32_t t) {
    const std::uint32_t seed = shape.seed + 1U;
    std::vector<std::uint16_t> result(checked_elements(shape.k, t, "activation"));
    for (std::int32_t token = 0; token < t; ++token) {
        for (std::int32_t column = 0; column < shape.k; ++column) {
            float value;
            if (shape.fixture == FixturePattern::Hashed) {
                std::uint32_t hash = seed ^ (static_cast<std::uint32_t>(column) * 0x9e3779b9U) ^
                                     (static_cast<std::uint32_t>(token) * 0x85ebca6bU);
                hash ^= hash >> 16;
                hash *= 0x7feb352dU;
                hash ^= hash >> 15;
                value = static_cast<float>(static_cast<int>(hash & 0xffU) - 128) * (1.0F / 256.0F);
            } else {
                const auto token_block = static_cast<std::uint64_t>(token / 256);
                const std::uint64_t coordinate =
                    static_cast<std::uint64_t>(column) * 17U +
                    static_cast<std::uint64_t>(token) * 31U +
                    static_cast<std::uint64_t>(seed) * 13U +
                    token_block * (static_cast<std::uint64_t>(column / 256) * 13U + 47U);
                value =
                    static_cast<float>(32 + static_cast<int>(coordinate & 0x5fU)) * (1.0F / 256.0F);
            }
            result[static_cast<std::size_t>(token) * shape.k + column] = f32_to_bf16(value);
        }
    }
    return result;
}

std::vector<std::uint16_t> make_residual(const InvocationCase& shape, std::int32_t t) {
    const std::uint32_t seed = shape.seed + 2U;
    std::vector<std::uint16_t> result(checked_elements(shape.n, t, "residual"));
    for (std::int32_t token = 0; token < t; ++token) {
        for (std::int32_t row = 0; row < shape.n; ++row) {
            const auto token_block = shape.fixture == FixturePattern::TokenBlocks
                                         ? static_cast<std::uint64_t>(token / 256)
                                         : 0U;
            const std::uint64_t coordinate =
                static_cast<std::uint64_t>(row) * 23U + static_cast<std::uint64_t>(token) * 41U +
                static_cast<std::uint64_t>(seed) * 7U +
                token_block * (static_cast<std::uint64_t>(row / 256) * 17U + 29U);
            const int raw = static_cast<int>(coordinate & 0xffU);
            result[static_cast<std::size_t>(token) * shape.n + row] =
                f32_to_bf16(static_cast<float>(raw - 128) * (1.0F / 128.0F));
        }
    }
    return result;
}

using HostWeight = std::variant<quantized_weight::PackedWeight, direct_bf16_weight::HostWeight>;

HostWeight make_weight(QType format, const InvocationCase& shape) {
    if (format == QType::BF16) {
        return direct_bf16_weight::make_patterned(shape.n, shape.k, shape.seed);
    }
    quantized_weight::PatternedWeightOptions options;
    if (format == QType::NVFP4) {
        options.weight_scale_divisor = 0.125F;
        options.input_scale_divisor  = 3.5F;
    } else if (format != QType::FP8_E4M3FN_ROW_BF16) {
        options.row_split_scale = (format == QType::Q4_G64_FP16 || format == QType::Q5_G64_FP16)
                                      ? quantized_weight::RowSplitScalePattern::Small
                                      : quantized_weight::RowSplitScalePattern::Tiny;
        options.row_split_codes = quantized_weight::RowSplitCodePattern::Hashed;
    }
    return quantized_weight::make_patterned_weight(format, shape.n, shape.k, shape.seed, options);
}

std::span<const std::uint8_t> weight_payload(const HostWeight& weight) {
    return std::visit(
        [](const auto& value) -> std::span<const std::uint8_t> {
            using Value = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<Value, quantized_weight::PackedWeight>) {
                return value.payload;
            } else {
                return {reinterpret_cast<const std::uint8_t*>(value.bits.data()),
                        value.bits.size() * sizeof(std::uint16_t)};
            }
        },
        weight);
}

std::vector<float> materialize_weight_rows(const HostWeight& weight,
                                           std::span<const std::int32_t> rows) {
    return std::visit(
        [&](const auto& value) {
            using Value = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<Value, quantized_weight::PackedWeight>) {
                return quantized_weight::materialize_rows_fp32(value, rows);
            } else {
                std::vector<float> result(checked_elements(static_cast<std::int32_t>(rows.size()),
                                                           value.k, "oracle weight"));
                for (std::size_t selected = 0; selected < rows.size(); ++selected) {
                    for (std::int32_t column = 0; column < value.k; ++column) {
                        result[selected * value.k + column] = bf16_to_f32(
                            value
                                .bits[static_cast<std::size_t>(rows[selected]) * value.k + column]);
                    }
                }
                return result;
            }
        },
        weight);
}

// A dot product is reused only for the same logical row and token. The full fused oracle below
// adds the original residual in FP64, without an intermediate projection cast.
struct Oracle {
    std::vector<std::int32_t> rows;
    std::vector<std::int32_t> column_index;
    std::vector<double> dots;
};

Oracle make_oracle(const InvocationCase& shape, const HostWeight& weight,
                   std::span<const std::uint16_t> activation, std::int32_t maximum_t) {
    Oracle oracle;
    oracle.rows = select_indices(shape.n, shape.rows);
    oracle.column_index.resize(maximum_t, -1);
    std::vector<std::int32_t> columns;
    for (const Invocation& invocation : shape.invocations) {
        for (const auto column : select_indices(invocation.tokens, shape.columns)) {
            if (oracle.column_index[column] == -1) {
                oracle.column_index[column] = static_cast<std::int32_t>(columns.size());
                columns.push_back(column);
            }
        }
    }
    const auto materialized_weight = materialize_weight_rows(weight, oracle.rows);
    oracle.dots.resize(oracle.rows.size() * columns.size());
    for (std::size_t selected_column = 0; selected_column < columns.size(); ++selected_column) {
        const auto token = columns[selected_column];
        for (std::size_t row = 0; row < oracle.rows.size(); ++row) {
            double sum = 0.0;
            for (std::int32_t column = 0; column < shape.k; ++column) {
                sum += static_cast<double>(materialized_weight[row * shape.k + column]) *
                       static_cast<double>(bf16_to_f32(
                           activation[static_cast<std::size_t>(token) * shape.k + column]));
            }
            oracle.dots[selected_column * oracle.rows.size() + row] = sum;
        }
    }
    return oracle;
}

int verify_output(std::string_view label, const GuardedDeviceBuffer& output,
                  const InvocationCase& shape, const Invocation& invocation, const Oracle& oracle,
                  std::span<const std::uint16_t> residual, ops::LinearPolicy policy,
                  bool negative) {
    const auto columns = select_indices(invocation.tokens, shape.columns);
    std::vector<std::size_t> wanted;
    std::vector<double> expected;
    for (const auto token : columns) {
        const auto dot_offset =
            static_cast<std::size_t>(oracle.column_index[token]) * oracle.rows.size();
        for (std::size_t row = 0; row < oracle.rows.size(); ++row) {
            const auto offset = static_cast<std::size_t>(token) * shape.n + oracle.rows[row];
            wanted.push_back(offset);
            const double dot = oracle.dots[dot_offset + row];
            expected.push_back((negative ? -dot : dot) +
                               static_cast<double>(bf16_to_f32(residual[offset])));
        }
    }

    std::vector<double> actual(wanted.size());
    const std::size_t total_words = checked_elements(shape.n, invocation.tokens, "output");
    std::vector<std::uint16_t> chunk(std::min(kOutputScanWords, total_words));
    std::size_t wanted_index    = 0;
    std::size_t nonfinite_count = 0;
    for (std::size_t begin = 0; begin < total_words; begin += chunk.size()) {
        const std::size_t count = std::min(chunk.size(), total_words - begin);
        test::cuda_check(cudaMemcpy(chunk.data(),
                                    static_cast<const std::uint16_t*>(output.data()) + begin,
                                    count * sizeof(std::uint16_t), cudaMemcpyDeviceToHost),
                         "copy linear_add output");
        {
            for (std::size_t index = 0; index < count; ++index) {
                if ((chunk[index] & 0x7f80U) == 0x7f80U) { ++nonfinite_count; }
            }
        }
        while (wanted_index < wanted.size() && wanted[wanted_index] < begin + count) {
            actual[wanted_index] =
                static_cast<double>(bf16_to_f32(chunk[wanted[wanted_index] - begin]));
            ++wanted_index;
        }
    }
    int failures = output.verify_guards(label);
    if (nonfinite_count != 0) {
        std::cerr << label << ": output contains " << nonfinite_count << " non-finite values\n";
        ++failures;
    }
    return failures + verify_reduction(label, actual, expected, criterion(policy));
}

int verify_zero_input(std::string_view label, const GuardedDeviceBuffer& output,
                      std::span<const std::uint16_t> residual) {
    std::vector<std::uint16_t> actual(output.bytes() / sizeof(std::uint16_t));
    output.copy_to_host(actual.data(), output.bytes());
    int failures = output.verify_guards(label);
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (bf16_to_f32(actual[index]) != bf16_to_f32(residual[index])) {
            std::cerr << label << ": graph reused stale activation or residual\n";
            ++failures;
            break;
        }
    }
    return failures;
}

} // namespace

static int run_cases(std::string_view label, QType format, const InvocationCase& shape) {
    if (shape.invocations.empty()) {
        throw std::invalid_argument("linear_add test: no invocations");
    }
    std::int32_t maximum_t = 0;
    for (const auto& invocation : shape.invocations) {
        if (invocation.tokens <= 0) {
            throw std::invalid_argument("linear_add test: invalid tokens");
        }
        maximum_t = std::max(maximum_t, invocation.tokens);
    }
    const auto host_weight  = make_weight(format, shape);
    const auto activation   = make_activation(shape, maximum_t);
    const auto residual     = make_residual(shape, maximum_t);
    const auto oracle       = make_oracle(shape, host_weight, activation, maximum_t);
    const auto host_payload = weight_payload(host_weight);
    GuardedDeviceBuffer device_activation(activation.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_weight(host_payload.size());
    device_activation.copy_from_host(activation.data(), device_activation.bytes());
    device_weight.copy_from_host(host_payload.data(), host_payload.size());
    const Weight weight = std::visit(
        [&](const auto& value) { return value.device_weight(device_weight.span()); }, host_weight);
    DeviceContext device;
    const QType qtype         = format;
    int failures              = 0;
    std::size_t executed_peak = 0;
    for (const Invocation& invocation : shape.invocations) {
        const auto t = invocation.tokens;
        GuardedDeviceBuffer output(checked_elements(shape.n, t, "output") * sizeof(std::uint16_t));
        output.copy_from_host(residual.data(), output.bytes());
        Tensor input(device_activation.span(), DType::BF16, {shape.k, t});
        Tensor residual_out(output.span(), DType::BF16, {shape.n, t});
        auto capacity =
            invocation.policy
                ? ops::linear_add_workspace_capacity_bytes(qtype, shape.n, shape.k,
                                                           *invocation.policy, t, t)
                : ops::linear_add_workspace_capacity_bytes(qtype, shape.n, shape.k, t, t);
        for (auto policy : {invocation.graph_policy, invocation.followup_policy}) {
            if (policy)
                capacity = std::max(capacity, ops::linear_add_workspace_capacity_bytes(
                                                  qtype, shape.n, shape.k, *policy, t, t));
        }
        GuardedDeviceBuffer scratch(capacity);
        MemoryArena workspace(scratch.span());
        const auto& case_label = invocation.id;
        const auto verify      = [&](std::string_view stage, ops::LinearPolicy policy,
                                bool negative = false) {
            failures += verify_output(case_label + " " + std::string(stage), output, shape,
                                           invocation, oracle, residual, policy, negative);
        };
        try {
            if (invocation.policy) {
                ops::linear_add(input, weight, residual_out, *invocation.policy, workspace,
                                ninfer::test::cuda_environment().default_execution);
            } else {
                ops::linear_add(input, weight, residual_out, workspace,
                                ninfer::test::cuda_environment().default_execution);
            }
            cuda_synchronize();
            verify("eager", invocation.policy.value_or(ops::LinearPolicy::A16Only));
            if (invocation.graph != GraphReplay::None) {
                DecodeGraphDefinition definition;
                DecodeGraphExecutable graph;
                definition.capture(device.stream(), [&] {
                    ops::linear_add(input, weight, residual_out,
                                    invocation.graph_policy.value_or(
                                        invocation.policy.value_or(ops::LinearPolicy::A16Only)),
                                    workspace, device.execution_view());
                });
                graph.instantiate(definition);
                std::vector<std::uint16_t> negative_activation;
                if (invocation.graph == GraphReplay::OriginalNegative) {
                    negative_activation = activation;
                    for (auto& value : negative_activation) { value ^= 0x8000U; }
                }
                const int replays = invocation.graph == GraphReplay::OriginalZeroOriginal ? 3 : 2;
                for (int replay = 0; replay < replays; ++replay) {
                    const bool zero =
                        invocation.graph == GraphReplay::OriginalZeroOriginal && replay == 1;
                    const bool negative =
                        invocation.graph == GraphReplay::OriginalNegative && replay == 1;
                    if (zero) {
                        CUDA_CHECK(cudaMemsetAsync(input.data, 0, input.bytes(), device.stream()));
                    } else if (invocation.graph != GraphReplay::OriginalTwice) {
                        const auto& values = negative ? negative_activation : activation;
                        const auto input_bytes =
                            invocation.graph == GraphReplay::OriginalZeroOriginal
                                ? input.bytes()
                                : device_activation.bytes();
                        CUDA_CHECK(cudaMemcpyAsync(device_activation.data(), values.data(),
                                                   input_bytes, cudaMemcpyHostToDevice,
                                                   device.stream()));
                    }
                    CUDA_CHECK(cudaMemcpyAsync(output.data(), residual.data(), output.bytes(),
                                               cudaMemcpyHostToDevice, device.stream()));
                    graph.launch(device.stream());
                    device.synchronize();
                    const auto stage = "graph replay " + std::to_string(replay);
                    if (zero) {
                        failures += verify_zero_input(case_label + " " + stage, output, residual);
                    } else {
                        verify(stage,
                               invocation.graph_policy.value_or(
                                   invocation.policy.value_or(ops::LinearPolicy::A16Only)),
                               negative);
                    }
                }
                if (invocation.graph == GraphReplay::OriginalNegative) {
                    device_activation.copy_from_host(activation.data(), device_activation.bytes());
                }
            }
            if (invocation.followup_policy) {
                output.copy_from_host(residual.data(), output.bytes());
                ops::linear_add(input, weight, residual_out, *invocation.followup_policy, workspace,
                                ninfer::test::cuda_environment().default_execution);
                cuda_synchronize();
                verify("permissive policy", *invocation.followup_policy);
            }
        } catch (const std::exception& error) {
            std::cerr << case_label << ": unexpected exception: " << error.what() << '\n';
            ++failures;
            continue;
        }
        if (workspace.used() != 0 || workspace.peak_used() != capacity) {
            std::cerr << case_label << ": exact workspace query/execution high-water mismatch\n";
            ++failures;
        }
        failures += scratch.verify_guards(case_label);
        executed_peak = std::max(executed_peak, workspace.peak_used());
    }
    if (shape.interval_witness && executed_peak != ops::linear_add_workspace_capacity_bytes(
                                                       qtype, shape.n, shape.k, 1, maximum_t)) {
        std::cerr << label << ": interval capacity has no executed high-water witness\n";
        ++failures;
    }
    failures += device_activation.verify_guards("linear_add activation");
    failures += device_weight.verify_guards("linear_add weight");
    failures += verify_preserved(device_activation.data(), std::as_bytes(std::span(activation)),
                                 "linear_add activation");
    failures +=
        verify_preserved(device_weight.data(), std::as_bytes(host_payload), "linear_add weight");
    return failures;
}

void add_cases(Suite& suite, QType format, const InvocationCase& shape) {
    std::vector<Invocation> calls(shape.invocations.begin(), shape.invocations.end());
    std::vector<linear_suite::Case> cases;
    const std::string fixture =
        "s" + std::to_string(shape.seed) +
        (shape.fixture == FixturePattern::TokenBlocks ? "/token-blocks" : "/hashed") + "/rows" +
        std::to_string(static_cast<int>(shape.rows)) + "/columns" +
        std::to_string(static_cast<int>(shape.columns));
    for (auto& call : calls) {
        const auto policy = call.policy.value_or(ops::LinearPolicy::A16Only);
        auto test         = linear_suite::make_case(
            linear_suite::format_name(format), shape.n, shape.k, call.tokens, policy,
            call.graph == GraphReplay::None ? linear_suite::Execution::Eager
                                                    : linear_suite::Execution::Graph,
            fixture + (call.policy ? "/policy" : "/convenience") + "/replay" +
                std::to_string(static_cast<int>(call.graph)));
        if (call.graph_policy) {
            test.policies |= linear_suite::policy_bit(*call.graph_policy);
            test.id += "/capture-" + std::string(linear_suite::policy_name(*call.graph_policy));
        }
        if (call.followup_policy) {
            test.policies |= linear_suite::policy_bit(*call.followup_policy);
            test.id += "/followup-" + std::string(linear_suite::policy_name(*call.followup_policy));
        }
        cases.push_back(std::move(test));
    }
    auto owned_shape        = shape;
    owned_shape.invocations = {};
    suite.add(std::move(cases), [shape = owned_shape, format,
                                 calls = std::move(calls)](auto selected, auto declared) {
        std::vector<Invocation> chosen;
        for (auto index : selected) {
            chosen.push_back(calls[index]);
            chosen.back().id = declared[index].id;
        }
        auto selected_shape        = shape;
        selected_shape.invocations = chosen;
        selected_shape.interval_witness &= selected.size() == calls.size();
        return run_cases(chosen.front().id, format, selected_shape);
    });
}

void add_shape(Suite& suite, QType format, const ShapeCase& shape) {
    std::vector<Invocation> invocations;
    for (const auto t : conformance_tokens(shape)) {
        const bool graph = std::find(shape.graph_tokens.begin(), shape.graph_tokens.end(), t) !=
                           shape.graph_tokens.end();
        invocations.push_back(Invocation{
            .tokens          = t,
            .graph           = graph ? GraphReplay::OriginalZeroOriginal : GraphReplay::None,
            .graph_policy    = graph ? std::optional(ops::LinearPolicy::AllowA4) : std::nullopt,
            .followup_policy = graph ? std::optional(ops::LinearPolicy::AllowA8) : std::nullopt,
        });
    }
    add_cases(suite, format,
              InvocationCase{
                  .n                = shape.n,
                  .k                = shape.k,
                  .seed             = shape.seed,
                  .invocations      = invocations,
                  .rows             = shape.full_output ? Samples::All : Samples::BoundarySeven,
                  .interval_witness = true,
              });
}

} // namespace ninfer::test::linear_add
