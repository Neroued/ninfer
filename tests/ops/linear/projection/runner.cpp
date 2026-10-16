#include "core/weight.h"
#include "cuda_fixture.h"
#include "core/buffer.h"
#include "ops/linear/projection/runner.h"

#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "ops/op_tester.h"
#include "ops/direct_bf16_weight.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::test::linear {
namespace {

using linear_common::all_indices;
using linear_common::checked_elements;
using linear_common::evenly_spaced_indices;

constexpr std::size_t kOutputGuardBytes   = 256;
constexpr std::uint8_t kOutputGuardByte   = 0xa5;
constexpr std::uint8_t kOutputPoisonByte  = 0xff;
constexpr std::size_t kOutputScanWords    = 1U << 20;
constexpr int kOracleTBlock               = 8;
constexpr double kBf16UnitRoundoff        = 1.0 / 256.0;
constexpr double kA8QuantizationAllowance = 0.04;
constexpr double kA4QuantizationAllowance = 0.16;

// Precision follows the public policy, including when a more accurate route is selected.
constexpr ReductionCriterion tolerance_for(ops::LinearPolicy policy) {
    switch (policy) {
    case ops::LinearPolicy::A16Only:
        return {kBf16UnitRoundoff, kBf16UnitRoundoff, 2.0 * kBf16UnitRoundoff};
    case ops::LinearPolicy::AllowA8:
        return {kA8QuantizationAllowance, kBf16UnitRoundoff, 1.5 * kA8QuantizationAllowance};
    case ops::LinearPolicy::AllowA4:
        return {kA4QuantizationAllowance, kBf16UnitRoundoff, kA4QuantizationAllowance};
    }
    throw std::invalid_argument("linear test: unknown Linear policy");
}

class GuardedOutput {
public:
    explicit GuardedOutput(std::size_t words)
        : storage_(words * sizeof(std::uint16_t), kOutputGuardBytes, kOutputGuardByte) {
        poison();
    }

    void* data() { return storage_.data(); }

    const void* data() const { return storage_.data(); }

    MemorySpan span() { return storage_.span(); }

    ConstMemorySpan span() const { return storage_.span(); }

    void poison() { storage_.fill(kOutputPoisonByte); }

    int verify_guards(std::string_view label) const { return storage_.verify_guards(label); }

private:
    GuardedDeviceBuffer storage_;
};

std::vector<std::uint16_t> make_activation(std::int32_t k, std::int32_t t, std::uint32_t seed,
                                           ActivationRecipe recipe) {
    const std::size_t elements = checked_elements(k, t, "activation");
    std::vector<std::uint16_t> result(elements);
    for (std::int32_t token = 0; token < t; ++token) {
        for (std::int32_t column = 0; column < k; ++column) {
            std::uint32_t coordinate = seed ^ (static_cast<std::uint32_t>(column) * 0x9e3779b9U) ^
                                       (static_cast<std::uint32_t>(token) * 0x85ebca6bU);
            if (recipe == ActivationRecipe::TokenBlocks) {
                const std::uint64_t token_block = static_cast<std::uint64_t>(token / 256);
                coordinate                      = static_cast<std::uint32_t>(
                    static_cast<std::uint64_t>(column) * 17U +
                    static_cast<std::uint64_t>(token) * 31U +
                    static_cast<std::uint64_t>(seed) * 13U +
                    token_block * (static_cast<std::uint64_t>(column / 256) * 13U + 47U));
            } else {
                coordinate ^= coordinate >> 16;
                coordinate *= 0x7feb352dU;
                coordinate ^= coordinate >> 15;
                coordinate *= 0x846ca68bU;
                coordinate ^= coordinate >> 16;
            }
            const int raw     = static_cast<int>(coordinate & 0xffU);
            const float value = static_cast<float>(raw - 128) * (1.0F / 256.0F);
            result[static_cast<std::size_t>(token) * k + column] = test::f32_to_bf16(value);
        }
    }
    return result;
}

std::vector<float> materialize_activation(const std::vector<std::uint16_t>& bits, std::int32_t k,
                                          std::span<const std::int32_t> columns) {
    std::vector<float> result(
        checked_elements(k, static_cast<std::int32_t>(columns.size()), "oracle activation"));
    for (std::size_t oracle_column = 0; oracle_column < columns.size(); ++oracle_column) {
        const std::uint16_t* source =
            bits.data() + static_cast<std::size_t>(columns[oracle_column]) * k;
        float* destination = result.data() + oracle_column * static_cast<std::size_t>(k);
        for (std::int32_t column = 0; column < k; ++column) {
            destination[column] = test::bf16_to_f32(source[column]);
        }
    }
    return result;
}

struct OutputRead {
    int failures = 0;
    std::vector<double> selected;
};

OutputRead read_output(const void* device, std::int32_t n, std::int32_t t,
                       std::span<const std::int32_t> rows, std::span<const std::int32_t> columns,
                       std::string_view label) {
    const std::size_t total_words = checked_elements(n, t, "output");
    std::vector<std::size_t> wanted;
    wanted.reserve(rows.size() * columns.size());
    for (const std::int32_t column : columns) {
        for (const std::int32_t row : rows) {
            wanted.push_back(static_cast<std::size_t>(column) * n + static_cast<std::size_t>(row));
        }
    }

    OutputRead result;
    result.selected.resize(wanted.size());
    std::vector<std::uint16_t> chunk(std::min(kOutputScanWords, total_words));
    std::size_t wanted_index    = 0;
    std::size_t poison_count    = 0;
    std::size_t nonfinite_count = 0;
    for (std::size_t begin = 0; begin < total_words; begin += chunk.size()) {
        const std::size_t count = std::min(chunk.size(), total_words - begin);
        cuda_check(
            cudaMemcpy(chunk.data(),
                       static_cast<const std::uint8_t*>(device) + begin * sizeof(std::uint16_t),
                       count * sizeof(std::uint16_t), cudaMemcpyDeviceToHost),
            "copy linear output");
        for (std::size_t index = 0; index < count; ++index) {
            const std::uint16_t bits = chunk[index];
            if (bits == 0xffffU) { ++poison_count; }
            if ((bits & 0x7f80U) == 0x7f80U) { ++nonfinite_count; }
        }
        while (wanted_index < wanted.size() && wanted[wanted_index] < begin + count) {
            result.selected[wanted_index] =
                static_cast<double>(test::bf16_to_f32(chunk[wanted[wanted_index] - begin]));
            ++wanted_index;
        }
    }
    if (poison_count != 0) {
        std::cerr << label << ": output retains " << poison_count << " poison values\n";
        ++result.failures;
    }
    if (nonfinite_count != 0) {
        std::cerr << label << ": output contains " << nonfinite_count << " non-finite values\n";
        ++result.failures;
    }
    return result;
}

int compare_output(std::string_view label, std::span<const double> actual,
                   std::span<const double> reference, ops::LinearPolicy policy) {
    return verify_reduction(label, actual, reference, tolerance_for(policy));
}

} // namespace

void cpu_linear_gemm_fp64(const float* weight, const float* activation, double* output,
                          std::int32_t n, std::int32_t k, std::int32_t t) {
    if (weight == nullptr || activation == nullptr || output == nullptr || n <= 0 || k <= 0 ||
        t <= 0) {
        throw std::invalid_argument("linear test: invalid FP64 GEMM argument");
    }

    linear_suite::parallel_rows(
        n, static_cast<std::uint64_t>(n) * k * t, [&](int row_begin, int row_end) {
            for (std::int32_t row = row_begin; row < row_end; ++row) {
                const float* weight_row = weight + static_cast<std::size_t>(row) * k;
                for (std::int32_t token_begin = 0; token_begin < t; token_begin += kOracleTBlock) {
                    const std::int32_t active = std::min(kOracleTBlock, t - token_begin);
                    std::array<double, kOracleTBlock> accumulators{};
                    for (std::int32_t column = 0; column < k; ++column) {
                        const double weight_value = static_cast<double>(weight_row[column]);
                        for (std::int32_t token = 0; token < active; ++token) {
                            accumulators[static_cast<std::size_t>(token)] +=
                                weight_value *
                                static_cast<double>(
                                    activation[static_cast<std::size_t>(token_begin + token) * k +
                                               column]);
                        }
                    }
                    for (std::int32_t token = 0; token < active; ++token) {
                        output[static_cast<std::size_t>(token_begin + token) * n + row] =
                            accumulators[static_cast<std::size_t>(token)];
                    }
                }
            }
        });
}

static int run_shape(std::string_view label, ActivationRecipe recipe, WeightGenerator generator,
                     const ShapeCase& shape) {
    if (shape.invocations.empty()) {
        throw std::invalid_argument("linear test: shape has no invocations");
    }
    const auto maximum = std::max_element(
        shape.invocations.begin(), shape.invocations.end(),
        [](const Invocation& left, const Invocation& right) { return left.t < right.t; });
    if (maximum->t <= 0) {
        throw std::invalid_argument("linear test: token extent must be positive");
    }

    const std::vector<std::int32_t> oracle_rows =
        shape.comparison == Comparison::Full ? all_indices(shape.n)
        : shape.row_sampler                  ? shape.row_sampler(shape.n)
                                             : evenly_spaced_indices(shape.n);
    quantized_weight::PackedWeight host_weight = generator(shape.n, shape.k, shape.seed);
    const std::vector<float> oracle_weight =
        quantized_weight::materialize_rows_fp32(host_weight, oracle_rows);
    std::vector<std::uint16_t> activation_bits =
        shape.activation_generator
            ? shape.activation_generator(shape.k, maximum->t, shape.seed + 1U)
            : make_activation(shape.k, maximum->t, shape.seed + 1U, recipe);
    if (shape.activation_pattern != ActivationPattern::Dense) {
        for (int token = 0; token < maximum->t; ++token) {
            for (int column = 0; column < shape.k; ++column) {
                float value = 0.0F;
                if (shape.activation_pattern == ActivationPattern::Cancellation) {
                    value = (column & 1) ? -0.25F : 0.25F;
                    if (column == shape.k - 1) value = 0.125F;
                } else if (column >= shape.k - 16) {
                    value = static_cast<float>((column + token) % 3 - 1) * 0.25F;
                }
                activation_bits[static_cast<std::size_t>(token) * shape.k + column] =
                    test::f32_to_bf16(value);
            }
        }
    }

    DeviceBuffer device_activation(ninfer::test::cuda_environment().context.device(),
                                   activation_bits.size() * sizeof(std::uint16_t));
    device_activation.copy_from_host(activation_bits.data(), device_activation.size());
    DeviceBuffer device_weight(ninfer::test::cuda_environment().context.device(),
                               host_weight.payload.size());
    device_weight.copy_from_host(host_weight.payload.data(), device_weight.size());
    const Weight weight = host_weight.device_weight(device_weight.span());

    std::vector<double> full_reference;
    if (shape.comparison == Comparison::Full) {
        const std::vector<std::int32_t> columns = all_indices(maximum->t);
        const std::vector<float> activation =
            materialize_activation(activation_bits, shape.k, columns);
        full_reference.resize(checked_elements(shape.n, maximum->t, "full reference"));
        cpu_linear_gemm_fp64(oracle_weight.data(), activation.data(), full_reference.data(),
                             shape.n, shape.k, maximum->t);
    }

    int failures = 0;
    for (const Invocation& invocation : shape.invocations) {
        const auto& case_label = invocation.id;
        GuardedOutput output(checked_elements(shape.n, invocation.t, "guarded output"));
        Tensor input(device_activation.span(), DType::BF16, {shape.k, invocation.t});
        Tensor destination(output.span(), DType::BF16, {shape.n, invocation.t});
        const std::size_t capacity = ops::linear_workspace_capacity_bytes(
            weight.qtype, shape.n, shape.k, invocation.policy, invocation.t, invocation.t);
        GuardedDeviceBuffer workspace_storage(capacity);
        MemoryArena workspace(workspace_storage.span());
        std::unique_ptr<DeviceContext> graph_context;
        DecodeGraphDefinition graph_definition;
        DecodeGraphExecutable graph;
        if (invocation.graph_replay) graph_context = std::make_unique<DeviceContext>();
        const auto cuda =
            graph_context ? graph_context->execution_view() : cuda_environment().default_execution;
        const auto stream = cuda.stream();
        const auto launch = [&] {
            if (invocation.call_form == CallForm::A16Convenience) {
                ops::linear(input, weight, destination, cuda);
            } else {
                ops::linear(input, weight, destination, invocation.policy, workspace, cuda);
            }
        };
        try {
            if (invocation.graph_replay) {
                cuda_check(cudaDeviceSynchronize(), "finish input initialization before capture");
                graph_definition.capture(stream, launch);
                graph.instantiate(graph_definition);
            }
            for (int replay = 0; replay < (invocation.graph_replay ? 2 : 1); ++replay) {
                if (replay == 1) {
                    // BF16 negation is exact. Evaluate the same FP64 oracle using -X below.
                    auto negative = activation_bits;
                    for (auto& bits : negative) bits ^= 0x8000;
                    device_activation.copy_from_host(negative.data(), device_activation.size());
                }
                output.poison();
                cuda_check(cudaDeviceSynchronize(), "finish poisoning before replay");
                if (invocation.graph_replay)
                    graph.launch(stream);
                else
                    launch();
                cuda_check(cudaStreamSynchronize(stream), "synchronize linear");
                if (workspace.peak_used() > capacity || workspace.used() != 0) {
                    std::cerr << case_label << ": workspace query/scope mismatch\n";
                    ++failures;
                }

                failures += output.verify_guards(case_label);
                failures += workspace_storage.verify_guards(case_label + " workspace");
                const std::vector<std::int32_t> columns =
                    shape.comparison == Comparison::Full ? all_indices(invocation.t)
                    : shape.column_sampler               ? shape.column_sampler(invocation.t)
                                                         : evenly_spaced_indices(invocation.t);
                OutputRead actual = read_output(output.data(), shape.n, invocation.t, oracle_rows,
                                                columns, case_label);
                failures += actual.failures;

                if (shape.comparison == Comparison::Full) {
                    std::span<const double> reference(
                        full_reference.data(),
                        checked_elements(shape.n, invocation.t, "reference prefix"));
                    std::vector<double> negative_reference;
                    if (replay == 1) {
                        negative_reference.assign(reference.begin(), reference.end());
                        for (double& value : negative_reference) value = -value;
                        reference = negative_reference;
                    }
                    failures +=
                        compare_output(case_label, actual.selected, reference, invocation.policy);
                } else {
                    std::vector<float> activation =
                        materialize_activation(activation_bits, shape.k, columns);
                    if (replay == 1)
                        for (float& value : activation) value = -value;
                    std::vector<double> reference(checked_elements(
                        static_cast<std::int32_t>(oracle_rows.size()),
                        static_cast<std::int32_t>(columns.size()), "sampled reference"));
                    cpu_linear_gemm_fp64(oracle_weight.data(), activation.data(), reference.data(),
                                         static_cast<std::int32_t>(oracle_rows.size()), shape.k,
                                         static_cast<std::int32_t>(columns.size()));
                    failures +=
                        compare_output(case_label, actual.selected, reference, invocation.policy);
                }
                if (replay == 1) {
                    std::vector<std::uint16_t> after(activation_bits.size());
                    device_activation.copy_to_host(after.data(), device_activation.size());
                    for (std::size_t i = 0; i < after.size(); ++i) {
                        if (after[i] != (activation_bits[i] ^ 0x8000)) {
                            std::cerr << case_label << ": modified graph activation input\n";
                            ++failures;
                            break;
                        }
                    }
                    device_activation.copy_from_host(activation_bits.data(),
                                                     device_activation.size());
                }
            }
        } catch (const std::exception& error) {
            std::cerr << case_label << ": unexpected exception: " << error.what() << '\n';
            ++failures;
        }
        if (shape.verify_each_invocation) {
            failures += linear_common::verify_preserved(
                device_activation.data(), std::as_bytes(std::span(activation_bits)), case_label);
            failures += linear_common::verify_preserved(
                device_weight.data(), std::as_bytes(std::span(host_weight.payload)), case_label);
        }
    }

    if (shape.verify_input_preservation && !shape.verify_each_invocation) {
        failures += linear_common::verify_preserved(
            device_activation.data(), std::as_bytes(std::span(activation_bits)), label);
        failures += linear_common::verify_preserved(
            device_weight.data(), std::as_bytes(std::span(host_weight.payload)), label);
    }
    return failures;
}

void add_shape(Suite& suite, QType format, ActivationRecipe recipe, WeightGenerator generator,
               const ShapeCase& shape) {
    std::vector<Invocation> calls(shape.invocations.begin(), shape.invocations.end());
    std::vector<linear_suite::Case> cases;
    std::string fixture = "s" + std::to_string(shape.seed) +
                          (recipe == ActivationRecipe::TokenBlocks ? "/token-blocks" : "/hashed") +
                          (shape.comparison == Comparison::Full ? "/full" : "/sampled");
    if (shape.activation_generator) fixture += "/projection-recipe";
    if (shape.activation_pattern == ActivationPattern::KTail) fixture += "/k-tail";
    if (shape.activation_pattern == ActivationPattern::Cancellation) fixture += "/cancellation";
    for (auto& call : calls) {
        auto test = linear_suite::make_case(
            linear_suite::format_name(format), shape.n, shape.k, call.t, call.policy,
            call.graph_replay ? linear_suite::Execution::Graph : linear_suite::Execution::Eager,
            fixture + (call.call_form == CallForm::A16Convenience ? "/convenience" : "/policy"));
        cases.push_back(std::move(test));
    }
    auto owned_shape        = shape;
    owned_shape.invocations = {};
    suite.add(std::move(cases), [shape = owned_shape, recipe, generator,
                                 calls = std::move(calls)](auto selected, auto declared) {
        std::vector<Invocation> chosen;
        for (auto index : selected) {
            chosen.push_back(calls[index]);
            chosen.back().id = declared[index].id;
        }
        auto selected_shape        = shape;
        selected_shape.invocations = chosen;
        return run_shape(chosen.front().id, recipe, generator, selected_shape);
    });
}

void add_workspace_cases(Suite& suite, QType format, int n, int k) {
    auto test     = linear_suite::make_case(linear_suite::format_name(format), n, k, 0,
                                            ops::LinearPolicy::A16Only, linear_suite::Execution::Eager,
                                            "workspace-intervals");
    test.policies = 7;
    suite.add(std::move(test), [=] { return verify_workspace_envelopes(format, n, k); });
}

int verify_workspace_envelopes(QType qtype, std::int32_t n, std::int32_t k) {
    int failures = 0;
    for (auto policy :
         {ops::LinearPolicy::A16Only, ops::LinearPolicy::AllowA8, ops::LinearPolicy::AllowA4}) {
        for (auto [first, last] : {std::pair{1, 4}, std::pair{2, 4}, std::pair{1, 128},
                                   std::pair{9, 25}, std::pair{24, 129}}) {
            const auto capacity =
                ops::linear_workspace_capacity_bytes(qtype, n, k, policy, first, last);
            for (int t = first; t <= last; ++t) {
                const auto point = ops::linear_workspace_capacity_bytes(qtype, n, k, policy, t, t);
                if (point > capacity) {
                    std::cerr << "Linear workspace interval [" << first << ',' << last
                              << "] cannot cover T=" << t << " for [" << n << ',' << k << "]\n";
                    ++failures;
                    break;
                }
            }
        }
    }
    return failures;
}

} // namespace ninfer::test::linear
