#include "core/weight.h"
#include "core/weight_view.h"
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
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::test::linear {

cudaError_t scan_sampled_output(const std::uint16_t* output, std::size_t words, int width,
                                const int* rows, int nr, const int* columns, int nc,
                                std::uint16_t* selected, unsigned* status, cudaStream_t stream);

namespace {

using linear_common::all_indices;
using linear_common::checked_elements;
using linear_common::evenly_spaced_indices;

constexpr std::size_t kOutputGuardBytes   = 256;
constexpr std::uint8_t kOutputGuardByte   = 0xa5;
constexpr std::uint8_t kOutputPoisonByte  = 0xff;
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
        : storage_(cuda_environment().context.device(),
                   words * sizeof(std::uint16_t) + 2 * kOutputGuardBytes) {}

    void* data() { return storage_.span().subspan(kOutputGuardBytes, payload_bytes_).data; }

    MemorySpan span() { return storage_.span().subspan(kOutputGuardBytes, payload_bytes_); }

    void set_extent(std::size_t words) { payload_bytes_ = words * sizeof(std::uint16_t); }

    void poison(cudaStream_t stream) {
        auto* payload = static_cast<std::uint8_t*>(data());
        cuda_check(cudaMemsetAsync(payload - kOutputGuardBytes, kOutputGuardByte, kOutputGuardBytes,
                                   stream),
                   "initialize output prefix");
        cuda_check(cudaMemsetAsync(payload, kOutputPoisonByte, payload_bytes_, stream),
                   "poison linear output");
        cuda_check(
            cudaMemsetAsync(payload + payload_bytes_, kOutputGuardByte, kOutputGuardBytes, stream),
            "initialize output suffix");
    }

    int verify_guards(std::string_view label) {
        const auto* payload = static_cast<const std::uint8_t*>(data());
        std::array<std::uint8_t, 2 * kOutputGuardBytes> guards;
        cuda_check(cudaMemcpy(guards.data(), payload - kOutputGuardBytes, kOutputGuardBytes,
                              cudaMemcpyDeviceToHost),
                   "read output prefix");
        cuda_check(cudaMemcpy(guards.data() + kOutputGuardBytes, payload + payload_bytes_,
                              kOutputGuardBytes, cudaMemcpyDeviceToHost),
                   "read output suffix");
        if (std::all_of(guards.begin(), guards.end(),
                        [](auto byte) { return byte == kOutputGuardByte; }))
            return 0;
        std::cerr << label << ": device buffer guard was overwritten\n";
        return 1;
    }

private:
    DeviceBuffer storage_;
    std::size_t payload_bytes_ = 0;
};

std::vector<std::uint16_t> make_activation(std::int32_t k, std::int32_t t, std::uint32_t seed,
                                           ActivationRecipe recipe) {
    const std::size_t elements = checked_elements(k, t, "activation");
    std::vector<std::uint16_t> result(elements);
    parallel_rows(t, elements, [&](int first, int last) {
        for (int token = first; token < last; ++token) {
            for (std::int32_t column = 0; column < k; ++column) {
                std::uint32_t coordinate = seed ^
                                           (static_cast<std::uint32_t>(column) * 0x9e3779b9U) ^
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
    });
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
                       std::span<std::uint16_t> host, std::string_view label) {
    const std::size_t total_words = checked_elements(n, t, "output");
    OutputRead result;
    result.selected.resize(rows.size() * columns.size());
    cuda_check(cudaMemcpy(host.data(), device, total_words * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost),
               "copy linear output");
    std::atomic<std::size_t> poison_count    = 0;
    std::atomic<std::size_t> nonfinite_count = 0;
    parallel_rows(t, total_words, [&](int first, int last) {
        std::size_t poison    = 0;
        std::size_t nonfinite = 0;
        for (std::size_t index = std::size_t(first) * n; index < std::size_t(last) * n; ++index) {
            const auto bits = host[index];
            poison += bits == 0xffffU;
            nonfinite += (bits & 0x7f80U) == 0x7f80U;
        }
        poison_count.fetch_add(poison, std::memory_order_relaxed);
        nonfinite_count.fetch_add(nonfinite, std::memory_order_relaxed);
    });
    parallel_rows(columns.size(), result.selected.size(), [&](int first, int last) {
        for (int c = first; c < last; ++c) {
            const auto* source = host.data() + std::size_t(columns[c]) * n;
            auto* target       = result.selected.data() + std::size_t(c) * rows.size();
            for (std::size_t r = 0; r < rows.size(); ++r)
                target[r] = test::bf16_to_f32(source[rows[r]]);
        }
    });
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

// Sampled oracles still check every output for poison and non-finite values. For large
// outputs only transfer the flags and the words the FP64 oracle actually compares.
struct SampledRead {
    DeviceBuffer row_indices, column_indices, selected, status;
    std::vector<std::uint16_t> host;

    SampledRead(ninfer::Device device, std::span<const int> rows, int max_columns)
        : row_indices(device, rows.size_bytes()), column_indices(device, max_columns * sizeof(int)),
          selected(device, rows.size() * max_columns * sizeof(std::uint16_t)),
          status(device, sizeof(unsigned)), host(rows.size() * max_columns) {
        row_indices.copy_from_host(rows.data(), rows.size_bytes());
    }

    OutputRead read(const void* output, int n, int t, int nr, std::span<const int> columns,
                    cudaStream_t stream, std::string_view label) {
        cuda_check(cudaMemcpyAsync(column_indices.data(), columns.data(), columns.size_bytes(),
                                   cudaMemcpyHostToDevice, stream),
                   "upload sampled columns");
        cuda_check(cudaMemsetAsync(status.data(), 0, sizeof(unsigned), stream),
                   "initialize output scan flags");
        cuda_check(scan_sampled_output(static_cast<const std::uint16_t*>(output),
                                       std::size_t(n) * t, n,
                                       static_cast<const int*>(row_indices.data()), nr,
                                       static_cast<const int*>(column_indices.data()),
                                       columns.size(), static_cast<std::uint16_t*>(selected.data()),
                                       static_cast<unsigned*>(status.data()), stream),
                   "scan and gather sampled linear output");
        cuda_check(cudaStreamSynchronize(stream), "complete sampled output scan");
        unsigned flags;
        status.copy_to_host(&flags, sizeof(flags));
        const auto count = nr * columns.size();
        selected.copy_to_host(host.data(), count * sizeof(std::uint16_t));
        OutputRead result;
        result.selected.resize(count);
        std::transform(host.begin(), host.begin() + count, result.selected.begin(), bf16_to_f32);
        if (flags & 1) {
            std::cerr << label << ": output retains poison values\n";
            ++result.failures;
        }
        if (flags & 2) {
            std::cerr << label << ": output contains non-finite values\n";
            ++result.failures;
        }
        return result;
    }
};

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

    // Transpose a block of independent tokens so SIMD lanes do not change the
    // accumulation order along K for any output.
    const int blocks = (t + kOracleTBlock - 1) / kOracleTBlock;
    using TokenBlock = std::array<double, kOracleTBlock>;
    std::vector<TokenBlock> x(std::size_t(blocks) * k);
    parallel_rows(blocks, std::uint64_t(t) * k, [&](int first, int last) {
        for (int block = first; block < last; ++block) {
            const int token_begin = block * kOracleTBlock;
            const int active      = std::min(kOracleTBlock, t - token_begin);
            for (int column = 0; column < k; ++column)
                for (int token = 0; token < active; ++token)
                    x[std::size_t(block) * k + column][token] =
                        activation[std::size_t(token_begin + token) * k + column];
        }
    });
    parallel_rows(n, std::uint64_t(n) * k * t, [&](int first, int last) {
        for (int row = first; row < last; ++row) {
            const float* weight_row = weight + std::size_t(row) * k;
            for (int block = 0; block < blocks; ++block) {
                const int token_begin = block * kOracleTBlock;
                const int active      = std::min(kOracleTBlock, t - token_begin);
                TokenBlock sums{};
                const auto* inputs = x.data() + std::size_t(block) * k;
                for (int column = 0; column < k; ++column) {
                    const double value = weight_row[column];
                    for (int token = 0; token < kOracleTBlock; ++token)
                        sums[token] += value * inputs[column][token];
                }
                for (int token = 0; token < active; ++token)
                    output[std::size_t(token_begin + token) * n + row] = sums[token];
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
    quantized_weight::PackedWeight host_weight =
        generator(shape.parent_rows ? shape.parent_rows : shape.n, shape.k, shape.seed);
    if (shape.parent_rows) {
        if (shape.parent_rows <= shape.n || host_weight.weight.qtype != QType::BF16)
            throw std::invalid_argument("linear test: prefix requires a larger BF16 parent");
        std::fill(host_weight.payload.begin() + std::size_t(shape.n) * shape.k * 2,
                  host_weight.payload.end(), 0xff);
    }
    const std::vector<float> oracle_weight =
        quantized_weight::materialize_rows_fp32(host_weight, oracle_rows);
    std::vector<std::uint16_t> activation_bits =
        shape.activation_generator
            ? shape.activation_generator(shape.k, maximum->t, shape.seed + 1U)
            : make_activation(shape.k, maximum->t, shape.seed + 1U, recipe);
    if (shape.activation_pattern != ActivationPattern::Dense) {
        parallel_rows(maximum->t, activation_bits.size(), [&](int first, int last) {
            for (int token = first; token < last; ++token) {
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
        });
    }

    DeviceBuffer device_activation(ninfer::test::cuda_environment().context.device(),
                                   activation_bits.size() * sizeof(std::uint16_t));
    device_activation.copy_from_host(activation_bits.data(), device_activation.size());
    DeviceBuffer device_weight(ninfer::test::cuda_environment().context.device(),
                               host_weight.payload.size());
    device_weight.copy_from_host(host_weight.payload.data(), device_weight.size());
    Weight weight = host_weight.device_weight(device_weight.span());
    if (shape.parent_rows) {
        const std::array parent_shape{std::uint64_t(shape.parent_rows), std::uint64_t(shape.k)};
        const WeightParent parent{
            weight_geometry(QType::BF16, QuantLayout::Contiguous, parent_shape),
            device_weight.device(), static_cast<const std::byte*>(device_weight.data())};
        weight = native_weight({{std::uint64_t(shape.n), std::uint64_t(shape.k)},
                                {{&parent, 0, std::uint64_t(shape.n) * shape.k}}});
    }

    std::vector<double> full_reference;
    if (shape.comparison == Comparison::Full) {
        const std::vector<std::int32_t> columns = all_indices(maximum->t);
        const std::vector<float> activation =
            materialize_activation(activation_bits, shape.k, columns);
        full_reference.resize(checked_elements(shape.n, maximum->t, "full reference"));
        cpu_linear_gemm_fp64(oracle_weight.data(), activation.data(), full_reference.data(),
                             shape.n, shape.k, maximum->t);
    }

    const auto max_words = checked_elements(shape.n, maximum->t, "guarded output");
    GuardedOutput output(max_words);
    constexpr auto kDeviceScanWords = std::size_t(1) << 20;
    std::optional<SampledRead> sampled_read;
    if (shape.comparison == Comparison::Sampled && max_words >= kDeviceScanWords)
        sampled_read.emplace(device_activation.device(), oracle_rows, maximum->t);
    std::vector<std::uint16_t> host_output(sampled_read ? kDeviceScanWords : max_words);
    int failures = 0;
    for (const Invocation& invocation : shape.invocations) {
        const auto& case_label = invocation.id;
        output.set_extent(checked_elements(shape.n, invocation.t, "guarded output"));
        Tensor input(device_activation.span(), DType::BF16, {shape.k, invocation.t});
        Tensor destination(output.span(), DType::BF16, {shape.n, invocation.t});
        const std::size_t capacity = ops::linear_workspace_capacity_bytes(
            weight.qtype, shape.n, shape.k, invocation.policy, invocation.t, invocation.t);
        GuardedDeviceBuffer workspace_storage(capacity + shape.workspace_prefix);
        MemoryArena workspace(workspace_storage.span());
        if (shape.workspace_prefix) {
            workspace_storage.fill(kOutputGuardByte);
            (void)workspace.alloc_bytes(shape.workspace_prefix, 1);
        }
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
        const auto active = std::span(activation_bits).first(std::size_t(invocation.t) * shape.k);
        std::vector<std::uint16_t> negative;
        try {
            if (invocation.graph_replay) {
                graph_definition.capture(stream, launch);
                graph.instantiate(graph_definition);
            }
            for (int replay = 0; replay < (invocation.graph_replay ? 2 : 1); ++replay) {
                if (replay == 1) {
                    // BF16 negation is exact. Evaluate the same FP64 oracle using -X below.
                    if (shape.verify_input_preservation)
                        failures += linear_common::verify_preserved(
                            device_activation.data(), std::as_bytes(active), case_label);
                    negative.assign(active.begin(), active.end());
                    for (auto& bits : negative) bits ^= 0x8000;
                    device_activation.copy_from_host(negative.data(), active.size_bytes());
                }
                output.poison(stream);
                if (invocation.graph_replay)
                    graph.launch(stream);
                else
                    launch();
                cuda_check(cudaStreamSynchronize(stream), "synchronize linear");
                if (workspace.peak_used() > capacity + shape.workspace_prefix ||
                    workspace.used() != shape.workspace_prefix) {
                    std::cerr << case_label << ": workspace query/scope mismatch\n";
                    ++failures;
                }

                failures += output.verify_guards(case_label);
                failures += workspace_storage.verify_guards(case_label + " workspace");
                if (shape.workspace_prefix) {
                    std::vector<std::uint8_t> prefix(shape.workspace_prefix);
                    cuda_check(cudaMemcpy(prefix.data(), workspace_storage.data(), prefix.size(),
                                          cudaMemcpyDeviceToHost),
                               "read caller workspace prefix");
                    if (!std::all_of(prefix.begin(), prefix.end(),
                                     [](auto byte) { return byte == kOutputGuardByte; })) {
                        std::cerr << case_label << ": modified caller workspace prefix\n";
                        ++failures;
                    }
                }
                const std::vector<std::int32_t> columns =
                    shape.comparison == Comparison::Full ? all_indices(invocation.t)
                    : shape.column_sampler               ? shape.column_sampler(invocation.t)
                                                         : evenly_spaced_indices(invocation.t);
                const auto output_words = std::size_t(shape.n) * invocation.t;
                OutputRead actual =
                    sampled_read && output_words >= kDeviceScanWords
                        ? sampled_read->read(output.data(), shape.n, invocation.t,
                                             oracle_rows.size(), columns, stream, case_label)
                        : read_output(output.data(), shape.n, invocation.t, oracle_rows, columns,
                                      host_output, case_label);
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
                    failures += linear_common::verify_preserved(device_activation.data(),
                                                                std::as_bytes(std::span(negative)),
                                                                case_label + " graph activation");
                }
            }
        } catch (const std::exception& error) {
            std::cerr << case_label << ": unexpected exception: " << error.what() << '\n';
            ++failures;
        }
        if (!negative.empty()) device_activation.copy_from_host(active.data(), active.size_bytes());
    }

    if (shape.verify_input_preservation) {
        failures += linear_common::verify_preserved(
            device_activation.data(), std::as_bytes(std::span(activation_bits)), label);
        failures += linear_common::verify_preserved(
            device_weight.data(), std::as_bytes(std::span(host_weight.payload)), label);
    }
    return failures;
}

void add_shape(Suite& suite, QType format, ActivationRecipe recipe, WeightGenerator generator,
               const ShapeCase& shape) {
    std::vector<Invocation> calls;
    std::vector<linear_suite::Case> cases;
    std::string fixture = "s" + std::to_string(shape.seed) +
                          (recipe == ActivationRecipe::TokenBlocks ? "/token-blocks" : "/hashed") +
                          (shape.comparison == Comparison::Full ? "/full" : "/sampled");
    if (shape.activation_generator) fixture += "/projection-recipe";
    if (shape.activation_pattern == ActivationPattern::KTail) fixture += "/k-tail";
    if (shape.activation_pattern == ActivationPattern::Cancellation) fixture += "/cancellation";
    for (const auto& call : shape.invocations) {
        auto test = linear_suite::make_case(
            linear_suite::format_name(format), shape.n, shape.k, call.t, call.policy,
            call.graph_replay ? linear_suite::Execution::Graph : linear_suite::Execution::Eager,
            fixture + (call.call_form == CallForm::A16Convenience ? "/convenience" : "/policy"));
        if (std::any_of(cases.begin(), cases.end(), [&](const auto& c) { return c.id == test.id; }))
            continue;
        calls.push_back(call);
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
    std::vector<std::pair<int, int>> intervals{{1, 4},  {2, 4},    {1, 128},
                                               {9, 25}, {24, 129}, {1, 2048}};
    if (qtype == QType::NVFP4) intervals.emplace_back(1, 4097);
    const bool block_tail = qtype == QType::FP8_E4M3FN_BLOCK128X128_FP32 && n == 4096 && k == 16384;
    if (block_tail) {
        intervals.insert(intervals.end(), {{1, 4096},
                                           {512, 641},
                                           {641, 896},
                                           {641, 897},
                                           {1024, 1409},
                                           {2049, 2433},
                                           {std::numeric_limits<int>::max() - 512,
                                            std::numeric_limits<int>::max()}});
    }
    for (auto policy :
         {ops::LinearPolicy::A16Only, ops::LinearPolicy::AllowA8, ops::LinearPolicy::AllowA4}) {
        std::size_t observed_peak = 0;
        for (auto [first, last] : intervals) {
            const auto capacity =
                ops::linear_workspace_capacity_bytes(qtype, n, k, policy, first, last);
            std::size_t interval_peak = 0;
            for (int t = first;; ++t) {
                const auto point = ops::linear_workspace_capacity_bytes(qtype, n, k, policy, t, t);
                interval_peak    = std::max(interval_peak, point);
                if (point > capacity) {
                    std::cerr << "Linear workspace interval [" << first << ',' << last
                              << "] cannot cover T=" << t << " for [" << n << ',' << k << "]\n";
                    ++failures;
                    break;
                }
                if (t == last) break;
            }
            observed_peak = std::max(observed_peak, interval_peak);
            if (block_tail && capacity != interval_peak) {
                std::cerr
                    << "Block-FP8 workspace interval does not match its point high-water mark\n";
                ++failures;
            }
        }
        if (block_tail && ops::linear_workspace_capacity_bytes(qtype, n, k, policy, 1,
                                                               std::numeric_limits<int>::max()) !=
                              observed_peak) {
            std::cerr << "Block-FP8 full token range misses transient workspace\n";
            ++failures;
        }
    }
    return failures;
}

} // namespace ninfer::test::linear
