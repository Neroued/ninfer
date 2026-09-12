#include "ops/linear/linear_test_common.h"

#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "ops/op_tester.h"

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
#include <thread>
#include <utility>

namespace ninfer::test::linear {
namespace {

constexpr std::size_t kOutputGuardBytes   = 256;
constexpr std::uint8_t kOutputGuardByte   = 0xa5;
constexpr std::uint8_t kOutputPoisonByte  = 0xff;
constexpr std::size_t kOutputScanWords    = 1U << 20;
constexpr int kOracleTBlock               = 8;
constexpr double kBf16UnitRoundoff        = 1.0 / 256.0;
constexpr double kA8QuantizationAllowance = 0.04;
constexpr double kA4QuantizationAllowance = 0.16;

// The criterion belongs to the activation compute path, not to a private kernel, schedule, or
// launcher selected inside that path. The relative-L2 allowance is one BF16 unit roundoff; the
// gross allowance covers final BF16 storage plus accumulation/reduction rounding.
constexpr ReductionCriterion tolerance_for(ActivationCompute activation_compute) {
    switch (activation_compute) {
    case ActivationCompute::A16:
        return {kBf16UnitRoundoff, kBf16UnitRoundoff, 2.0 * kBf16UnitRoundoff};
    case ActivationCompute::A8:
        return {kA8QuantizationAllowance, kBf16UnitRoundoff, 1.5 * kA8QuantizationAllowance};
    case ActivationCompute::A4:
        return {kA4QuantizationAllowance, kBf16UnitRoundoff, kA4QuantizationAllowance};
    }
    throw std::invalid_argument("linear test: unknown activation compute path");
}

std::size_t checked_elements(std::int32_t first, std::int32_t second, const char* label) {
    if (first <= 0 || second <= 0) {
        throw std::invalid_argument(std::string("linear test: invalid ") + label + " extent");
    }
    const auto a = static_cast<std::size_t>(first);
    const auto b = static_cast<std::size_t>(second);
    if (a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error(std::string("linear test: ") + label + " size overflow");
    }
    return a * b;
}

class GuardedOutput {
public:
    explicit GuardedOutput(std::size_t words)
        : storage_(words * sizeof(std::uint16_t), kOutputGuardBytes, kOutputGuardByte) {
        poison();
    }

    void* data() { return storage_.data(); }

    const void* data() const { return storage_.data(); }

    void poison() { storage_.fill(kOutputPoisonByte); }

    int verify_guards(std::string_view label) const { return storage_.verify_guards(label); }

private:
    GuardedDeviceBuffer storage_;
};

std::vector<std::int32_t> all_indices(std::int32_t extent) {
    std::vector<std::int32_t> indices(static_cast<std::size_t>(extent));
    std::iota(indices.begin(), indices.end(), 0);
    return indices;
}

std::vector<std::int32_t> sampled_indices(std::int32_t extent) {
    std::vector<std::int32_t> result;
    constexpr std::int32_t kSamples = 32;
    for (std::int32_t sample = 0; sample < kSamples; ++sample) {
        const std::int32_t index = static_cast<std::int32_t>(
            (static_cast<std::int64_t>(extent - 1) * sample) / (kSamples - 1));
        if (index >= 0 && index < extent &&
            std::find(result.begin(), result.end(), index) == result.end()) {
            result.push_back(index);
        }
    }
    std::sort(result.begin(), result.end());
    return result;
}

std::vector<std::uint16_t> make_activation(std::int32_t k, std::int32_t t, std::uint32_t seed,
                                           ActivationCompute activation_compute) {
    const std::size_t elements = checked_elements(k, t, "activation");
    std::vector<std::uint16_t> result(elements);
    for (std::int32_t token = 0; token < t; ++token) {
        for (std::int32_t column = 0; column < k; ++column) {
            std::uint32_t coordinate = seed ^ (static_cast<std::uint32_t>(column) * 0x9e3779b9U) ^
                                       (static_cast<std::uint32_t>(token) * 0x85ebca6bU);
            if (activation_compute == ActivationCompute::A16) {
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
                   std::span<const double> reference, ActivationCompute activation_compute) {
    return verify_reduction(label, actual, reference, tolerance_for(activation_compute));
}

} // namespace

quantized_weight::PackedWeight make_q4g64_f16s_weight(std::int32_t n, std::int32_t k,
                                                      std::uint32_t seed) {
    return quantized_weight::make_patterned_weight(QType::Q4G64_F16S, n, k, seed,
                                                   {quantized_weight::RowSplitScalePattern::Small,
                                                    quantized_weight::RowSplitCodePattern::Hashed});
}

quantized_weight::PackedWeight make_q5g64_f16s_weight(std::int32_t n, std::int32_t k,
                                                      std::uint32_t seed) {
    return quantized_weight::make_patterned_weight(QType::Q5G64_F16S, n, k, seed,
                                                   {quantized_weight::RowSplitScalePattern::Small,
                                                    quantized_weight::RowSplitCodePattern::Hashed});
}

quantized_weight::PackedWeight make_q6g64_f16s_weight(std::int32_t n, std::int32_t k,
                                                      std::uint32_t seed) {
    return quantized_weight::make_patterned_weight(QType::Q6G64_F16S, n, k, seed,
                                                   {quantized_weight::RowSplitScalePattern::Small,
                                                    quantized_weight::RowSplitCodePattern::Hashed});
}

quantized_weight::PackedWeight make_w8g32_f16s_weight(std::int32_t n, std::int32_t k,
                                                      std::uint32_t seed) {
    return quantized_weight::make_patterned_weight(QType::W8G32_F16S, n, k, seed,
                                                   {quantized_weight::RowSplitScalePattern::Small,
                                                    quantized_weight::RowSplitCodePattern::Hashed});
}

quantized_weight::PackedWeight make_nvfp4_weight(std::int32_t n, std::int32_t k,
                                                 std::uint32_t seed) {
    quantized_weight::PatternedWeightOptions options;
    options.weight_scale_divisor = 0.125F;
    options.input_scale_divisor  = 3.5F;
    return quantized_weight::make_patterned_weight(QType::NVFP4, n, k, seed, options);
}

quantized_weight::PackedWeight make_fp8_weight(std::int32_t n, std::int32_t k, std::uint32_t seed) {
    return quantized_weight::make_patterned_weight(QType::FP8_E4M3FN_ROW_BF16S, n, k, seed);
}

void cpu_linear_gemm_fp64(const float* weight, const float* activation, double* output,
                          std::int32_t n, std::int32_t k, std::int32_t t) {
    if (weight == nullptr || activation == nullptr || output == nullptr || n <= 0 || k <= 0 ||
        t <= 0) {
        throw std::invalid_argument("linear test: invalid FP64 GEMM argument");
    }

    const unsigned hardware_threads = std::max(1U, std::thread::hardware_concurrency());
    const std::int32_t thread_count = std::min(n, static_cast<std::int32_t>(hardware_threads));
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(thread_count));
    for (std::int32_t thread = 0; thread < thread_count; ++thread) {
        const std::int32_t row_begin =
            static_cast<std::int32_t>((static_cast<std::int64_t>(n) * thread) / thread_count);
        const std::int32_t row_end =
            static_cast<std::int32_t>((static_cast<std::int64_t>(n) * (thread + 1)) / thread_count);
        workers.emplace_back([=] {
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
    for (std::thread& worker : workers) { worker.join(); }
}

bool cuda_available() { return !test::cuda_unavailable(); }

int run_shape(std::string_view label, ActivationCompute activation_compute,
              WeightGenerator generator, const ShapeCase& shape) {
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
        shape.comparison == Comparison::Full ? all_indices(shape.n) : sampled_indices(shape.n);
    quantized_weight::PackedWeight host_weight = generator(shape.n, shape.k, shape.seed);
    const std::vector<float> oracle_weight =
        quantized_weight::materialize_rows_fp32(host_weight, oracle_rows);
    const std::vector<std::uint16_t> activation_bits =
        make_activation(shape.k, maximum->t, shape.seed + 1U, activation_compute);

    DeviceBuffer device_activation(activation_bits.size() * sizeof(std::uint16_t));
    device_activation.copy_from_host(activation_bits.data(), device_activation.bytes);
    DeviceBuffer device_weight(host_weight.payload.size());
    device_weight.copy_from_host(host_weight.payload.data(), device_weight.bytes);
    const Weight weight = host_weight.device_weight(device_weight.p);

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
        const std::string case_label = std::string(label) + " [" + std::to_string(shape.n) + "," +
                                       std::to_string(shape.k) +
                                       "] T=" + std::to_string(invocation.t);
        GuardedOutput output(checked_elements(shape.n, invocation.t, "guarded output"));
        Tensor input(device_activation.p, DType::BF16, {shape.k, invocation.t});
        Tensor destination(output.data(), DType::BF16, {shape.n, invocation.t});
        const std::size_t capacity = ops::linear_workspace_capacity_bytes(
            weight.qtype, shape.n, shape.k, invocation.policy, invocation.t, invocation.t);
        DeviceArena workspace(std::max<std::size_t>(capacity, 256));
        std::unique_ptr<DeviceContext> graph_context;
        DecodeGraphDefinition graph_definition;
        DecodeGraphExecutable graph;
        if (invocation.graph_replay) graph_context = std::make_unique<DeviceContext>();
        const cudaStream_t stream = graph_context ? graph_context->stream : nullptr;
        const auto launch         = [&] {
            if (invocation.call_form == CallForm::A16Convenience) {
                ops::linear(input, weight, destination, stream);
            } else {
                ops::linear(input, weight, destination, invocation.policy, workspace, stream);
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
                    device_activation.copy_from_host(negative.data(), device_activation.bytes);
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
                const std::vector<std::int32_t> columns = shape.comparison == Comparison::Full
                                                              ? all_indices(invocation.t)
                                                              : sampled_indices(invocation.t);
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
                        compare_output(case_label, actual.selected, reference, activation_compute);
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
                        compare_output(case_label, actual.selected, reference, activation_compute);
                }
                if (replay == 1) {
                    std::vector<std::uint16_t> after(activation_bits.size());
                    device_activation.copy_to_host(after.data(), device_activation.bytes);
                    for (std::size_t i = 0; i < after.size(); ++i) {
                        if (after[i] != (activation_bits[i] ^ 0x8000)) {
                            std::cerr << case_label << ": modified graph activation input\n";
                            ++failures;
                            break;
                        }
                    }
                    device_activation.copy_from_host(activation_bits.data(),
                                                     device_activation.bytes);
                }
            }
        } catch (const std::exception& error) {
            std::cerr << case_label << ": unexpected exception: " << error.what() << '\n';
            ++failures;
        }
    }

    if (shape.verify_input_preservation) {
        std::vector<std::uint16_t> activation_after(activation_bits.size());
        device_activation.copy_to_host(activation_after.data(), device_activation.bytes);
        if (activation_after != activation_bits) {
            std::cerr << label << ": linear modified its activation input\n";
            ++failures;
        }
        std::vector<std::uint8_t> weight_after(host_weight.payload.size());
        device_weight.copy_to_host(weight_after.data(), device_weight.bytes);
        if (weight_after != host_weight.payload) {
            std::cerr << label << ": linear modified its persistent weight\n";
            ++failures;
        }
    }
    return failures;
}

} // namespace ninfer::test::linear
