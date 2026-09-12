#include "core/device.h"
#include "ops/linear_add/linear_add_test_common.h"

#include "ninfer/ops/linear_add.h"
#include "ops/direct_bf16_weight.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

namespace ninfer::test::linear_add {
namespace {

constexpr std::size_t kOutputScanWords = 1U << 20;
constexpr double kBf16UnitRoundoff     = 1.0 / 256.0;

// One criterion for the complete A16 fused Op. It is not selected by T, route, or kernel.
constexpr ReductionCriterion kLinearAddA16Tolerance{
    kBf16UnitRoundoff,
    kBf16UnitRoundoff,
    2.0 * kBf16UnitRoundoff,
};

std::size_t checked_elements(std::int32_t first, std::int32_t second, const char* label) {
    if (first <= 0 || second <= 0) {
        throw std::invalid_argument(std::string("linear_add test: invalid ") + label + " extent");
    }
    const std::size_t a = static_cast<std::size_t>(first);
    const std::size_t b = static_cast<std::size_t>(second);
    if (a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error(std::string("linear_add test: ") + label + " size overflow");
    }
    return a * b;
}

std::vector<std::int32_t> sampled_indices(std::int32_t extent) {
    std::vector<std::int32_t> result;
    for (const std::int32_t index :
         {0, 1, extent / 4, extent / 2, (3 * extent) / 4, extent - 2, extent - 1}) {
        if (index >= 0 && index < extent &&
            std::find(result.begin(), result.end(), index) == result.end()) {
            result.push_back(index);
        }
    }
    std::sort(result.begin(), result.end());
    return result;
}

std::vector<std::int32_t> all_indices(std::int32_t extent) {
    std::vector<std::int32_t> result(static_cast<std::size_t>(extent));
    std::iota(result.begin(), result.end(), 0);
    return result;
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
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

std::vector<std::uint16_t> make_activation(std::int32_t k, std::int32_t t, std::uint32_t seed) {
    std::vector<std::uint16_t> result(checked_elements(k, t, "activation"));
    for (std::int32_t token = 0; token < t; ++token) {
        for (std::int32_t column = 0; column < k; ++column) {
            const std::uint64_t token_block = static_cast<std::uint64_t>(token / 256);
            const std::uint64_t coordinate =
                static_cast<std::uint64_t>(column) * 17U + static_cast<std::uint64_t>(token) * 31U +
                static_cast<std::uint64_t>(seed) * 13U +
                token_block * (static_cast<std::uint64_t>(column / 256) * 13U + 47U);
            const int raw = 32 + static_cast<int>(coordinate & 0x5fU);
            result[static_cast<std::size_t>(token) * k + column] =
                test::f32_to_bf16(static_cast<float>(raw) * (1.0F / 256.0F));
        }
    }
    return result;
}

std::vector<std::uint16_t> make_residual(std::int32_t n, std::int32_t t, std::uint32_t seed) {
    std::vector<std::uint16_t> result(checked_elements(n, t, "residual"));
    for (std::int32_t token = 0; token < t; ++token) {
        for (std::int32_t row = 0; row < n; ++row) {
            const std::uint64_t token_block = static_cast<std::uint64_t>(token / 256);
            const std::uint64_t coordinate =
                static_cast<std::uint64_t>(row) * 23U + static_cast<std::uint64_t>(token) * 41U +
                static_cast<std::uint64_t>(seed) * 7U +
                token_block * (static_cast<std::uint64_t>(row / 256) * 17U + 29U);
            const int raw = static_cast<int>(coordinate & 0xffU);
            result[static_cast<std::size_t>(token) * n + row] =
                test::f32_to_bf16(static_cast<float>(raw - 128) * (1.0F / 128.0F));
        }
    }
    return result;
}

std::vector<double> linear_add_oracle(std::int32_t n, std::int32_t k,
                                      std::span<const std::int32_t> oracle_rows,
                                      std::span<const float> oracle_weight,
                                      const std::vector<std::uint16_t>& activation,
                                      const std::vector<std::uint16_t>& residual,
                                      std::span<const std::int32_t> columns) {
    const std::int32_t oracle_n = static_cast<std::int32_t>(oracle_rows.size());
    std::vector<double> result(
        checked_elements(oracle_n, static_cast<std::int32_t>(columns.size()), "oracle output"));
    for (std::size_t selected_column = 0; selected_column < columns.size(); ++selected_column) {
        const std::int32_t token = columns[selected_column];
        for (std::int32_t oracle_row = 0; oracle_row < oracle_n; ++oracle_row) {
            double sum = 0.0;
            const float* weight_row =
                oracle_weight.data() + static_cast<std::size_t>(oracle_row) * k;
            const std::uint16_t* activation_column =
                activation.data() + static_cast<std::size_t>(token) * k;
            for (std::int32_t column = 0; column < k; ++column) {
                sum += static_cast<double>(weight_row[column]) *
                       static_cast<double>(test::bf16_to_f32(activation_column[column]));
            }
            const std::int32_t physical_row = oracle_rows[static_cast<std::size_t>(oracle_row)];
            // This is the complete fused formula. There is deliberately no private projection
            // materialization or BF16 rounding between GEMM and residual addition.
            result[selected_column * static_cast<std::size_t>(oracle_n) + oracle_row] =
                sum + static_cast<double>(test::bf16_to_f32(
                          residual[static_cast<std::size_t>(token) * n + physical_row]));
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
            wanted.push_back(static_cast<std::size_t>(column) * n + row);
        }
    }

    OutputRead result;
    result.selected.resize(wanted.size());
    std::vector<std::uint16_t> chunk(std::min(kOutputScanWords, total_words));
    std::size_t wanted_index    = 0;
    std::size_t nonfinite_count = 0;
    for (std::size_t begin = 0; begin < total_words; begin += chunk.size()) {
        const std::size_t count = std::min(chunk.size(), total_words - begin);
        test::cuda_check(
            cudaMemcpy(chunk.data(),
                       static_cast<const std::uint8_t*>(device) + begin * sizeof(std::uint16_t),
                       count * sizeof(std::uint16_t), cudaMemcpyDeviceToHost),
            "copy linear_add output");
        for (std::size_t index = 0; index < count; ++index) {
            const std::uint16_t bits = chunk[index];
            if ((bits & 0x7f80U) == 0x7f80U) { ++nonfinite_count; }
        }
        while (wanted_index < wanted.size() && wanted[wanted_index] < begin + count) {
            result.selected[wanted_index] =
                static_cast<double>(test::bf16_to_f32(chunk[wanted[wanted_index] - begin]));
            ++wanted_index;
        }
    }
    if (nonfinite_count != 0) {
        std::cerr << label << ": output contains " << nonfinite_count << " non-finite values\n";
        ++result.failures;
    }
    return result;
}

int compare_output(std::string_view label, std::span<const double> actual,
                   std::span<const double> reference) {
    return verify_reduction(label, actual, reference, kLinearAddA16Tolerance);
}

int verify_preserved(const test::GuardedDeviceBuffer& device,
                     std::span<const std::uint8_t> expected, const char* label) {
    std::vector<std::uint8_t> actual(expected.size());
    device.copy_to_host(actual.data(), actual.size());
    if (std::equal(actual.begin(), actual.end(), expected.begin(), expected.end())) { return 0; }
    std::cerr << label << ": input payload was modified\n";
    return 1;
}

using HostWeight = std::variant<quantized_weight::PackedWeight, direct_bf16_weight::HostWeight>;

QType qtype_for(WeightFormat format) {
    switch (format) {
    case WeightFormat::BF16:
        return QType::BF16_CTRL;
    case WeightFormat::Q5G64F16S:
        return QType::Q5G64_F16S;
    case WeightFormat::W8G32F16S:
        return QType::W8G32_F16S;
    }
    throw std::invalid_argument("linear_add test: unknown weight format");
}

HostWeight make_weight(WeightFormat format, const ShapeCase& shape) {
    if (format == WeightFormat::BF16) {
        return direct_bf16_weight::make_patterned(shape.n, shape.k, shape.seed);
    }
    const quantized_weight::PatternedWeightOptions options{
        format == WeightFormat::Q5G64F16S ? quantized_weight::RowSplitScalePattern::Small
                                          : quantized_weight::RowSplitScalePattern::Tiny,
        quantized_weight::RowSplitCodePattern::Hashed,
    };
    return quantized_weight::make_patterned_weight(qtype_for(format), shape.n, shape.k, shape.seed,
                                                   options);
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

Weight make_device_weight_view(const HostWeight& weight, void* data) {
    return std::visit([&](const auto& value) { return value.device_weight(data); }, weight);
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
                    const std::uint16_t* source =
                        value.bits.data() + static_cast<std::size_t>(rows[selected]) * value.k;
                    float* destination =
                        result.data() + selected * static_cast<std::size_t>(value.k);
                    for (std::int32_t column = 0; column < value.k; ++column) {
                        destination[column] = test::bf16_to_f32(source[column]);
                    }
                }
                return result;
            }
        },
        weight);
}

} // namespace

bool cuda_available() { return !test::cuda_unavailable(); }

int run_shape(std::string_view label, WeightFormat format, const ShapeCase& shape) {
    const std::vector<std::int32_t> tokens = conformance_tokens(shape);
    if (tokens.empty()) { throw std::invalid_argument("linear_add test: no token cases"); }
    const std::int32_t maximum_t = tokens.back();

    const std::vector<std::int32_t> oracle_rows = sampled_indices(shape.n);
    const QType qtype                           = qtype_for(format);
    const HostWeight host_weight                = make_weight(format, shape);
    const std::vector<float> oracle_weight      = materialize_weight_rows(host_weight, oracle_rows);
    const std::vector<std::uint16_t> activation =
        make_activation(shape.k, maximum_t, shape.seed + 1U);
    const std::vector<std::uint16_t> residual = make_residual(shape.n, maximum_t, shape.seed + 2U);
    const std::vector<std::int32_t> all_columns = all_indices(maximum_t);
    const std::vector<double> full_reference    = linear_add_oracle(
        shape.n, shape.k, oracle_rows, oracle_weight, activation, residual, all_columns);

    test::GuardedDeviceBuffer device_activation(activation.size() * sizeof(std::uint16_t));
    device_activation.copy_from_host(activation.data(), device_activation.bytes());
    const std::span<const std::uint8_t> host_payload = weight_payload(host_weight);
    test::GuardedDeviceBuffer device_weight(host_payload.size());
    device_weight.copy_from_host(host_payload.data(), host_payload.size());
    const Weight weight = make_device_weight_view(host_weight, device_weight.data());

    const std::size_t workspace_bytes =
        ops::linear_add_workspace_capacity_bytes(qtype, shape.n, shape.k, 1, maximum_t);
    WorkspaceArena workspace(std::max<std::size_t>(workspace_bytes, 256));

    int failures              = 0;
    std::size_t executed_peak = 0;
    for (const std::int32_t t : tokens) {
        const std::size_t output_words = checked_elements(shape.n, t, "output");
        test::GuardedDeviceBuffer output(output_words * sizeof(std::uint16_t));
        output.copy_from_host(residual.data(), output.bytes());

        Tensor input(device_activation.data(), DType::BF16, {shape.k, t});
        Tensor residual_out(output.data(), DType::BF16, {shape.n, t});
        workspace.reset();
        workspace.reset_peak();

        const std::string case_label = std::string(label) + " [" + std::to_string(shape.n) + "," +
                                       std::to_string(shape.k) + "] T=" + std::to_string(t);
        try {
            ops::linear_add(input, weight, residual_out, workspace, nullptr);
            test::cuda_check(cudaDeviceSynchronize(), "synchronize linear_add");
            if (t == 128 && format == WeightFormat::Q5G64F16S) {
                cudaStream_t stream;
                cudaGraph_t graph;
                cudaGraphExec_t executable;
                CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
                CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
                ops::linear_add(input, weight, residual_out, workspace, stream);
                CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
                CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
                for (int replay = 0; replay < 2; ++replay) {
                    CUDA_CHECK(cudaMemcpyAsync(output.data(), residual.data(), output.bytes(),
                        cudaMemcpyHostToDevice, stream));
                    CUDA_CHECK(cudaGraphLaunch(executable, stream));
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                }
                CUDA_CHECK(cudaGraphExecDestroy(executable));
                CUDA_CHECK(cudaGraphDestroy(graph));
                CUDA_CHECK(cudaStreamDestroy(stream));
            }
        } catch (const std::exception& error) {
            std::cerr << case_label << ": unexpected exception: " << error.what() << '\n';
            ++failures;
            continue;
        }
        const std::size_t exact_workspace =
            ops::linear_add_workspace_capacity_bytes(qtype, shape.n, shape.k, t, t);
        if (workspace.used() != 0 || workspace.peak_used() != exact_workspace) {
            std::cerr << case_label << ": exact workspace query/execution high-water mismatch\n";
            ++failures;
        }
        executed_peak = std::max(executed_peak, workspace.peak_used());

        failures += output.verify_guards(case_label.c_str());
        const std::vector<std::int32_t> columns = all_indices(t);
        const OutputRead actual =
            read_output(output.data(), shape.n, t, oracle_rows, columns, case_label);
        failures += actual.failures;
        failures += compare_output(
            case_label, actual.selected,
            std::span<const double>(
                full_reference.data(),
                checked_elements(static_cast<std::int32_t>(oracle_rows.size()), t, "reference")));
    }
    if (executed_peak != workspace_bytes) {
        std::cerr << label << ": interval workspace capacity has no executed high-water witness\n";
        ++failures;
    }

    failures += device_activation.verify_guards("linear_add activation");
    failures += device_weight.verify_guards("linear_add weight");
    failures += verify_preserved(
        device_activation,
        std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(activation.data()),
                                      activation.size() * sizeof(std::uint16_t)),
        "linear_add activation");
    failures += verify_preserved(device_weight, host_payload, "linear_add weight");
    return failures;
}

} // namespace ninfer::test::linear_add
