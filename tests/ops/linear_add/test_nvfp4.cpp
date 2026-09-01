#include "ninfer/ops/linear_add.h"

#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test;

constexpr double kBf16UnitRoundoff = 1.0 / 256.0;
constexpr ReductionCriterion kA16Tolerance{
    kBf16UnitRoundoff,
    kBf16UnitRoundoff,
    2.0 * kBf16UnitRoundoff,
};
constexpr ReductionCriterion kA4Tolerance{0.16, kBf16UnitRoundoff, 0.16};

struct Invocation {
    std::int32_t tokens;
    ops::LinearPolicy policy;
};

std::vector<std::int32_t> sampled_indices(std::int32_t extent) {
    std::vector<std::int32_t> result;
    for (const std::int32_t index :
         {0, 1, extent / 4, extent / 2, (3 * extent) / 4, extent - 2, extent - 1}) {
        if (index >= 0 && index < extent &&
            std::find(result.begin(), result.end(), index) == result.end()) {
            result.push_back(index);
        }
    }
    return result;
}

std::vector<std::uint16_t> make_activation(std::int32_t rows, std::int32_t tokens,
                                           std::uint32_t seed) {
    std::vector<std::uint16_t> result(static_cast<std::size_t>(rows) * tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t row = 0; row < rows; ++row) {
            std::uint32_t value = seed ^ (static_cast<std::uint32_t>(row) * 0x9e3779b9U) ^
                                  (static_cast<std::uint32_t>(token) * 0x85ebca6bU);
            value ^= value >> 16;
            value *= 0x7feb352dU;
            value ^= value >> 15;
            const float represented =
                static_cast<float>(static_cast<int>(value & 0xffU) - 128) * (1.0F / 256.0F);
            result[static_cast<std::size_t>(token) * rows + row] = f32_to_bf16(represented);
        }
    }
    return result;
}

std::vector<std::uint16_t> make_residual(std::int32_t rows, std::int32_t tokens,
                                         std::uint32_t seed) {
    std::vector<std::uint16_t> result(static_cast<std::size_t>(rows) * tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t row = 0; row < rows; ++row) {
            const std::uint32_t coordinate = static_cast<std::uint32_t>(row) * 23U +
                                             static_cast<std::uint32_t>(token) * 41U + seed * 7U;
            const float represented =
                static_cast<float>(static_cast<int>(coordinate & 0xffU) - 128) * (1.0F / 128.0F);
            result[static_cast<std::size_t>(token) * rows + row] = f32_to_bf16(represented);
        }
    }
    return result;
}

int verify_preserved(const GuardedDeviceBuffer& device, std::span<const std::uint8_t> expected,
                     std::string_view label) {
    std::vector<std::uint8_t> actual(expected.size());
    device.copy_to_host(actual.data(), actual.size());
    if (std::equal(actual.begin(), actual.end(), expected.begin(), expected.end())) { return 0; }
    std::cerr << label << ": payload was modified\n";
    return 1;
}

int run_shape(std::int32_t n, std::int32_t k, std::uint32_t seed) {
#ifdef NINFER_VOLTA_BUILD
    const std::array invocations{
        Invocation{1, ops::LinearPolicy::A16Only},
        Invocation{4, ops::LinearPolicy::A16Only},
    };
#else
    const std::int32_t first_a4 = k == 6144 ? 7 : 8;
    const std::array invocations{
        Invocation{1, ops::LinearPolicy::A16Only},
        Invocation{4, ops::LinearPolicy::A16Only},
        Invocation{first_a4, ops::LinearPolicy::AllowA4},
        Invocation{17, ops::LinearPolicy::AllowA4},
        Invocation{1024, ops::LinearPolicy::AllowA4},
    };
#endif
    constexpr std::int32_t kMaximumTokens = 1024;
    quantized_weight::PatternedWeightOptions options;
    options.weight_scale_divisor = 0.125F;
    options.input_scale_divisor  = 3.5F;
    quantized_weight::PackedWeight host_weight =
        quantized_weight::make_patterned_weight(QType::NVFP4, n, k, seed, options);
    const std::vector<std::int32_t> rows = sampled_indices(n);
    const std::vector<float> materialized_weight =
        quantized_weight::materialize_rows_fp32(host_weight, rows);
    const std::vector<std::uint16_t> activation = make_activation(k, kMaximumTokens, seed + 1U);
    const std::vector<std::uint16_t> initial_residual = make_residual(n, kMaximumTokens, seed + 2U);

    GuardedDeviceBuffer device_activation(activation.size() * sizeof(std::uint16_t));
    device_activation.copy_from_host(activation.data(), device_activation.bytes());
    GuardedDeviceBuffer device_weight(host_weight.payload.size());
    device_weight.copy_from_host(host_weight.payload.data(), host_weight.payload.size());
    const Weight weight = host_weight.device_weight(device_weight.data());

    int failures = 0;
    for (const Invocation invocation : invocations) {
        const std::size_t output_words = static_cast<std::size_t>(n) * invocation.tokens;
        GuardedDeviceBuffer output(output_words * sizeof(std::uint16_t));
        output.copy_from_host(initial_residual.data(), output.bytes());
        Tensor x(device_activation.data(), DType::BF16, {k, invocation.tokens});
        Tensor residual(output.data(), DType::BF16, {n, invocation.tokens});
        const std::size_t capacity = ops::linear_add_workspace_capacity_bytes(
            QType::NVFP4, n, k, invocation.policy, invocation.tokens, invocation.tokens);
        WorkspaceArena workspace(std::max<std::size_t>(capacity, 256));
        ops::linear_add(x, weight, residual, invocation.policy, workspace, nullptr);
        cuda_check(cudaDeviceSynchronize(), "synchronize NVFP4 linear_add");

        const bool a4           = invocation.policy == ops::LinearPolicy::AllowA4;
        const std::string label = "NVFP4 linear_add [" + std::to_string(n) + "," +
                                  std::to_string(k) + "] " + (a4 ? "A4" : "A16") +
                                  " T=" + std::to_string(invocation.tokens);
        if (workspace.peak_used() != capacity) {
            std::cerr << label << ": workspace query/execution high-water mismatch\n";
            ++failures;
        }
        failures += output.verify_guards(label);

        std::vector<std::uint16_t> actual_bits(output_words);
        output.copy_to_host(actual_bits.data(), output.bytes());
        const std::vector<std::int32_t> tokens = sampled_indices(invocation.tokens);
        std::vector<double> actual;
        std::vector<double> expected;
        actual.reserve(rows.size() * tokens.size());
        expected.reserve(rows.size() * tokens.size());
        for (std::size_t sampled_row = 0; sampled_row < rows.size(); ++sampled_row) {
            const std::int32_t row = rows[sampled_row];
            const float* weight_row =
                materialized_weight.data() + sampled_row * static_cast<std::size_t>(k);
            for (const std::int32_t token : tokens) {
                double sum = 0.0;
                const std::uint16_t* activation_row =
                    activation.data() + static_cast<std::size_t>(token) * k;
                for (std::int32_t column = 0; column < k; ++column) {
                    sum += static_cast<double>(weight_row[column]) *
                           static_cast<double>(bf16_to_f32(activation_row[column]));
                }
                const std::size_t index = static_cast<std::size_t>(token) * n + row;
                actual.push_back(static_cast<double>(bf16_to_f32(actual_bits[index])));
                expected.push_back(sum + static_cast<double>(bf16_to_f32(initial_residual[index])));
            }
        }
        failures += verify_reduction(label, actual, expected, a4 ? kA4Tolerance : kA16Tolerance);
    }

    failures += device_activation.verify_guards("NVFP4 linear_add activation");
    failures += device_weight.verify_guards("NVFP4 linear_add weight");
    failures += verify_preserved(
        device_activation,
        std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(activation.data()),
                                      activation.size() * sizeof(std::uint16_t)),
        "NVFP4 linear_add activation");
    failures += verify_preserved(device_weight, host_weight.payload, "NVFP4 linear_add weight");
    return failures;
}

} // namespace

int main() {
    if (ninfer::test::cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    failures += run_shape(5120, 6144, 811U);
    failures += run_shape(5120, 17408, 821U);
    std::cout << (failures == 0 ? "OK" : "FAIL") << " NVFP4 linear_add\n";
    return failures == 0 ? 0 : 1;
}
