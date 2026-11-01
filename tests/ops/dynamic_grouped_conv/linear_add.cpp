#include "ops/suite.h"
#include "core/weight.h"
#include "cuda_fixture.h"
#include "core/buffer.h"
#include "ninfer/ops/dynamic_grouped_conv.h"
#include "core/decode_graph.h"
#include "core/device.h"

#include "ops/input_projection_test_common.h"
#include "ops/op_tester.h"
#include "ops/linear_common.h"
#include "ops/quantized_weight.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <string>
#include <utility>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kHidden       = 5120;
constexpr std::int32_t kMaximumWidth = 16;
constexpr std::int32_t kGroups       = 320;
constexpr std::int32_t kTaps         = 2;
constexpr std::int32_t kSides        = 2;
constexpr std::int32_t kMaximumBatch = 8;
constexpr std::int32_t kMaximumCols  = kMaximumWidth * kMaximumBatch;

constexpr std::array<std::int32_t, 27> kSampledRows{
    0,   7,   8,   15,  16,   17,   31,   32,   63,   64,   127,  128,  255, 256,
    319, 320, 639, 640, 1023, 1024, 2047, 2048, 2559, 2560, 4095, 4096, 5119};
constexpr std::array<std::pair<int, int>, 6> kGraphCases{
    {{3, 3}, {13, 5}, {11, 7}, {14, 6}, {15, 6}, {16, 8}}};
constexpr ReductionCriterion kCriterion{/*relative_l2=*/3.2e-3,
                                        /*gross_absolute=*/1.0e-3,
                                        /*gross_relative=*/2.0e-3};

std::uint32_t mix32(std::uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    return value ^ (value >> 16);
}

float signed_pattern(std::uint32_t index, std::uint32_t seed, float scale) {
    const std::uint32_t mixed = mix32(index ^ (seed * 0x9e3779b9U));
    int centered              = static_cast<int>((mixed >> 8) & 0xffU) - 128;
    if (centered == 0) { centered = (index & 1U) == 0 ? 1 : -1; }
    return static_cast<float>(centered) * scale;
}

std::vector<float> make_activation(std::int32_t input_rows) {
    std::vector<float> result(static_cast<std::size_t>(input_rows) * kMaximumCols);
    for (std::int32_t col = 0; col < kMaximumCols; ++col) {
        for (std::int32_t row = 0; row < input_rows; ++row) {
            result[static_cast<std::size_t>(col) * input_rows + row] = signed_pattern(
                static_cast<std::uint32_t>(static_cast<std::uint64_t>(col) * input_rows + row),
                101U + static_cast<std::uint32_t>(input_rows), 1.0F / 32768.0F);
        }
    }
    round_to_bf16(result);
    return result;
}

std::vector<std::uint16_t> make_base_kernel() {
    std::vector<std::uint16_t> result(static_cast<std::size_t>(kHidden) * kTaps * kSides);
    for (std::int32_t side = 0; side < kSides; ++side) {
        for (std::int32_t tap = 0; tap < kTaps; ++tap) {
            for (std::int32_t row = 0; row < kHidden; ++row) {
                const float center =
                    side == 1 ? (tap == 0 ? 0.375F : -0.15625F) : (tap == 0 ? -0.25F : 0.0625F);
                const float variation =
                    signed_pattern(static_cast<std::uint32_t>((side * kTaps + tap) * kHidden + row),
                                   211U, 1.0F / 32768.0F);
                result[static_cast<std::size_t>(side * kTaps + tap) * kHidden + row] =
                    f32_to_bf16(center + variation);
            }
        }
    }
    return result;
}

std::vector<std::uint16_t> make_finish_delta() {
    std::vector<std::uint16_t> result(static_cast<std::size_t>(kGroups) * kTaps * kMaximumCols);
    for (std::int32_t col = 0; col < kMaximumCols; ++col) {
        for (std::int32_t tap = 0; tap < kTaps; ++tap) {
            for (std::int32_t group = 0; group < kGroups; ++group) {
                const float center    = tap == 0 ? 0.046875F : -0.03125F;
                const float variation = signed_pattern(
                    static_cast<std::uint32_t>((col * kTaps + tap) * kGroups + group), 307U,
                    1.0F / 65536.0F);
                result[(static_cast<std::size_t>(col) * kTaps + tap) * kGroups + group] =
                    f32_to_bf16(center + variation);
            }
        }
    }
    return result;
}

std::vector<std::uint16_t> make_residual() {
    std::vector<std::uint16_t> result(static_cast<std::size_t>(kHidden) * kMaximumCols);
    for (std::int32_t col = 0; col < kMaximumCols; ++col) {
        for (std::int32_t row = 0; row < kHidden; ++row) {
            const float value = signed_pattern(
                static_cast<std::uint32_t>(static_cast<std::uint64_t>(col) * kHidden + row), 401U,
                1.0F / 4096.0F);
            result[static_cast<std::size_t>(col) * kHidden + row] = f32_to_bf16(value);
        }
    }
    return result;
}

std::vector<double> projection_oracle(const quantized_weight::PackedWeight& weight,
                                      const std::vector<float>& activation,
                                      std::int32_t input_rows) {
    std::vector<double> projection(kSampledRows.size() * kMaximumCols);
    for (std::size_t sample = 0; sample < kSampledRows.size(); ++sample) {
        for (std::int32_t col = 0; col < kMaximumCols; ++col) {
            projection[sample * kMaximumCols + col] = quantized_weight::dot_fp64(
                weight, kSampledRows[sample],
                activation.data() + static_cast<std::size_t>(col) * input_rows, input_rows);
        }
    }
    return projection;
}

std::vector<double> result_oracle(const std::vector<double>& projection,
                                  const std::vector<std::uint16_t>& base_kernel,
                                  const std::vector<std::uint16_t>& finish_delta,
                                  const std::vector<std::uint16_t>& residual, std::int32_t width,
                                  std::int32_t batch_size) {
    const std::int32_t cols = width * batch_size;
    std::vector<double> result;
    result.reserve(kSampledRows.size() * static_cast<std::size_t>(cols));
    for (std::size_t sample = 0; sample < kSampledRows.size(); ++sample) {
        const std::int32_t row   = kSampledRows[sample];
        const std::int32_t group = row / 16;
        for (std::int32_t col = 0; col < cols; ++col) {
            const std::size_t residual_offset = static_cast<std::size_t>(col) * kHidden + row;
            const std::size_t delta0_offset =
                (static_cast<std::size_t>(col) * kTaps) * kGroups + group;
            const double current_coefficient =
                static_cast<double>(bf16_to_f32(base_kernel[2 * kHidden + row])) +
                static_cast<double>(bf16_to_f32(finish_delta[delta0_offset]));
            double value = static_cast<double>(bf16_to_f32(residual[residual_offset])) +
                           current_coefficient * projection[sample * kMaximumCols + col];
            if ((col % width) != 0) {
                const double previous_coefficient =
                    static_cast<double>(bf16_to_f32(base_kernel[3 * kHidden + row])) +
                    static_cast<double>(bf16_to_f32(finish_delta[delta0_offset + kGroups]));
                value += previous_coefficient * projection[sample * kMaximumCols + col - 1];
            }
            result.push_back(value);
        }
    }
    return result;
}

std::vector<double> gather_actual(const void* device, std::int32_t width, std::int32_t batch_size) {
    const std::int32_t cols = width * batch_size;
    const std::vector<double> full =
        from_device_bf16(device, static_cast<std::size_t>(kHidden) * cols);
    std::vector<double> result;
    result.reserve(kSampledRows.size() * static_cast<std::size_t>(cols));
    for (const std::int32_t row : kSampledRows) {
        for (std::int32_t col = 0; col < cols; ++col) {
            result.push_back(full[static_cast<std::size_t>(col) * kHidden + row]);
        }
    }
    return result;
}

int run_profile(std::int32_t input_rows, std::span<const std::pair<int, int>> cases) {
    const std::vector<float> activation = make_activation(input_rows);
    std::vector<std::uint16_t> activation_bits(activation.size());
    std::transform(activation.begin(), activation.end(), activation_bits.begin(), f32_to_bf16);
    const std::vector<std::uint16_t> base_kernel  = make_base_kernel();
    const std::vector<std::uint16_t> finish_delta = make_finish_delta();
    const std::vector<std::uint16_t> residual     = make_residual();

    quantized_weight::PatternedWeightOptions weight_options;
    weight_options.row_split_scale = quantized_weight::RowSplitScalePattern::Tiny;
    weight_options.row_split_codes = quantized_weight::RowSplitCodePattern::Hashed;
    input_projection::DevicePackedWeight projection_weight(quantized_weight::make_patterned_weight(
        QType::Q8_G32_FP16, kHidden, input_rows, 503U + static_cast<std::uint32_t>(input_rows),
        weight_options));
    const std::vector<double> projection =
        projection_oracle(projection_weight.host, activation, input_rows);

    DeviceBuffer activation_device = to_device(activation_bits);
    DeviceBuffer base_device       = to_device(base_kernel);
    DeviceBuffer delta_device      = to_device(finish_delta);
    const std::size_t capacity     = ops::linear_dynamic_grouped_conv_add_workspace_capacity_bytes(
        input_rows, 2, 16, 1, kMaximumBatch);

    Tensor base(base_device.span(), DType::BF16, {kHidden, kTaps, kSides});
    const Weight weight = projection_weight.view();
    int failures        = 0;
    for (auto [width, batch_size] : cases) {
        const std::size_t bytes =
            static_cast<std::size_t>(kHidden) * width * batch_size * sizeof(std::uint16_t);
        const auto exact = ops::linear_dynamic_grouped_conv_add_workspace_capacity_bytes(
            input_rows, width, width, batch_size, batch_size);
        if (exact > capacity)
            throw std::runtime_error("workspace interval does not cover exact shape");
        GuardedDeviceBuffer scratch(exact);
        MemoryArena workspace(scratch.span().subspan(0, exact));
        GuardedDeviceBuffer residual_device(bytes);
        Tensor x(activation_device.span(), DType::BF16, {input_rows, width, batch_size});
        Tensor delta(delta_device.span(), DType::BF16, {kGroups, kTaps, width, batch_size});
        Tensor residual_view(residual_device.span(), DType::BF16, {kHidden, width, batch_size});
        const auto expected =
            result_oracle(projection, base_kernel, finish_delta, residual, width, batch_size);
        const bool graph_case = std::find(kGraphCases.begin(), kGraphCases.end(),
                                          std::pair{width, batch_size}) != kGraphCases.end();
        const auto verify     = [&](const std::string& label) {
            failures +=
                verify_reduction(label, gather_actual(residual_device.data(), width, batch_size),
                                     expected, kCriterion);
            failures += residual_device.verify_guards(label);
            failures += scratch.verify_guards(label + " workspace");
            if (workspace.used() != 0 || workspace.peak_used() != exact) {
                std::cerr << label << ": workspace mismatch\n";
                ++failures;
            }
        };
        for (int replay = 0; replay < (graph_case ? 2 : 1); ++replay) {
            residual_device.copy_from_host(residual.data(), bytes);
            workspace.reset_peak();
            const std::string label =
                "dynamic conv add C=" + std::to_string(input_rows) + " W=" + std::to_string(width) +
                " B=" + std::to_string(batch_size) + " graph=" + std::to_string(replay);
            const auto launch = [&](CudaExecutionView cuda) {
                ops::linear_dynamic_grouped_conv_add(x, weight, base, delta, residual_view,
                                                     workspace, cuda);
            };
            if (replay) {
                DeviceContext context;
                cuda_synchronize();
                DecodeGraphDefinition definition;
                DecodeGraphExecutable executable;
                definition.capture(context.stream(), [&] { launch(context.execution_view()); });
                executable.instantiate(definition);
                for (int iteration = 0; iteration < 2; ++iteration) {
                    cuda_check(cudaMemcpyAsync(residual_device.data(), residual.data(), bytes,
                                               cudaMemcpyHostToDevice, context.stream()),
                               "reset graph residual");
                    executable.launch(context.stream());
                    cuda_synchronize(context.stream());
                    verify(label + " phase=" + std::to_string(iteration));
                }
            } else {
                launch(ninfer::test::cuda_environment().default_execution);
                cuda_synchronize();
                verify(label);
            }
        }
    }

    failures += linear_common::verify_preserved(activation_device.data(),
                                                std::as_bytes(std::span(activation_bits)),
                                                "linear dynamic grouped conv add x");
    failures +=
        linear_common::verify_preserved(base_device.data(), std::as_bytes(std::span(base_kernel)),
                                        "linear dynamic grouped conv add base");
    failures +=
        linear_common::verify_preserved(delta_device.data(), std::as_bytes(std::span(finish_delta)),
                                        "linear dynamic grouped conv add delta");
    failures += projection_weight.verify_preserved("linear dynamic grouped conv add weight");
    return failures;
}

} // namespace

void add_linear_conv_add_cases(ninfer::test::OpSuite& suite) {
    for (int input_rows : {4096, 17408}) {
        std::vector<std::pair<int, int>> shapes;
        std::vector<OpCase> cases;
        for (int width = 2; width <= 16; ++width)
            for (int batch = 1; batch <= 8; ++batch) {
                shapes.emplace_back(width, batch);
                const bool graph = std::find(kGraphCases.begin(), kGraphCases.end(),
                                             std::pair{width, batch}) != kGraphCases.end();
                cases.push_back({"linear_conv_add/k" + std::to_string(input_rows) + "/w" +
                                     std::to_string(width) + "/b" + std::to_string(batch),
                                 {{"execution", graph ? "graph" : "eager"}, {"format", "q8"}}});
            }
        suite.add(std::move(cases), [shapes = std::move(shapes), input_rows](auto selected) {
            std::vector<std::pair<int, int>> cases;
            for (auto i : selected) cases.push_back(shapes[i]);
            return run_profile(input_rows, cases);
        });
    }
}
