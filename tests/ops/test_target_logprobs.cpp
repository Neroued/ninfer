#include "ninfer/ops/target_logprobs.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr ReductionCriterion kTargetLogprobsFp32Criterion{
    /*relative_l2=*/2.0e-5,
    /*gross_absolute=*/2.0e-4,
    /*gross_relative_to_max_reference=*/0.0,
};

std::vector<std::int32_t> make_targets(std::int32_t valid_rows, std::int32_t columns) {
    std::vector<std::int32_t> targets(static_cast<std::size_t>(columns));
    for (std::int32_t column = 0; column < columns; ++column) {
        if (column % 4 == 0) {
            targets[static_cast<std::size_t>(column)] = 0;
        } else if (column % 4 == 1) {
            targets[static_cast<std::size_t>(column)] = valid_rows - 1;
        } else {
            targets[static_cast<std::size_t>(column)] =
                static_cast<std::int32_t>((static_cast<std::uint64_t>(column + 1) * 7919u) %
                                          static_cast<std::uint32_t>(valid_rows));
        }
    }
    return targets;
}

std::vector<std::uint16_t> make_random_logits(std::int32_t physical_rows, std::int32_t valid_rows,
                                              std::int32_t columns) {
    std::vector<std::uint16_t> logits(static_cast<std::size_t>(physical_rows) * columns);
    for (std::int32_t column = 0; column < columns; ++column) {
        const std::size_t base = static_cast<std::size_t>(column) * physical_rows;
        for (std::int32_t row = 0; row < valid_rows; ++row) {
            const std::uint32_t mixed = static_cast<std::uint32_t>(row) * 1664525u +
                                        static_cast<std::uint32_t>(column + 1) * 1013904223u;
            const float value = -24.0f + static_cast<float>(mixed % 6144u) * (1.0f / 128.0f);
            logits[base + static_cast<std::size_t>(row)] = f32_to_bf16(value);
        }
        for (std::int32_t row = valid_rows; row < physical_rows; ++row) {
            logits[base + static_cast<std::size_t>(row)] = f32_to_bf16(96.0f);
        }
    }
    return logits;
}

std::vector<std::uint16_t> make_uniform_logits(std::int32_t physical_rows, std::int32_t valid_rows,
                                               std::int32_t columns, float value) {
    std::vector<std::uint16_t> logits(static_cast<std::size_t>(physical_rows) * columns,
                                      f32_to_bf16(value));
    for (std::int32_t column = 0; column < columns; ++column) {
        const std::size_t base = static_cast<std::size_t>(column) * physical_rows;
        for (std::int32_t row = valid_rows; row < physical_rows; ++row) {
            logits[base + static_cast<std::size_t>(row)] = f32_to_bf16(112.0f);
        }
    }
    return logits;
}

std::vector<std::uint16_t> make_shift_logits(std::int32_t physical_rows, std::int32_t valid_rows,
                                             std::int32_t columns, float shift) {
    std::vector<std::uint16_t> logits(static_cast<std::size_t>(physical_rows) * columns);
    for (std::int32_t column = 0; column < columns; ++column) {
        const std::size_t base = static_cast<std::size_t>(column) * physical_rows;
        for (std::int32_t row = 0; row < valid_rows; ++row) {
            const int centered = (row * 37 + column * 11) % 65 - 32;
            const float value  = static_cast<float>(centered) * 0.25f + shift;
            logits[base + static_cast<std::size_t>(row)] = f32_to_bf16(value);
        }
        for (std::int32_t row = valid_rows; row < physical_rows; ++row) {
            logits[base + static_cast<std::size_t>(row)] = f32_to_bf16(120.0f);
        }
    }
    return logits;
}

std::vector<std::uint16_t> make_extreme_logits(std::int32_t physical_rows, std::int32_t valid_rows,
                                               std::int32_t columns) {
    std::vector<std::uint16_t> logits(static_cast<std::size_t>(physical_rows) * columns);
    constexpr float values[] = {-80.0f, -32.0f, -1.0f, 0.0f, 1.0f, 32.0f, 80.0f};
    for (std::int32_t column = 0; column < columns; ++column) {
        const std::size_t base = static_cast<std::size_t>(column) * physical_rows;
        for (std::int32_t row = 0; row < valid_rows; ++row) {
            const auto index = static_cast<std::size_t>(row + column) % std::size(values);
            logits[base + static_cast<std::size_t>(row)] = f32_to_bf16(values[index]);
        }
        for (std::int32_t row = valid_rows; row < physical_rows; ++row) {
            logits[base + static_cast<std::size_t>(row)] = f32_to_bf16(120.0f);
        }
    }
    return logits;
}

std::vector<double> target_logprobs_oracle(const std::vector<std::uint16_t>& logits,
                                           const std::vector<std::int32_t>& targets,
                                           std::int32_t physical_rows, std::int32_t valid_rows) {
    std::vector<double> expected(targets.size());
    for (std::size_t column = 0; column < targets.size(); ++column) {
        const std::size_t base = column * static_cast<std::size_t>(physical_rows);
        double maximum         = -std::numeric_limits<double>::infinity();
        for (std::int32_t row = 0; row < valid_rows; ++row) {
            maximum = std::max(maximum, static_cast<double>(bf16_to_f32(logits[base + row])));
        }
        double sum = 0.0;
        for (std::int32_t row = 0; row < valid_rows; ++row) {
            sum += std::exp(static_cast<double>(bf16_to_f32(logits[base + row])) - maximum);
        }
        const double target = static_cast<double>(bf16_to_f32(logits[base + targets[column]]));
        expected[column]    = target - maximum - std::log(sum);
    }
    return expected;
}

std::vector<double> fp32_as_double(const void* device, std::size_t count) {
    const auto values = from_device<float>(device, count);
    return {values.begin(), values.end()};
}

int run_case(const std::string& label, std::int32_t physical_rows, std::int32_t valid_rows,
             std::int32_t columns, const std::vector<std::uint16_t>& logits) {
    const auto targets  = make_targets(valid_rows, columns);
    const auto expected = target_logprobs_oracle(logits, targets, physical_rows, valid_rows);

    GuardedDeviceBuffer device_logits(logits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_targets(targets.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer device_output(targets.size() * sizeof(float));
    device_logits.copy_from_host(logits.data(), device_logits.bytes());
    device_targets.copy_from_host(targets.data(), device_targets.bytes());
    device_output.fill(0xcd);

    Tensor logits_tensor(device_logits.data(), DType::BF16, {physical_rows, columns});
    Tensor targets_tensor(device_targets.data(), DType::I32, {columns});
    Tensor output_tensor(device_output.data(), DType::FP32, {columns});
    ops::target_logprobs(logits_tensor, targets_tensor, valid_rows, output_tensor, nullptr);
    cuda_synchronize();

    int failures = verify_reduction(label, fp32_as_double(device_output.data(), targets.size()),
                                    expected, kTargetLogprobsFp32Criterion);
    failures +=
        verify_exact((label + " preserves logits").c_str(),
                     from_device<std::uint16_t>(device_logits.data(), logits.size()), logits);
    failures +=
        verify_exact((label + " preserves targets").c_str(),
                     from_device<std::int32_t>(device_targets.data(), targets.size()), targets);
    failures += device_logits.verify_guards(label + " logits guards");
    failures += device_targets.verify_guards(label + " target guards");
    failures += device_output.verify_guards(label + " output guards");
    return failures;
}

template <class Function>
int expect_invalid(const char* label, Function&& function) {
    try {
        function();
    } catch (const std::invalid_argument&) { return 0; } catch (const std::exception& error) {
        std::cerr << label << ": expected invalid_argument, got " << error.what() << '\n';
        return 1;
    }
    std::cerr << label << ": expected invalid_argument\n";
    return 1;
}

int run_validation_cases() {
    DeviceBuffer logits_data(8 * 3 * sizeof(std::uint16_t));
    DeviceBuffer targets_data(3 * sizeof(std::int32_t));
    DeviceBuffer output_data(3 * sizeof(float));
    Tensor logits(logits_data.p, DType::BF16, {8, 3});
    Tensor targets(targets_data.p, DType::I32, {3});
    Tensor output(output_data.p, DType::FP32, {3});

    int failures = 0;
    failures += expect_invalid("target_logprobs rejects valid_rows=0",
                               [&] { ops::target_logprobs(logits, targets, 0, output, nullptr); });
    failures += expect_invalid("target_logprobs rejects valid_rows>physical_rows",
                               [&] { ops::target_logprobs(logits, targets, 9, output, nullptr); });
    failures += expect_invalid("target_logprobs rejects target shape mismatch", [&] {
        Tensor wrong_targets(targets_data.p, DType::I32, {2});
        ops::target_logprobs(logits, wrong_targets, 8, output, nullptr);
    });
    failures += expect_invalid("target_logprobs rejects output dtype", [&] {
        Tensor wrong_output(output_data.p, DType::BF16, {3});
        ops::target_logprobs(logits, targets, 8, wrong_output, nullptr);
    });
    failures += expect_invalid("target_logprobs rejects non-contiguous logits", [&] {
        Tensor strided_logits = logits;
        strided_logits.nb[1] += 2;
        ops::target_logprobs(strided_logits, targets, 8, output, nullptr);
    });
    failures += expect_invalid("target_logprobs rejects null output", [&] {
        Tensor null_output(nullptr, DType::FP32, {3});
        ops::target_logprobs(logits, targets, 8, null_output, nullptr);
    });
    failures += expect_invalid("target_logprobs rejects output alias", [&] {
        Tensor alias_output(logits_data.p, DType::FP32, {3});
        ops::target_logprobs(logits, targets, 8, alias_output, nullptr);
    });
    failures += expect_invalid("target_logprobs rejects non-matrix logits", [&] {
        Tensor rank_three = logits;
        rank_three.ne[2]  = 2;
        ops::target_logprobs(rank_three, targets, 8, output, nullptr);
    });
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    failures += run_case("target_logprobs full vocabulary C=3", 248320, 248077, 3,
                         make_random_logits(248320, 248077, 3));
    failures += run_case("target_logprobs non-aligned rows C=1025", 523, 509, 1025,
                         make_random_logits(523, 509, 1025));
    failures += run_case("target_logprobs uniform logits", 263, 257, 1024,
                         make_uniform_logits(263, 257, 1024, 3.5f));
    failures +=
        run_case("target_logprobs one valid row", 13, 1, 7, make_uniform_logits(13, 1, 7, -7.0f));
    failures += run_case("target_logprobs extreme finite logits", 263, 257, 17,
                         make_extreme_logits(263, 257, 17));
    failures += run_case("target_logprobs base for constant shift", 257, 257, 31,
                         make_shift_logits(257, 257, 31, 0.0f));
    failures += run_case("target_logprobs shifted logits", 257, 257, 31,
                         make_shift_logits(257, 257, 31, 32.0f));
    failures += run_validation_cases();

    std::cout << (failures ? "FAIL" : "OK") << " target_logprobs\n";
    return failures ? 1 : 0;
}
