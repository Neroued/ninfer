#include "ops/suite.h"
#include "ninfer/ops/gelu.h"
#include "cuda_fixture.h"
#include "ops/op_tester.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr PointwiseCriterion gelu_bf16_criterion() {
    return {/*absolute*/ 2.0e-6, /*relative*/ 4.05e-3};
}

std::vector<std::uint16_t> encode_bf16(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        bits[index] = f32_to_bf16(values[index]);
    }
    return bits;
}

double gelu_oracle(double input, ops::GeluMode mode) {
    if (mode == ops::GeluMode::Exact) {
        return 0.5 * input * (1.0 + std::erf(input / std::sqrt(2.0)));
    }
    constexpr double sqrt_two_over_pi = 0.79788456080286535587989211986876;
    return 0.5 * input *
           (1.0 + std::tanh(sqrt_two_over_pi * (input + 0.044715 * input * input * input)));
}

int run_case(const char* label, ops::GeluMode mode, std::int32_t rows, std::int32_t columns,
             std::uint32_t seed) {
    const std::size_t count = static_cast<std::size_t>(rows) * columns;
    std::vector<float> input(count);
    fill_uniform(input, seed, -8.0f, 8.0f);
    round_to_bf16(input);

    std::vector<double> expected(count);
    for (std::size_t index = 0; index < count; ++index) {
        expected[index] = gelu_oracle(input[index], mode);
    }
    const auto input_bits = encode_bf16(input);
    GuardedDeviceBuffer device_input(input_bits.size() * sizeof(std::uint16_t));
    device_input.copy_from_host(input_bits.data(), device_input.bytes());

    Tensor input_tensor(device_input.span(), DType::BF16, {rows, columns});
    ops::gelu(input_tensor, mode, ninfer::test::cuda_environment().default_execution);
    cuda_synchronize();

    int failures = verify_pointwise(label, from_device_bf16(device_input.data(), count), expected,
                                    gelu_bf16_criterion());
    failures += device_input.verify_guards("gelu input");
    return failures;
}

int run_edge_case(ops::GeluMode mode, const char* label) {
    std::vector<float> input{-12.0f, -8.0f, -3.0f, -1.0f, -0.0f, 0.0f, 1.0f, 3.0f, 8.0f, 12.0f};
    round_to_bf16(input);
    std::vector<double> expected(input.size());
    for (std::size_t index = 0; index < input.size(); ++index) {
        expected[index] = gelu_oracle(input[index], mode);
    }

    const auto input_bits = encode_bf16(input);
    GuardedDeviceBuffer device_input(input_bits.size() * sizeof(std::uint16_t));
    device_input.copy_from_host(input_bits.data(), device_input.bytes());
    Tensor input_tensor(device_input.span(), DType::BF16,
                        {static_cast<std::int32_t>(input.size())});
    ops::gelu(input_tensor, mode, ninfer::test::cuda_environment().default_execution);
    cuda_synchronize();

    int failures = verify_pointwise(label, from_device_bf16(device_input.data(), input.size()),
                                    expected, gelu_bf16_criterion());
    failures += device_input.verify_guards("gelu edge input");
    return failures;
}

} // namespace

void add_gelu_cases(ninfer::test::OpSuite& suite) {
    suite.add("gelu/tanh/4304/64",
              [] { return run_case("gelu tanh [4304,64]", ops::GeluMode::Tanh, 4304, 64, 401u); },
              {{"execution", "eager"}});
    suite.add("gelu/exact/4608/16",
              [] { return run_case("gelu exact [4608,16]", ops::GeluMode::Exact, 4608, 16, 501u); },
              {{"execution", "eager"}});
    suite.add("gelu/tanh/edge/values",
              [] { return run_edge_case(ops::GeluMode::Tanh, "gelu tanh edge values"); },
              {{"execution", "eager"}});
    suite.add("gelu/exact/edge/values",
              [] { return run_edge_case(ops::GeluMode::Exact, "gelu exact edge values"); },
              {{"execution", "eager"}});
}
