#include "ops/suite.h"
#include "ninfer/ops/add_bias.h"
#include "cuda_fixture.h"
#include "ops/op_tester.h"

#include <cstdint>
#include <iostream>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr PointwiseCriterion add_bias_bf16_criterion() {
    return {/*absolute*/ 0.0, /*relative*/ 3.95e-3};
}

std::vector<std::uint16_t> encode_bf16(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        bits[index] = f32_to_bf16(values[index]);
    }
    return bits;
}

std::vector<double> add_bias_oracle(const std::vector<float>& bias,
                                    const std::vector<float>& input) {
    std::vector<double> expected(input.size());
    for (std::size_t index = 0; index < input.size(); ++index) {
        expected[index] =
            static_cast<double>(input[index]) + static_cast<double>(bias[index % bias.size()]);
    }
    return expected;
}

int run_case(const char* label, std::int32_t rows, std::int32_t columns, std::uint32_t seed) {
    const std::size_t count = static_cast<std::size_t>(rows) * columns;
    std::vector<float> bias(rows), input(count);
    fill_uniform(bias, seed, -2.0f, 2.0f);
    fill_uniform(input, seed + 1, -8.0f, 8.0f);
    round_to_bf16(bias);
    round_to_bf16(input);

    const auto expected   = add_bias_oracle(bias, input);
    const auto bias_bits  = encode_bf16(bias);
    const auto input_bits = encode_bf16(input);
    GuardedDeviceBuffer device_bias(bias_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_input(input_bits.size() * sizeof(std::uint16_t));
    device_bias.copy_from_host(bias_bits.data(), device_bias.bytes());
    device_input.copy_from_host(input_bits.data(), device_input.bytes());

    Tensor bias_tensor(device_bias.span(), DType::BF16, {rows});
    Tensor input_tensor(device_input.span(), DType::BF16, {rows, columns});
    ops::add_bias(bias_tensor, input_tensor, ninfer::test::cuda_environment().default_execution);
    cuda_synchronize();

    int failures = verify_pointwise(label, from_device_bf16(device_input.data(), count), expected,
                                    add_bias_bf16_criterion());
    failures +=
        verify_exact("add_bias bias unchanged",
                     from_device<std::uint16_t>(device_bias.data(), bias_bits.size()), bias_bits);
    failures += device_bias.verify_guards("add_bias bias");
    failures += device_input.verify_guards("add_bias input");
    return failures;
}

} // namespace

void add_add_bias_cases(ninfer::test::OpSuite& suite) {
    suite.add("add_bias/1152/1", [] { return run_case("add_bias [1152,1]", 1152, 1, 101u); },
              {{"execution", "eager"}});
    suite.add("add_bias/3456/64", [] { return run_case("add_bias [3456,64]", 3456, 64, 201u); },
              {{"execution", "eager"}});
    suite.add("add_bias/4304/257", [] { return run_case("add_bias [4304,257]", 4304, 257, 301u); },
              {{"execution", "eager"}});
}
