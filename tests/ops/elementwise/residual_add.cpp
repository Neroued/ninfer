#include "ops/suite.h"
#include "ninfer/ops/residual_add.h"
#include "cuda_fixture.h"
#include "ops/op_tester.h"

#include <cstdint>
#include <iostream>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr PointwiseCriterion residual_add_bf16_criterion() {
    return {/*absolute*/ 0.0, /*relative*/ 3.95e-3};
}

std::vector<std::uint16_t> encode_bf16(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) bits[i] = f32_to_bf16(values[i]);
    return bits;
}

std::vector<double> residual_add_oracle(const std::vector<float>& y, const std::vector<float>& x) {
    std::vector<double> expected(y.size());
    for (std::size_t i = 0; i < y.size(); ++i) {
        expected[i] = static_cast<double>(x[i]) + static_cast<double>(y[i]);
    }
    return expected;
}

int run_case(const char* label, std::int32_t rows, std::int32_t columns, std::uint32_t seed) {
    const std::size_t count = static_cast<std::size_t>(rows) * columns;
    std::vector<float> y(count), x(count);
    fill_uniform(y, seed, -8.0f, 8.0f);
    fill_uniform(x, seed + 1, -8.0f, 8.0f);
    round_to_bf16(y);
    round_to_bf16(x);

    const auto expected = residual_add_oracle(y, x);
    const auto y_bits   = encode_bf16(y);
    const auto x_bits   = encode_bf16(x);
    GuardedDeviceBuffer device_y(count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_x(count * sizeof(std::uint16_t));
    device_y.copy_from_host(y_bits.data(), device_y.bytes());
    device_x.copy_from_host(x_bits.data(), device_x.bytes());

    Tensor y_tensor(device_y.span(), DType::BF16, {rows, columns});
    Tensor x_tensor(device_x.span(), DType::BF16, {rows, columns});
    ops::residual_add(y_tensor, x_tensor, ninfer::test::cuda_environment().default_execution);
    cuda_synchronize();

    int failures = verify_pointwise(label, from_device_bf16(device_x.data(), count), expected,
                                    residual_add_bf16_criterion());
    failures += verify_exact("residual_add y unchanged",
                             from_device<std::uint16_t>(device_y.data(), count), y_bits);
    failures += device_y.verify_guards("residual_add y");
    failures += device_x.verify_guards("residual_add x");
    return failures;
}

int run_cancellation_case() {
    std::vector<float> y{8.0f, -8.0f, 1.0f, -1.0f, 0.0f, 3.5f, -3.5f};
    std::vector<float> x{-8.0f, 8.0f, -1.0f, 1.0f, -0.0f, -3.5f, 3.5f};
    round_to_bf16(y);
    round_to_bf16(x);

    const auto expected = residual_add_oracle(y, x);
    const auto y_bits   = encode_bf16(y);
    const auto x_bits   = encode_bf16(x);
    GuardedDeviceBuffer device_y(y_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_x(x_bits.size() * sizeof(std::uint16_t));
    device_y.copy_from_host(y_bits.data(), device_y.bytes());
    device_x.copy_from_host(x_bits.data(), device_x.bytes());

    Tensor y_tensor(device_y.span(), DType::BF16, {static_cast<int>(y.size())});
    Tensor x_tensor(device_x.span(), DType::BF16, {static_cast<int>(x.size())});
    ops::residual_add(y_tensor, x_tensor, ninfer::test::cuda_environment().default_execution);
    cuda_synchronize();

    int failures = verify_pointwise("residual_add cancellation",
                                    from_device_bf16(device_x.data(), expected.size()), expected,
                                    residual_add_bf16_criterion());
    failures += verify_exact("residual_add cancellation y unchanged",
                             from_device<std::uint16_t>(device_y.data(), y_bits.size()), y_bits);
    failures += device_y.verify_guards("residual_add cancellation y");
    failures += device_x.verify_guards("residual_add cancellation x");
    return failures;
}

} // namespace

void add_residual_add_cases(ninfer::test::OpSuite& suite) {
    suite.add("residual_add/5120/1",
              [] { return run_case("residual_add [5120,1]", 5120, 1, 101u); },
              {{"execution", "eager"}});
    suite.add("residual_add/5120/48",
              [] { return run_case("residual_add [5120,48]", 5120, 48, 102u); },
              {{"execution", "eager"}});
    suite.add("residual_add/2048/17",
              [] { return run_case("residual_add [2048,17]", 2048, 17, 201u); },
              {{"execution", "eager"}});
    suite.add("residual_add/2048/48",
              [] { return run_case("residual_add [2048,48]", 2048, 48, 202u); },
              {{"execution", "eager"}});
    suite.add("residual_add/1152/128",
              [] { return run_case("residual_add [1152,128]", 1152, 128, 301u); },
              {{"execution", "eager"}});
    suite.add("residual_add/cancellation", [] { return run_cancellation_case(); },
              {{"execution", "eager"}});
}
