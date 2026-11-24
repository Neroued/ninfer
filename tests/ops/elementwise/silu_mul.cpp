#include "ops/suite.h"
#include "ninfer/ops/silu_mul.h"
#include "cuda_fixture.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr PointwiseCriterion silu_mul_bf16_criterion() {
    return {/*absolute*/ 2.0e-5, /*relative*/ 4.05e-3};
}

std::vector<std::uint16_t> encode_bf16(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) bits[i] = f32_to_bf16(values[i]);
    return bits;
}

std::vector<double> silu_mul_oracle(const std::vector<float>& gate, const std::vector<float>& up) {
    std::vector<double> expected(gate.size());
    for (std::size_t i = 0; i < gate.size(); ++i) {
        const double g = gate[i];
        expected[i]    = (g / (1.0 + std::exp(-g))) * static_cast<double>(up[i]);
    }
    return expected;
}

int run_contiguous_case(const char* label, std::int32_t rows, std::int32_t columns,
                        std::uint32_t seed) {
    const std::size_t count = static_cast<std::size_t>(rows) * columns;
    std::vector<float> gate(count), up(count);
    fill_uniform(gate, seed, -12.0f, 12.0f);
    fill_uniform(up, seed + 1, -8.0f, 8.0f);
    round_to_bf16(gate);
    round_to_bf16(up);

    const auto expected  = silu_mul_oracle(gate, up);
    const auto gate_bits = encode_bf16(gate);
    const auto up_bits   = encode_bf16(up);
    GuardedDeviceBuffer device_gate(count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_up(count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_out(count * sizeof(std::uint16_t));
    device_gate.copy_from_host(gate_bits.data(), device_gate.bytes());
    device_up.copy_from_host(up_bits.data(), device_up.bytes());

    Tensor gate_tensor(device_gate.span(), DType::BF16, {rows, columns});
    Tensor up_tensor(device_up.span(), DType::BF16, {rows, columns});
    Tensor out_tensor(device_out.span(), DType::BF16, {rows, columns});
    ops::silu_mul(gate_tensor, up_tensor, out_tensor,
                  ninfer::test::cuda_environment().default_execution);
    cuda_synchronize();

    int failures = verify_pointwise(label, from_device_bf16(device_out.data(), count), expected,
                                    silu_mul_bf16_criterion());
    failures += verify_exact("silu_mul gate unchanged",
                             from_device<std::uint16_t>(device_gate.data(), count), gate_bits);
    failures += verify_exact("silu_mul up unchanged",
                             from_device<std::uint16_t>(device_up.data(), count), up_bits);
    failures += device_gate.verify_guards("silu_mul gate");
    failures += device_up.verify_guards("silu_mul up");
    failures += device_out.verify_guards("silu_mul out");
    return failures;
}

int run_strided_gate_up_case() {
    constexpr std::int32_t rows    = 17408;
    constexpr std::int32_t columns = 17;
    const std::size_t count        = static_cast<std::size_t>(rows) * columns;
    std::vector<float> gate(count), up(count);
    fill_uniform(gate, 301u, -12.0f, 12.0f);
    fill_uniform(up, 302u, -8.0f, 8.0f);
    round_to_bf16(gate);
    round_to_bf16(up);

    std::vector<float> gate_up(2 * count);
    for (std::int32_t column = 0; column < columns; ++column) {
        const std::size_t source = static_cast<std::size_t>(column) * rows;
        const std::size_t target = static_cast<std::size_t>(column) * 2 * rows;
        std::copy_n(gate.data() + source, rows, gate_up.data() + target);
        std::copy_n(up.data() + source, rows, gate_up.data() + target + rows);
    }

    const auto expected     = silu_mul_oracle(gate, up);
    const auto gate_up_bits = encode_bf16(gate_up);
    GuardedDeviceBuffer device_gate_up(gate_up_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_out(count * sizeof(std::uint16_t));
    device_gate_up.copy_from_host(gate_up_bits.data(), device_gate_up.bytes());

    Tensor gate_up_tensor(device_gate_up.span(), DType::BF16, {2 * rows, columns});
    Tensor gate_tensor = gate_up_tensor.slice(0, 0, rows);
    Tensor up_tensor   = gate_up_tensor.slice(0, rows, rows);
    Tensor out_tensor(device_out.span(), DType::BF16, {rows, columns});
    ops::silu_mul(gate_tensor, up_tensor, out_tensor,
                  ninfer::test::cuda_environment().default_execution);
    cuda_synchronize();

    int failures =
        verify_pointwise("silu_mul strided gate/up", from_device_bf16(device_out.data(), count),
                         expected, silu_mul_bf16_criterion());
    failures += verify_exact("silu_mul strided inputs unchanged",
                             from_device<std::uint16_t>(device_gate_up.data(), gate_up_bits.size()),
                             gate_up_bits);
    failures += device_gate_up.verify_guards("silu_mul strided inputs");
    failures += device_out.verify_guards("silu_mul strided out");
    return failures;
}

int run_edge_case() {
    std::vector<float> gate{-30.0f, -8.0f, -1.0f, 0.0f, 1.0f, 8.0f, 30.0f};
    std::vector<float> up{2.0f, -3.0f, 0.5f, -4.0f, 5.0f, -6.0f, 7.0f};
    round_to_bf16(gate);
    round_to_bf16(up);

    const auto expected  = silu_mul_oracle(gate, up);
    const auto gate_bits = encode_bf16(gate);
    const auto up_bits   = encode_bf16(up);
    GuardedDeviceBuffer device_gate(gate_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_up(up_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_out(expected.size() * sizeof(std::uint16_t));
    device_gate.copy_from_host(gate_bits.data(), device_gate.bytes());
    device_up.copy_from_host(up_bits.data(), device_up.bytes());

    Tensor gate_tensor(device_gate.span(), DType::BF16, {static_cast<int>(gate.size())});
    Tensor up_tensor(device_up.span(), DType::BF16, {static_cast<int>(up.size())});
    Tensor out_tensor(device_out.span(), DType::BF16, {static_cast<int>(expected.size())});
    ops::silu_mul(gate_tensor, up_tensor, out_tensor,
                  ninfer::test::cuda_environment().default_execution);
    cuda_synchronize();

    int failures = verify_pointwise("silu_mul edge values",
                                    from_device_bf16(device_out.data(), expected.size()), expected,
                                    silu_mul_bf16_criterion());
    failures += device_gate.verify_guards("silu_mul edge gate");
    failures += device_up.verify_guards("silu_mul edge up");
    failures += device_out.verify_guards("silu_mul edge out");
    return failures;
}

} // namespace

void add_silu_mul_cases(ninfer::test::OpSuite& suite) {
    suite.add("silu_mul/17408/1",
              [] { return run_contiguous_case("silu_mul [17408,1]", 17408, 1, 101u); },
              {{"execution", "eager"}});
    suite.add("silu_mul/17408/48",
              [] { return run_contiguous_case("silu_mul [17408,48]", 17408, 48, 102u); },
              {{"execution", "eager"}});
    suite.add("silu_mul/strided_gate_up", [] { return run_strided_gate_up_case(); },
              {{"execution", "eager"}});
    suite.add("silu_mul/edge", [] { return run_edge_case(); }, {{"execution", "eager"}});
}
