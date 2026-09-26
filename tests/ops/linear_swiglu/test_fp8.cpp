#include "core/weight.h"
#include "ops/linear_swiglu/linear_swiglu_test_common.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <array>
#include <exception>
#include <iostream>

namespace {
int check_negative_gate() {
    using namespace ninfer;
    namespace qw = test::quantized_weight;
    if (test::cuda_unavailable()) return 0;
    constexpr int rows = 34816, k = 5120, half = rows / 2, max_tokens = 65;
    auto packed = qw::make_patterned_weight(QType::FP8_E4M3FN_ROW_BF16, rows, k, 1817U);
    std::fill(packed.payload.begin(), packed.payload.end(), 0);
    for (int row = 0; row < rows; ++row) {
        // E4M3 -20 and +32, with independently represented BF16 row multipliers.
        packed.payload[static_cast<std::size_t>(row) * k] = row < half ? 0xda : 0x60;
        const auto scale = test::f32_to_bf16(row < half ? 1.0078125F : 1048576.0F);
        std::memcpy(packed.payload.data() + packed.scale_plane_offset + 2 * row, &scale, 2);
    }
    std::vector<std::uint16_t> activation(static_cast<std::size_t>(k) * max_tokens, 0);
    for (int token = 0; token < max_tokens; ++token)
        activation[static_cast<std::size_t>(token) * k] = test::f32_to_bf16(1.0F);
    test::GuardedDeviceBuffer weights(packed.payload.size()), x(activation.size() * 2);
    test::GuardedDeviceBuffer y(static_cast<std::size_t>(half) * max_tokens * 2);
    weights.copy_from_host(packed.payload.data(), packed.payload.size());
    x.copy_from_host(activation.data(), activation.size() * 2);
    const Weight weight = packed.device_weight(weights.data());
    const double gate   = qw::logical_weight_fp64(packed, 0, 0);
    const double up     = qw::logical_weight_fp64(packed, half, 0);
    const double ideal  = gate / (1.0 + std::exp(-gate)) * up;
    const auto bytes    = ops::linear_swiglu_workspace_capacity_bytes(
        weight.qtype, rows, k, ops::LinearPolicy::AllowA8, 1, max_tokens);
    WorkspaceArena workspace(bytes);
    int failures = 0;
    for (int tokens : {1, 64, 65}) {
        Tensor input(x.data(), DType::BF16, {k, tokens});
        Tensor output(y.data(), DType::BF16, {half, tokens});
        ops::linear_swiglu(input, weight, output, ops::LinearPolicy::AllowA8, workspace, nullptr);
        test::cuda_synchronize();
        std::vector<std::uint16_t> bits(static_cast<std::size_t>(half) * tokens);
        y.copy_to_host(bits.data(), bits.size() * 2);
        std::vector<double> got(bits.size()), reference(bits.size(), ideal);
        for (std::size_t i = 0; i < bits.size(); ++i) got[i] = test::bf16_to_f32(bits[i]);
        // This input is exactly representable by A8. The negative gate makes loss of
        // projection precision observable in the complete nonlinear formula.
        failures += test::verify_reduction("FP8 SwiGLU negative gate T=" + std::to_string(tokens),
                                           got, reference, {1.0 / 256.0, 0.0, 2.0 / 256.0});
    }
    return failures;
}
} // namespace

int main() {
    using namespace ninfer;
    using namespace ninfer::test::linear_swiglu;

    try {
        constexpr std::array kA16Cases{1,  2,  4,  5,  8,  9,  16,  17,  24,  25,
                                       32, 33, 64, 65, 96, 97, 128, 129, 1024};
        constexpr std::array<std::int32_t, 11> kA8Cases{1, 2, 3, 8, 16, 48, 64, 65, 96, 128, 1024};
        int failures = 0;
        failures += check_negative_gate();
        failures += run_profile(
            "LinearSwiGLU FP8_A16",
            {QType::FP8_E4M3FN_ROW_BF16, 34816, 5120, 17408, 1811U, ActivationCompute::A16},
            kA16Cases, std::array<std::int32_t, 1>{16});
        failures += run_profile(
            "LinearSwiGLU FP8_A8",
            {QType::FP8_E4M3FN_ROW_BF16, 34816, 5120, 17408, 1813U, ActivationCompute::A8},
            kA8Cases, std::array<std::int32_t, 3>{2, 65, 128});
        std::cout << (failures == 0 ? "OK" : "FAIL") << " LinearSwiGLU FP8 correctness\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "LinearSwiGLU FP8 test failed: " << error.what() << '\n';
        return 1;
    }
}
