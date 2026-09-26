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
    constexpr int rows = 34816, k = 5120, half = rows / 2, max_tokens = 257;
    auto packed = qw::make_patterned_weight(
        QType::NVFP4, rows, k, 1837U, {.weight_scale_divisor = 1.0f, .input_scale_divisor = 6.0f});
    std::fill(packed.payload.begin(), packed.payload.end(), 0);
    // Preserve the scalar metadata, while constructing the two independently scaled groups.
    const float divisor = 1.0f;
    std::memcpy(packed.payload.data() + packed.weight_divisor_offset, &divisor, 4);
    const auto scale_offset = [&](int row, int group) {
        return packed.scale_plane_offset +
               static_cast<std::size_t>((row / 128) * (k / 64) + group / 4) * 512 +
               (row % 32) * 16 + ((row % 128) / 32) * 4 + group % 4;
    };
    for (int row = 0; row < rows; ++row) {
        const std::size_t base = static_cast<std::size_t>(row) * (k / 2);
        if (row < half) {
            packed.payload[base]                 = 0x0f; // -6
            packed.payload[base + 8]             = 0x01; // +0.5 in the next G16 group
            packed.payload[scale_offset(row, 0)] = 0x46; // 3.5
            packed.payload[scale_offset(row, 1)] = 0x18; // 0.0625
        } else {
            packed.payload[base]                 = 0x07; // +6
            packed.payload[scale_offset(row, 0)] = 0x7e; // 448
        }
    }
    std::vector<std::uint16_t> activation(static_cast<std::size_t>(k) * max_tokens, 0);
    for (int token = 0; token < max_tokens; ++token)
        activation[static_cast<std::size_t>(token) * k] =
            activation[static_cast<std::size_t>(token) * k + 16] = test::f32_to_bf16(1.0F);
    test::GuardedDeviceBuffer weights(packed.payload.size()), x(activation.size() * 2);
    test::GuardedDeviceBuffer y(static_cast<std::size_t>(half) * max_tokens * 2);
    weights.copy_from_host(packed.payload.data(), packed.payload.size());
    x.copy_from_host(activation.data(), activation.size() * 2);
    const Weight weight = packed.device_weight(weights.data());
    const double gate =
        qw::logical_weight_fp64(packed, 0, 0) + qw::logical_weight_fp64(packed, 0, 16);
    const double up    = qw::logical_weight_fp64(packed, half, 0);
    const double ideal = gate / (1.0 + std::exp(-gate)) * up;
    const auto bytes   = ops::linear_swiglu_workspace_capacity_bytes(
        weight.qtype, rows, k, ops::LinearPolicy::AllowA4, 1, max_tokens);
    WorkspaceArena workspace(bytes);
    int failures = 0;
    for (int tokens : {1, 64, 129, 256, 257}) {
        Tensor input(x.data(), DType::BF16, {k, tokens});
        Tensor output(y.data(), DType::BF16, {half, tokens});
        ops::linear_swiglu(input, weight, output, ops::LinearPolicy::AllowA4, workspace, nullptr);
        test::cuda_synchronize();
        std::vector<std::uint16_t> bits(static_cast<std::size_t>(half) * tokens);
        y.copy_to_host(bits.data(), bits.size() * 2);
        std::vector<double> got(bits.size()), reference(bits.size(), ideal);
        for (std::size_t i = 0; i < bits.size(); ++i) got[i] = test::bf16_to_f32(bits[i]);
        // This input is exactly representable by A4. The negative gate makes loss of
        // projection precision observable in the complete nonlinear formula.
        failures += test::verify_reduction("NVFP4 SwiGLU negative gate T=" + std::to_string(tokens),
                                           got, reference, {1.0 / 256.0, 0.0, 2.0 / 256.0});
    }
    return failures;
}
} // namespace

int main() {
    using namespace ninfer;
    using namespace ninfer::test::linear_swiglu;

    try {
        constexpr std::array<std::int32_t, 4> kA16Cases{1, 4, 8, 16};
        // Exercise both sides of the native MMA/TMA boundary, including the partial TMA tile.
        constexpr std::array<std::int32_t, 17> kA4Cases{2,   4,   5,   16,  56,  64,  65,  96,  97,
                                                        112, 128, 129, 255, 256, 257, 512, 1024};
        int failures = check_negative_gate();
        failures += run_profile("LinearSwiGLU NVFP4_A16",
                                {QType::NVFP4, 34816, 5120, 17408, 1801U, ActivationCompute::A16},
                                kA16Cases);
        failures += run_profile("LinearSwiGLU NVFP4_A4",
                                {QType::NVFP4, 34816, 5120, 17408, 1803U, ActivationCompute::A4},
                                kA4Cases, std::array<std::int32_t, 4>{65, 97, 128, 129});
        std::cout << (failures == 0 ? "OK" : "FAIL") << " LinearSwiGLU NVFP4 correctness\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "LinearSwiGLU NVFP4 test failed: " << error.what() << '\n';
        return 1;
    }
}
