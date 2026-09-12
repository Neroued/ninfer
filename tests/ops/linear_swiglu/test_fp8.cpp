#include "ops/linear_swiglu/linear_swiglu_test_common.h"

#include <array>
#include <exception>
#include <iostream>

int main() {
    using namespace ninfer;
    using namespace ninfer::test::linear_swiglu;

    try {
        constexpr std::array<std::int32_t, 5> kA16Cases{1, 2, 4, 16, 128};
        constexpr std::array<std::int32_t, 11> kA8Cases{1, 2, 3, 8, 16, 48, 64, 65, 96, 128, 1024};
        int failures = 0;
        failures += run_profile(
            "LinearSwiGLU FP8_A16",
            {QType::FP8_E4M3FN_ROW_BF16S, 34816, 5120, 17408, 1811U, ActivationCompute::A16},
            kA16Cases, std::array<std::int32_t, 1>{16});
        failures += run_profile(
            "LinearSwiGLU FP8_A8",
            {QType::FP8_E4M3FN_ROW_BF16S, 34816, 5120, 17408, 1813U, ActivationCompute::A8},
            kA8Cases, std::array<std::int32_t, 3>{2, 65, 128});
        std::cout << (failures == 0 ? "OK" : "FAIL") << " LinearSwiGLU FP8 correctness\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "LinearSwiGLU FP8 test failed: " << error.what() << '\n';
        return 1;
    }
}
