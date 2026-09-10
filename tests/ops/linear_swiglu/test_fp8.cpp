#include "ops/linear_swiglu/linear_swiglu_test_common.h"

#include <array>
#include <exception>
#include <iostream>

int main() {
    using namespace ninfer;
    using namespace ninfer::test::linear_swiglu;

    try {
        constexpr std::array<std::int32_t, 5> kA16Cases{1, 2, 4, 16, 128};
        // The last three entries are the point of this list: below them the TMA-staged route
        // declines, so without them the paired-rows instantiation this op is the only user of -
        // two TMA loads per stage, a non-identity row policy - is executed by no test. 1153 is
        // not a whole cp.async token tile, which is a case the route reaches only since the
        // multiple-of-tile condition was removed.
        constexpr std::array<std::int32_t, 14> kA8Cases{1,  2,  3,   8,    16,   48,   64,
                                                        65, 96, 128, 1024, 1153, 4096, 4288};
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
