#include "ops/linear_swiglu/linear_swiglu_test_common.h"

#include <array>
#include <exception>
#include <iostream>

int main() {
    using namespace ninfer;
    using namespace ninfer::test::linear_swiglu;

    try {
        constexpr std::array<std::int32_t, 2> kA16Cases{1, 2};
        constexpr std::array<std::int32_t, 6> kA8Cases{1, 2, 3, 48, 65, 1024};
        int failures = 0;
        failures += run_profile(
            "LinearSwiGLU FP8_A16",
            {QType::FP8_E4M3FN_ROW_BF16S, 34816, 5120, 17408, 1811U, ActivationCompute::A16},
            kA16Cases);
#ifndef NINFER_VOLTA_BUILD
        // A4 (cvt.e2m1x2) and A8 (mma.sync.kind::f8f6f4) are Blackwell / sm_89 hardware;
        // the Volta build reaches a __trap() in both, which aborts the whole binary and
        // takes the A16 results with it. See the V100 performance summary.
        failures += run_profile(
            "LinearSwiGLU FP8_A8",
            {QType::FP8_E4M3FN_ROW_BF16S, 34816, 5120, 17408, 1813U, ActivationCompute::A8},
            kA8Cases);
#else
        std::cout << "SKIP FP8_A8: no Volta route for A8 activation compute\n";
#endif
        std::cout << (failures == 0 ? "OK" : "FAIL") << " LinearSwiGLU FP8 correctness\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "LinearSwiGLU FP8 test failed: " << error.what() << '\n';
        return 1;
    }
}
