#include "ops/linear_swiglu/linear_swiglu_test_common.h"

#include <array>
#include <exception>
#include <iostream>

int main() {
    using namespace ninfer;
    using namespace ninfer::test::linear_swiglu;

    try {
        constexpr std::array<std::int32_t, 4> kA16Cases{1, 4, 8, 16};
        constexpr std::array<std::int32_t, 5> kA4Cases{5, 48, 49, 128, 1024};
        int failures = 0;
        failures += run_profile("LinearSwiGLU NVFP4_A16",
                                {QType::NVFP4, 34816, 5120, 17408, 1801U, ActivationCompute::A16},
                                kA16Cases);
#ifndef NINFER_VOLTA_BUILD
        // A4 (cvt.e2m1x2) and A8 (mma.sync.kind::f8f6f4) are Blackwell / sm_89 hardware;
        // the Volta build reaches a __trap() in both, which aborts the whole binary and
        // takes the A16 results with it. See the V100 performance summary.
        failures +=
            run_profile("LinearSwiGLU NVFP4_A4",
                        {QType::NVFP4, 34816, 5120, 17408, 1803U, ActivationCompute::A4}, kA4Cases);
#else
        constexpr std::array<std::int32_t, 3> kPrepackedCases{1, 5, 16};
        failures += run_profile(
            "LinearSwiGLU NVFP4_A16 QPN-prepacked",
            {QType::NVFP4, 34816, 5120, 17408, 1805U, ActivationCompute::A16},
            kPrepackedCases, true);
        std::cout << "SKIP NVFP4_A4: no Volta route for A4 activation compute\n";
#endif
        std::cout << (failures == 0 ? "OK" : "FAIL") << " LinearSwiGLU NVFP4 correctness\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "LinearSwiGLU NVFP4 test failed: " << error.what() << '\n';
        return 1;
    }
}
