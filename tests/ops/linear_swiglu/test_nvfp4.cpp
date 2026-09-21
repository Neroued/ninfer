#include "core/weight.h"
#include "ops/linear_swiglu/linear_swiglu_test_common.h"

#include <array>
#include <exception>
#include <iostream>

int main() {
    using namespace ninfer;
    using namespace ninfer::test::linear_swiglu;

    try {
        constexpr std::array<std::int32_t, 4> kA16Cases{1, 4, 8, 16};
        // Test boundary and ragged token sizes across tile multiples (128, 256, 512, 1024, 1408)
        // to verify TMA fused ragged SwiGLU correctness against the FP32/FP64 mathematical oracle.
        constexpr std::array<std::int32_t, 23> kA4Cases{
            2,   4,   5,   16,  56,   64,   65,   96,   97,   112,  128, 129,
            255, 256, 257, 300, 511,  512,  513,  1023, 1024, 1025, 1408};
        int failures = 0;
        failures += run_profile("LinearSwiGLU NVFP4_A16",
                                {QType::NVFP4, 34816, 5120, 17408, 1801U, ActivationCompute::A16},
                                kA16Cases);
        failures += run_profile("LinearSwiGLU NVFP4_A4",
                                {QType::NVFP4, 34816, 5120, 17408, 1803U, ActivationCompute::A4},
                                kA4Cases, std::array<std::int32_t, 7>{65, 97, 128, 129, 257, 513, 1025});
        std::cout << (failures == 0 ? "OK" : "FAIL") << " LinearSwiGLU NVFP4 correctness\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "LinearSwiGLU NVFP4 test failed: " << error.what() << '\n';
        return 1;
    }
}
