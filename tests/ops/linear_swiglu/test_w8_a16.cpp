#include "ops/linear_swiglu/linear_swiglu_test_common.h"

#include <array>
#include <exception>
#include <iostream>
#include <vector>

int dflash2_conformance() {
    using namespace ninfer;
    using namespace ninfer::test::linear_swiglu;
    std::vector<std::int32_t> tokens;
    for (int t = 1; t <= 128; ++t) tokens.push_back(t);
    for (int t : {129, 256, 1024}) tokens.push_back(t);
    constexpr std::array graphs{1,  16, 32, 40, 41, 51, 52, 63,  64,
                                65, 80, 81, 88, 89, 96, 97, 128, 129};
    return run_profile("LinearSwiGLU W8_A16 DFlash2",
                       {QType::W8G32_F16S, 34816, 5120, 17408, 1603U, ActivationCompute::A16},
                       tokens, graphs);
}

int main() {
    using namespace ninfer;
    using namespace ninfer::test::linear_swiglu;

    try {
        // One public numerical case begins each materially distinct W8 implementation interval;
        // selected endpoints exercise exact-T and predicated tails without inspecting selectors.
        constexpr std::array<std::int32_t, 25> kTokenCases{
            1,   2,   6,   32,  33,  40,  41,  48,  49,  65,  81,  97,  129,
            193, 241, 256, 257, 265, 289, 321, 385, 449, 513, 560, 561,
        };
        int failures = run_profile(
            "LinearSwiGLU W8_A16",
            {QType::W8G32_F16S, 12288, 2048, 6144, 1601U, ActivationCompute::A16}, kTokenCases);

        failures += dflash2_conformance();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " LinearSwiGLU W8_A16 correctness\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "LinearSwiGLU W8_A16 test failed: " << error.what() << '\n';
        return 1;
    }
}
