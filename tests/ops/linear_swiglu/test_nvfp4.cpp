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
        // 511 and 513 straddle the ragged floor: the first still reaches the composition, the
        // second is the narrowest ragged width the fused route admits and leaves one real token
        // in a third M tile. 767 leaves that tile all but full, and 1025 leaves one after four
        // whole tiles. 255/256 straddle the whole-tile floor, which does not move; 257 stays as
        // the width just past it that the composition keeps.
        constexpr std::array<std::int32_t, 21> kA4Cases{2,   4,   5,   16,  56,  64,   65,
                                                        96,  97,  112, 128, 129, 255,  256,
                                                        257, 511, 512, 513, 767, 1024, 1025};
        int failures = 0;
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
