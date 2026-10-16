#include "core/weight.h"
#include "ops/linear/swiglu/runner.h"
#include "ops/linear_common.h"

#include <array>

void ninfer::test::linear_swiglu::add_q4_cases(Suite& suite) {

    // Public numerical cases straddle each registered Q4 implementation interval. They make
    // no assertion about the private route selected for any T.
    constexpr std::array<std::int32_t, 36> kTokenCases{
        1,   2,   7,   8,   9,   16,  17,  24,  25,  31,  32,  33,  40,  41,  48,  49,  96,  128,
        129, 152, 153, 168, 169, 176, 192, 224, 225, 256, 257, 384, 385, 512, 513, 640, 641, 1024,
    };
    add_profile(suite, {QType::Q4_G64_FP16, 34816, 5120, 17408, 1401U, ops::LinearPolicy::A16Only},
                kTokenCases, std::array<std::int32_t, 7>{7, 25, 49, 128, 152, 168, 1024});
}
