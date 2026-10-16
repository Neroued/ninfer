#include "core/weight.h"
#include "ops/linear/swiglu/runner.h"
#include "ops/linear_common.h"

#include <array>
#include <vector>

namespace {

void add_wide_cases(ninfer::test::linear_suite::Suite& suite) {
    using namespace ninfer;
    using namespace ninfer::test::linear_swiglu;
    std::vector<std::int32_t> tokens;
    for (int t = 1; t <= 128; ++t) tokens.push_back(t);
    for (int t : {129, 256, 1024}) tokens.push_back(t);
    constexpr std::array graphs{1,  16, 32, 40, 41, 51, 52, 63,  64,
                                65, 80, 81, 88, 89, 96, 97, 128, 129};
    add_profile(suite, {QType::Q8_G32_FP16, 34816, 5120, 17408, 1603U, ops::LinearPolicy::A16Only},
                tokens, graphs);
}

} // namespace

void ninfer::test::linear_swiglu::add_q8_cases(Suite& suite) {

    // One public numerical case begins each materially distinct Q8 implementation interval;
    // selected endpoints exercise exact-T and predicated tails without inspecting selectors.
    constexpr std::array<std::int32_t, 25> kTokenCases{
        1,   2,   6,   32,  33,  40,  41,  48,  49,  65,  81,  97,  129,
        193, 241, 256, 257, 265, 289, 321, 385, 449, 513, 560, 561,
    };
    add_profile(suite, {QType::Q8_G32_FP16, 12288, 2048, 6144, 1601U, ops::LinearPolicy::A16Only},
                kTokenCases);

    add_wide_cases(suite);
}
