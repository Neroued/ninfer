#include "core/weight.h"
#include "core/platform.h"
#include "ops/linear/weights.h"
#include "ops/float_codec.h"
#include "ops/linear/swiglu/runner.h"
#include "ops/linear_common.h"

#include <array>
#include <vector>

namespace {

#if NINFER_TARGET_SPARK
ninfer::test::quantized_weight::PackedWeight clamp_weight(int recipe) {
    using namespace ninfer;
    using namespace ninfer::test;
    auto packed = linear_weights::make_q8_g32_fp16_weight(2048, 4096, 2201U);
    std::fill(packed.payload.begin(), packed.payload.end(), 0);
    const auto set = [&](int row, int k, int code, float scale) {
        packed.payload[std::size_t(row) * 4096 + k] = static_cast<std::uint8_t>(code);
        const auto bits                             = float_codec::f32_to_f16_bits(scale);
        std::memcpy(packed.payload.data() + packed.scale_plane_offset +
                        (std::size_t(row) * 128 + k / 32) * 2,
                    &bits, sizeof(bits));
    };
    for (int row = 0; row < 1024; ++row) {
        if (recipe < 3) {
            set(row, 0, recipe == 0 ? -21 : 40, .5f);
            set(row + 1024, 0, recipe == 2 ? -40 : 40, .5f);
        } else if (recipe == 3) {
            set(row, 0, 64, 1.f / 64);
            set(row, 32, 1, 1.f / 128);
            set(row, 64, 1, 1.f / 256);
            set(row + 1024, 0, 16, 1.f / 16);
        } else {
            set(row, 0, 64, 1.f / 64);
            set(row + 1024, 0, 64, 1.f / 64);
            set(row + 1024, 32, 1, 1.f / 256);
        }
    }
    return packed;
}
#endif

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

#if NINFER_TARGET_SPARK
    constexpr std::array tokens{1,   2,   4,   8,   15,  16,  17,  31,   32,   33,   63,   64,  65,
                                128, 191, 192, 193, 511, 512, 513, 2559, 2560, 2561, 4096, 4097};
    constexpr std::array graphs{1, 17, 65, 513, 2561};
    add_profile(suite,
                {QType::Q8_G32_FP16, 2048, 4096, 1024, 2201U, ops::LinearPolicy::A16Only, 10.f},
                tokens, graphs);
    add_profile(suite, {QType::Q8_G32_FP16, 2048, 4096, 1024, 2203U, ops::LinearPolicy::A16Only},
                std::array{1, 8, 17, 33, 65, 193, 513, 2561}, std::array{1, 2561});
    for (const auto policy : {ops::LinearPolicy::AllowA8, ops::LinearPolicy::AllowA4})
        add_profile(
            suite,
            {QType::Q8_G32_FP16, 2048, 4096, 1024, std::uint32_t(2211 + int(policy)), policy, 10.f},
            std::array{1, 33, 2561}, std::array{33});
    for (int recipe = 0; recipe < 5; ++recipe) {
        const std::vector<int> columns = recipe == 3   ? std::vector<int>{0, 32, 64}
                                         : recipe == 4 ? std::vector<int>{0, 32}
                                                       : std::vector<int>{0};
        add_negative_gate(suite,
                          {QType::Q8_G32_FP16, 2048, 4096, 1024, std::uint32_t(2221 + recipe),
                           ops::LinearPolicy::A16Only, 10.f},
                          std::array{1, 8, 17, 33, 65, 193, 513, 2561}, columns,
                          [recipe] { return clamp_weight(recipe); });
    }
    add_negative_gate(
        suite, {QType::Q8_G32_FP16, 2048, 4096, 1024, 2231U, ops::LinearPolicy::A16Only, 3.f},
        std::array{1, 17, 513}, std::array{0}, [] { return clamp_weight(1); });
#endif

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
