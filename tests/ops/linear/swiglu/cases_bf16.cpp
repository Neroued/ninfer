#include "core/platform.h"
#include <array>
#include "ops/linear/swiglu/runner.h"
#include "ops/linear/weights.h"
#include "ops/float_codec.h"

namespace ninfer::test::linear_swiglu {
namespace {
quantized_weight::PackedWeight clamp_weight(int recipe) {
    auto packed = linear_weights::make_bf16_weight(2048, 4096, 2101U);
    std::fill(packed.payload.begin(), packed.payload.end(), 0);
    for (int row = 0; row < 2048; ++row) {
        const float value =
            row < 1024 ? (recipe == 0 ? -10.5f : 20.f) : (recipe == 2 ? -20.f : 20.f);
        const auto bits = f32_to_bf16(value);
        std::memcpy(packed.payload.data() + std::size_t(row) * 4096 * 2, &bits, 2);
    }
    return packed;
}
} // namespace

void add_bf16_cases(Suite& suite) {
#if NINFER_TARGET_SPARK
    constexpr std::array tokens{1,   2,   6,   8,   16,  17,   32,   64,  128,
                                129, 256, 257, 512, 513, 2048, 4096, 4097};
    add_profile(suite, {QType::BF16, 2048, 4096, 1024, 2101U, ops::LinearPolicy::A16Only, 10.f},
                tokens,
                std::array{1, 4, 8, 16, 17, 32, 64, 128, 129, 256, 512, 513, 1024, 2048, 4096});
    add_profile(suite, {QType::BF16, 2048, 4096, 1024, 2103U, ops::LinearPolicy::A16Only},
                std::array{1, 4, 8, 16, 17, 32, 64, 128, 129, 256, 512, 513, 1024, 2048, 4096},
                std::array{4, 129});
    for (auto policy : {ops::LinearPolicy::AllowA8, ops::LinearPolicy::AllowA4})
        add_profile(
            suite, {QType::BF16, 2048, 4096, 1024, std::uint32_t(2111 + int(policy)), policy, 10.f},
            std::array{1, 16, 17, 128, 129, 512, 513, 4096}, std::array{17, 513});
    for (int recipe = 0; recipe < 3; ++recipe)
        add_negative_gate(
            suite,
            {QType::BF16, 2048, 4096, 1024, std::uint32_t(2121 + recipe),
             ops::LinearPolicy::A16Only, 10.f},
            std::array{1, 4, 8, 16, 17, 32, 64, 128, 129, 256, 512, 513, 1024, 2048, 4096},
            std::array{0}, [recipe] { return clamp_weight(recipe); });
    add_negative_gate(suite,
                      {QType::BF16, 2048, 4096, 1024, 2131U, ops::LinearPolicy::A16Only, 3.f},
                      std::array{1, 17, 513}, std::array{0}, [] { return clamp_weight(1); });
#endif
}
} // namespace ninfer::test::linear_swiglu
