#include "ops/linear/projection/runner.h"
#include "core/platform.h"
#include <array>

namespace ninfer::test::linear {
namespace {
template <QType Format>
quantized_weight::PackedWeight native_weight(int n, int k, std::uint32_t seed) {
    if constexpr (Format == QType::NVFP4_MUL)
        return quantized_weight::make_patterned_weight(
            Format, n, k, seed, {.weight_scale = 0.031713F, .input_scale = 0.01931F});
    else
        return quantized_weight::make_patterned_weight(Format, n, k, seed);
}
} // namespace

void add_native_cases(Suite& suite) {
#if NINFER_TARGET_SPARK
    const std::array<Invocation, 6> mx_calls{
        {{1, CallForm::A16Convenience},
         {5, CallForm::Policy, ops::LinearPolicy::AllowA8},
         {17, CallForm::Policy, ops::LinearPolicy::A16Only},
         {129, CallForm::Policy, ops::LinearPolicy::AllowA8},
         {513, CallForm::Policy, ops::LinearPolicy::AllowA8},
         {33, CallForm::Policy, ops::LinearPolicy::AllowA8, true}}};
    const std::array<std::array<int, 2>, 14> mx_shapes{{{12288, 4096},
                                                        {4096, 4096},
                                                        {4096, 6144},
                                                        {77440, 4096},
                                                        {2048, 4096},
                                                        {4096, 1024},
                                                        {2208, 4096},
                                                        {2176, 4096},
                                                        {640, 4096},
                                                        {20480, 1536},
                                                        {16384, 1536},
                                                        {4096, 16384},
                                                        {3072, 1024},
                                                        {1024, 1024}}};
    for (auto [n, k] : mx_shapes)
        add_shape(suite, QType::MXFP8, ActivationRecipe::Hashed, native_weight<QType::MXFP8>,
                  {n, k, 1451U, Comparison::Sampled, true, mx_calls});
    const std::array<Invocation, 5> a16_calls{
        {{1, CallForm::A16Convenience},
         {5},
         {129},
         {513},
         {33, CallForm::Policy, ops::LinearPolicy::A16Only, true}}};
    for (auto [n, k] : std::array<std::array<int, 2>, 5>{
             {{2048, 4096}, {512, 4096}, {16384, 1536}, {4096, 16384}, {4096, 1024}}})
        add_shape(suite, QType::FP8_E4M3FN_BLOCK128X128_FP32, ActivationRecipe::Hashed,
                  native_weight<QType::FP8_E4M3FN_BLOCK128X128_FP32>,
                  {n, k, 1457U, Comparison::Sampled, true, a16_calls});
    for (auto [n, k] : std::array<std::array<int, 2>, 5>{
             {{8192, 1024}, {1024, 4096}, {4096, 4096}, {20480, 4096}, {4096, 10240}}})
        add_shape(suite, QType::NVFP4_MUL, ActivationRecipe::Hashed,
                  native_weight<QType::NVFP4_MUL>,
                  {n, k, 1459U, Comparison::Sampled, true, a16_calls});
#endif
}
} // namespace ninfer::test::linear
