#include "ops/linear/projection/runner.h"
#include "core/platform.h"
#include <utility>
#include <vector>
#include <exception>
#include <iostream>

namespace {
using namespace ninfer::test::linear_weights;

using namespace ninfer;
using namespace ninfer::test::linear;
namespace linear_suite = ninfer::test::linear_suite;

void run_nvfp4_a16(Suite& suite) {
    std::vector<Invocation> invocations;
    for (int t = 1; t <= 33; ++t) invocations.push_back({t});
    for (int t : {63, 64, 65, 95, 96, 97, 127, 128, 129, 512, 1024}) invocations.push_back({t});
    for (int t : {3, 7, 11, 15, 19, 23, 27, 31, 33})
        invocations.push_back({t, CallForm::Policy, ops::LinearPolicy::A16Only, true});
    invocations.push_back({1, CallForm::A16Convenience});
    invocations.push_back({33, CallForm::Policy, ops::LinearPolicy::AllowA8});
    add_shape(suite, ninfer::QType::NVFP4, ActivationRecipe::TokenBlocks, make_nvfp4_weight,
              {14336, 5120, 701U, Comparison::Sampled, true, invocations});
    add_shape(suite, ninfer::QType::NVFP4, ActivationRecipe::TokenBlocks, make_nvfp4_weight,
              {16384, 5120, 703U, Comparison::Sampled, true, invocations});
    add_shape(suite, ninfer::QType::NVFP4, ActivationRecipe::TokenBlocks, make_nvfp4_weight,
              {34816, 5120, 704U, Comparison::Sampled, true, invocations});
    add_shape(suite, ninfer::QType::NVFP4, ActivationRecipe::TokenBlocks, make_nvfp4_weight,
              {5120, 6144, 705U, Comparison::Sampled, true, invocations});
    add_shape(suite, ninfer::QType::NVFP4, ActivationRecipe::TokenBlocks, make_nvfp4_weight,
              {5120, 17408, 707U, Comparison::Sampled, true, invocations});
    for (auto [n, k] : {std::pair{14336, 5120}, std::pair{16384, 5120}, std::pair{34816, 5120},
                        std::pair{5120, 6144}, std::pair{5120, 17408}}) {
        add_workspace_cases(suite, QType::NVFP4, n, k);
    }
}

} // namespace

namespace {
using namespace ninfer::test::linear_weights;

using namespace ninfer;
using namespace ninfer::test::linear;
namespace linear_suite = ninfer::test::linear_suite;

void run_nvfp4_a4(Suite& suite) {
    std::vector<Invocation> invocations;
    for (int t :
         {1,   2,   3,   4,   5,   7,   8,   16,  17,  63,  64,   65,   95,   96,   97,   127, 128,
          129, 191, 192, 193, 383, 384, 385, 511, 512, 513, 1023, 1024, 1025, 1280, 1500, 2048}) {
        invocations.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowA4});
    }
    for (int t : {63, 97, 193, 385, 1023, 1024})
        invocations.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowA4, true});
    add_shape(suite, ninfer::QType::NVFP4, ActivationRecipe::Hashed, make_nvfp4_weight,
              {14336, 5120, 719U, Comparison::Sampled, true, invocations});
    add_shape(suite, ninfer::QType::NVFP4, ActivationRecipe::Hashed, make_nvfp4_weight,
              {16384, 5120, 721U, Comparison::Sampled, true, invocations});
    // These calls regenerate A4 codes/scales on graph replay and cover complete and partial tiles.
    const std::vector<Invocation> n34816_full{
        {1, CallForm::Policy, ops::LinearPolicy::AllowA4, true},
        {4, CallForm::Policy, ops::LinearPolicy::AllowA4, true},
        {8, CallForm::Policy, ops::LinearPolicy::AllowA4, true},
        {32, CallForm::Policy, ops::LinearPolicy::AllowA4, true},
        {33, CallForm::Policy, ops::LinearPolicy::AllowA4, true},
        {64, CallForm::Policy, ops::LinearPolicy::AllowA4, true},
        {65, CallForm::Policy, ops::LinearPolicy::AllowA4, true},
        {128, CallForm::Policy, ops::LinearPolicy::AllowA4, true},
    };
    add_shape(suite, ninfer::QType::NVFP4, ActivationRecipe::Hashed, make_nvfp4_weight,
              {34816, 5120, 722U, Comparison::Full, true, n34816_full});
    auto n34816_invocations = invocations;
    for (int t : {31, 32, 33, 34, 255, 256, 257, 511, 512, 513, 767, 768, 769})
        n34816_invocations.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowA4, true});
    add_shape(suite, ninfer::QType::NVFP4, ActivationRecipe::Hashed, make_nvfp4_weight,
              {34816, 5120, 722U, Comparison::Sampled, false, n34816_invocations});
    add_shape(suite, ninfer::QType::NVFP4, ActivationRecipe::Hashed, make_nvfp4_weight,
              {5120, 6144, 723U, Comparison::Sampled, true, invocations});
    add_shape(suite, ninfer::QType::NVFP4, ActivationRecipe::Hashed, make_nvfp4_weight,
              {5120, 17408, 725U, Comparison::Sampled, true, invocations});
}

} // namespace

#if NINFER_TARGET_SPARK
namespace {
void run_nvfp4_spark_shape(Suite& suite, int n, int k, std::uint32_t seed) {
    std::vector<Invocation> a16;
    for (int t : {1,  2,  3,  4,  6,  8,   9,   15,  16,  17,  23,   24,   25,   31,
                  32, 33, 63, 64, 65, 127, 128, 129, 256, 512, 1024, 2048, 4096, 4097})
        a16.push_back({t});
    for (int t : {1, 2, 4, 6, 8, 16, 24, 32, 64, 128, 129, 512, 1024, 4096})
        a16.push_back({t, CallForm::Policy, ops::LinearPolicy::A16Only, true});
    for (int t : {1, 8, 128, 1024}) a16.push_back({t, CallForm::A16Convenience});
    for (int t : {2, 8, 16, 32, 128, 1024})
        a16.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowA8});
    add_shape(suite, QType::NVFP4, ActivationRecipe::Hashed, make_nvfp4_weight,
              {n, k, seed + 1, Comparison::Sampled, true, a16});
    const std::vector<Invocation> full_a16{{1}, {2}, {4}, {8}};
    add_shape(suite, QType::NVFP4, ActivationRecipe::Hashed, make_nvfp4_weight,
              {n, k, seed + 3, Comparison::Full, true, full_a16});
    std::vector<Invocation> a4;
    for (int t = 1; t <= 128; ++t) a4.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowA4});
    for (int t : {129,  191,  192,  193,  255,  256,  257,  383,  384,  385,  511,  512,  513,
                  1023, 1024, 1025, 2047, 2048, 2049, 3839, 3840, 3841, 3968, 4095, 4096, 4097})
        a4.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowA4});
    for (int t : {1, 4, 6, 8, 16, 32, 64, 96, 128, 192, 256, 512, 1024, 2048, 4096})
        a4.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowA4, true});
    add_shape(suite, QType::NVFP4, ActivationRecipe::Hashed, make_nvfp4_weight,
              {n, k, seed + 5, Comparison::Sampled, true, a4});
    const std::vector<Invocation> full_a4{
        {1, CallForm::Policy, ops::LinearPolicy::AllowA4},
        {4, CallForm::Policy, ops::LinearPolicy::AllowA4},
        {6, CallForm::Policy, ops::LinearPolicy::AllowA4},
        {8, CallForm::Policy, ops::LinearPolicy::AllowA4},
        {16, CallForm::Policy, ops::LinearPolicy::AllowA4},
        {17, CallForm::Policy, ops::LinearPolicy::AllowA4},
        {32, CallForm::Policy, ops::LinearPolicy::AllowA4},
        {64, CallForm::Policy, ops::LinearPolicy::AllowA4},
        {128, CallForm::Policy, ops::LinearPolicy::AllowA4, true}};
    add_shape(suite, QType::NVFP4, ActivationRecipe::Hashed, make_nvfp4_weight,
              {n, k, seed + 7, Comparison::Full, true, full_a4});
    const std::vector<Invocation> tail{{16, CallForm::Policy, ops::LinearPolicy::AllowA4},
                                       {17, CallForm::Policy, ops::LinearPolicy::AllowA4},
                                       {32, CallForm::Policy, ops::LinearPolicy::AllowA4},
                                       {128, CallForm::Policy, ops::LinearPolicy::AllowA4, true},
                                       {1024, CallForm::Policy, ops::LinearPolicy::AllowA4}};
    add_shape(suite, QType::NVFP4, ActivationRecipe::Hashed, make_nvfp4_weight,
              {n, k, seed + 9, Comparison::Sampled, true, tail, ActivationPattern::KTail});
    add_workspace_cases(suite, QType::NVFP4, n, k);
}

} // namespace
#endif

void ninfer::test::linear::add_nvfp4_cases(Suite& suite) {
    run_nvfp4_a16(suite);
    run_nvfp4_a4(suite);
#if NINFER_TARGET_SPARK
    run_nvfp4_spark_shape(suite, 12288, 4096, 1200U);
    run_nvfp4_spark_shape(suite, 4096, 6144, 1300U);
#endif
}
