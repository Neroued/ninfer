#include "ops/linear/projection/runner.h"
#include "core/platform.h"

#include <algorithm>
#include <array>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace ninfer::test::linear_weights;
using namespace ninfer::test::linear;
namespace linear_suite = ninfer::test::linear_suite;

struct Geometry {
    std::int32_t n;
    std::int32_t k;
    std::uint32_t seed;
};

const std::array kGeometries{
#if NINFER_TARGET_SPARK
    Geometry{4096, 16384, 1009U}, Geometry{4096, 1024, 983U},  Geometry{4096, 4096, 947U},
    Geometry{12288, 4096, 911U},  Geometry{2560, 3072, 383U},  Geometry{8192, 2560, 373U},
#endif
    Geometry{1024, 2048, 257U},   Geometry{1024, 5120, 223U},  Geometry{2048, 4096, 251U},
    Geometry{2048, 4608, 271U},   Geometry{2048, 16384, 283U}, Geometry{2560, 6144, 307U},
    Geometry{4608, 4608, 277U},   Geometry{5120, 4608, 281U},  Geometry{5120, 6144, 239U},
    Geometry{5120, 10240, 211U},  Geometry{5120, 17408, 241U}, Geometry{5120, 25600, 293U},
    Geometry{6144, 2560, 347U},   Geometry{6144, 5120, 227U},  Geometry{9216, 2048, 263U},
    Geometry{10240, 2560, 331U},  Geometry{12288, 2048, 269U}, Geometry{12288, 2560, 313U},
    Geometry{14336, 5120, 229U},  Geometry{16384, 2560, 359U}, Geometry{34816, 5120, 233U},
    Geometry{248320, 5120, 197U}};

int q8_workspace_domain(std::int32_t n, std::int32_t k) {
    int failures = 0;
    for (auto policy : {ninfer::ops::LinearPolicy::A16Only, ninfer::ops::LinearPolicy::AllowA8,
                        ninfer::ops::LinearPolicy::AllowA4}) {
        if (ninfer::ops::linear_workspace_capacity_bytes(
                ninfer::QType::Q8_G32_FP16, n, k, policy, 1,
                std::numeric_limits<std::int32_t>::max()) != 0) {
            std::cerr << "Q8 [" << n << ',' << k
                      << "]: expected zero workspace across the positive T domain\n";
            ++failures;
        }
        for (auto [first, last] : {std::pair{0, 1}, std::pair{-1, 8}, std::pair{8, 7}}) {
            try {
                (void)ninfer::ops::linear_workspace_capacity_bytes(ninfer::QType::Q8_G32_FP16, n, k,
                                                                   policy, first, last);
                std::cerr << "Q8 [" << n << ',' << k << "]: accepted invalid workspace interval\n";
                ++failures;
            } catch (const std::invalid_argument&) {}
        }
    }
    return failures;
}

void q8_a16_conformance(Suite& suite) {
    for (const auto& shape : kGeometries) {
        std::vector<Invocation> calls;
        // Cover live-column tails and the transitions from K-split to tiled contractions.
        for (int t : {1,  2,  3,  4,  5,  7,  8,  9,  15,  16,  17,  23,  24,  25,
                      31, 32, 33, 39, 40, 41, 44, 47, 48,  49,  55,  56,  57,  63,
                      64, 65, 79, 80, 81, 95, 96, 97, 127, 128, 129, 256, 1024}) {
            calls.push_back({t});
        }
        if (shape.n == 2048 && shape.k == 4096) {
            for (int t : {895, 896, 897}) calls.push_back({t});
            for (int t : {191, 192, 193, 575, 576, 577, 2047, 2048, 2049, 4096, 4097})
                calls.push_back({t});
            for (int t : {49, 65, 129, 577, 2049})
                calls.push_back({t, CallForm::Policy, ninfer::ops::LinearPolicy::A16Only, true});
            add_workspace_cases(suite, ninfer::QType::Q8_G32_FP16, shape.n, shape.k);
        }
        if (shape.n == 4096 && (shape.k == 1024 || shape.k == 16384)) {
            for (int t : {191, 192, 193, 287, 288, 289, 511, 512, 513, 1023, 1025, 4096, 4097})
                calls.push_back({t});
            if (shape.k == 1024)
                for (int t : {575, 576, 577}) calls.push_back({t});
            for (int t : {4, 17, 49, 129, 577, 1025})
                calls.push_back({t, CallForm::Policy, ninfer::ops::LinearPolicy::A16Only, true});
            add_workspace_cases(suite, ninfer::QType::Q8_G32_FP16, shape.n, shape.k);
        }
        if (shape.n == 4096 && shape.k == 4096) {
            for (int t : {255, 257}) calls.push_back({t});
            calls.push_back({256, CallForm::Policy, ninfer::ops::LinearPolicy::A16Only, true});
        }
        if ((shape.n == 12288 || shape.n == 4096) && shape.k == 4096) {
            for (int t :
                 {6, 191, 192, 193, 511, 512, 513, 1023, 1025, 2047, 2048, 2049, 4095, 4096, 4097})
                calls.push_back({t});
            for (int t : {1,   4,   6,   8,   9,   24,   25,   40,   41,   96,  97,
                          128, 129, 192, 193, 512, 1023, 1024, 1025, 2048, 4096})
                calls.push_back({t, CallForm::Policy, ninfer::ops::LinearPolicy::A16Only, true});
        }
        if (shape.n == 8192 && shape.k == 2560) {
            for (int t : {6, 191, 192, 193, 511, 512, 513, 1023, 1025, 2048, 4096, 4097})
                calls.push_back({t});
            for (int t : {1, 6, 16, 17, 32, 33, 96, 97, 128, 129, 192, 193, 512, 1024, 4096})
                calls.push_back({t, CallForm::Policy, ninfer::ops::LinearPolicy::A16Only, true});
        }
        if (shape.n == 2560 && shape.k == 3072) {
            for (int t : {6, 191, 192, 193, 511, 512, 513, 575, 576, 577, 1023, 1025, 2047, 2048,
                          2049, 4096, 4097})
                calls.push_back({t});
            for (int t :
                 {1, 6, 16, 17, 32, 33, 64, 65, 96, 97, 128, 129, 192, 193, 512, 1024, 4096})
                calls.push_back({t, CallForm::Policy, ninfer::ops::LinearPolicy::A16Only, true});
        }
        if (shape.n == 2560 && shape.k == 6144) {
            for (int t : {50, 51, 52, 511, 512, 513, 1025, 2047, 2048, 2049, 4096})
                calls.push_back({t});
            for (int t : {4, 33, 49, 50, 51, 129, 512, 1024, 1025, 2049})
                calls.push_back({t, CallForm::Policy, ninfer::ops::LinearPolicy::A16Only, true});
            for (int t : {40, 512, 1024, 1025}) calls.push_back({t, CallForm::A16Convenience});
        }
        if (shape.n == 12288 && shape.k == 2560) {
            for (int t : {191, 192, 193, 1023, 1151, 1152, 1153, 2048, 4096}) calls.push_back({t});
            for (int t : {192, 1152})
                calls.push_back({t, CallForm::Policy, ninfer::ops::LinearPolicy::A16Only, true});
            for (int t : {34, 35, 36, 72, 104, 105, 110, 111, 112, 113, 512, 1025})
                calls.push_back({t});
            for (int t : {17, 32,  34,  35,  40,  41,  64,  79,  80,   81,  96,
                          97, 104, 110, 111, 112, 113, 129, 512, 1024, 1025})
                calls.push_back({t, CallForm::Policy, ninfer::ops::LinearPolicy::A16Only, true});
            for (int t : {34, 80, 112, 512, 1024, 1025})
                calls.push_back({t, CallForm::A16Convenience});
        }
        if (shape.n == 10240 && shape.k == 2560) {
            for (int t : {66, 67, 68, 512, 1025}) calls.push_back({t});
            for (int t : {17, 24, 25, 32, 33,  40,  41,  48,  49,   64,  65,
                          67, 68, 96, 97, 127, 128, 129, 512, 1024, 1025})
                calls.push_back({t, CallForm::Policy, ninfer::ops::LinearPolicy::A16Only, true});
            for (int t : {24, 32, 40, 48, 67, 96, 127, 128, 512, 1024, 1025})
                calls.push_back({t, CallForm::A16Convenience});
        }
        if (shape.n == 6144 && shape.k == 2560) {
            for (int t : {6, 94, 112, 113, 192, 193, 512, 513, 1023, 1025, 2048, 4096, 4097})
                calls.push_back({t});
            calls.push_back({6, CallForm::Policy, ninfer::ops::LinearPolicy::A16Only, true});
            for (int t : {17, 24, 25,  32,  33,  40,  41,  64,  65,  80,   81,  95,
                          96, 97, 112, 113, 127, 128, 129, 512, 513, 1024, 1025})
                calls.push_back({t, CallForm::Policy, ninfer::ops::LinearPolicy::A16Only, true});
            for (int t : {24, 32, 40, 64, 80, 96, 112, 127, 128, 512, 1024, 1025})
                calls.push_back({t, CallForm::A16Convenience});
        }
        if (shape.n == 16384 && shape.k == 2560) {
            for (int t : {191, 192, 193, 1023, 1151, 1152, 1153, 2048, 4096}) calls.push_back({t});
            for (int t : {192, 1152})
                calls.push_back({t, CallForm::Policy, ninfer::ops::LinearPolicy::A16Only, true});
            for (int t : {512, 513, 1025}) calls.push_back({t});
            for (int t : {1,  8,  9,  16, 17, 24, 25,  32,  33,  40,  41,   48,
                          49, 63, 64, 65, 96, 97, 128, 129, 512, 513, 1024, 1025})
                calls.push_back({t, CallForm::Policy, ninfer::ops::LinearPolicy::A16Only, true});
            for (int t : {8, 16, 24, 32, 40, 48, 63, 64, 96, 128, 512, 1024})
                calls.push_back({t, CallForm::A16Convenience});
        }
        if (((shape.n == 12288 || shape.n == 4096) && shape.k == 4096) ||
            (shape.n == 2560 && shape.k == 3072) || (shape.n == 8192 && shape.k == 2560) ||
            (shape.n == 2560 && shape.k == 6144) || (shape.n == 12288 && shape.k == 2560) ||
            (shape.n == 10240 && shape.k == 2560) || (shape.n == 6144 && shape.k == 2560) ||
            (shape.n == 16384 && shape.k == 2560)) {
            auto test     = linear_suite::make_case("q8", shape.n, shape.k, 0,
                                                    ninfer::ops::LinearPolicy::A16Only,
                                                    linear_suite::Execution::Eager, "workspace-domain");
            test.policies = 7;
            suite.add(std::move(test), [shape] { return q8_workspace_domain(shape.n, shape.k); });
            add_workspace_cases(suite, ninfer::QType::Q8_G32_FP16, shape.n, shape.k);
        }
        if (shape.n == 6144 && shape.k == 5120) {
            for (int t : {191, 192, 193}) calls.push_back({t});
        }
        if (shape.k == 4608) {
            for (int t : {6, 11, 12, 13, 14, 19, 20, 21, 27, 28, 29}) calls.push_back({t});
            if (shape.n == 2048) {
                for (int t : {870, 871, 872}) calls.push_back({t});
            }
            if (shape.n == 4608) {
                for (int t : {255, 257}) calls.push_back({t});
            }
        }
        if (shape.n == 9216 && shape.k == 2048) {
            for (int t : {12, 13, 14}) calls.push_back({t});
        }
        if (shape.n == 248320) calls.push_back({34});
        if (shape.n == 2048 && shape.k == 16384) {
            for (int t : {383,  384,  385,  479,  480,  481,  639,  640,  641,  703,
                          704,  705,  959,  960,  961,  1343, 1344, 1345, 1679, 1680,
                          1681, 2015, 2016, 2017, 2111, 2112, 2113, 4096}) {
                calls.push_back({t});
            }
        }
        for (int t : {1, 8, 16, 32, 48, 64, 65, 128}) {
            calls.push_back({t, CallForm::Policy, ninfer::ops::LinearPolicy::A16Only, true});
        }
        for (int t : {1, 16, 64, 128}) calls.push_back({t, CallForm::A16Convenience});
        calls.push_back({17, CallForm::Policy, ninfer::ops::LinearPolicy::AllowA8});
        calls.push_back({64, CallForm::Policy, ninfer::ops::LinearPolicy::AllowA4});
        add_shape(suite, ninfer::QType::Q8_G32_FP16, ActivationRecipe::TokenBlocks,
                  make_q8_g32_fp16_weight,
                  {shape.n, shape.k, shape.seed, Comparison::Sampled, true, calls});
    }
#if NINFER_TARGET_SPARK
    const std::array stress_calls{Invocation{1}, Invocation{6}, Invocation{41}, Invocation{128},
                                  Invocation{1024}};
    for (const Geometry geometry : {Geometry{12288, 4096, 929U}, Geometry{4096, 4096, 967U}}) {
        for (const auto pattern : {ActivationPattern::KTail, ActivationPattern::Cancellation}) {
            ShapeCase shape{geometry.n,          geometry.k, geometry.seed,
                            Comparison::Sampled, true,       stress_calls};
            shape.activation_pattern = pattern;
            add_shape(suite, ninfer::QType::Q8_G32_FP16, ActivationRecipe::Hashed,
                      make_q8_g32_fp16_weight, shape);
        }
    }
#endif
    const std::array full_calls{Invocation{1}, Invocation{4}, Invocation{8}};
    for (const Geometry shape : {Geometry{2048, 4096, 977U}, Geometry{2560, 6144, 311U},
                                 Geometry{12288, 2560, 317U}, Geometry{10240, 2560, 337U},
#if NINFER_TARGET_SPARK
                                 Geometry{2560, 3072, 389U}, Geometry{8192, 2560, 379U},
                                 Geometry{12288, 4096, 919U}, Geometry{4096, 4096, 953U},
#endif
                                 Geometry{6144, 2560, 349U}, Geometry{16384, 2560, 367U}}) {
        add_shape(suite, ninfer::QType::Q8_G32_FP16, ActivationRecipe::TokenBlocks,
                  make_q8_g32_fp16_weight,
                  {shape.n, shape.k, shape.seed, Comparison::Full, true, full_calls});
    }

    {
        const std::array calls{Invocation{1}, Invocation{17}, Invocation{49}, Invocation{65},
                               Invocation{2049}};
        for (auto pattern : {ActivationPattern::KTail, ActivationPattern::Cancellation}) {
            add_shape(suite, ninfer::QType::Q8_G32_FP16, ActivationRecipe::Hashed,
                      make_q8_g32_fp16_weight,
                      {2048, 4096, 1017U, Comparison::Sampled, true, calls, pattern});
        }
    }
#if NINFER_TARGET_SPARK
    for (const Geometry shape : {Geometry{4096, 1024, 1031U}, Geometry{4096, 16384, 1043U}}) {
        const std::array calls{Invocation{1},  Invocation{16},  Invocation{17},  Invocation{49},
                               Invocation{65}, Invocation{577}, Invocation{1025}};
        for (auto pattern : {ActivationPattern::KTail, ActivationPattern::Cancellation}) {
            add_shape(suite, ninfer::QType::Q8_G32_FP16, ActivationRecipe::Hashed,
                      make_q8_g32_fp16_weight,
                      {shape.n, shape.k, shape.seed, Comparison::Sampled, true, calls, pattern});
        }
        add_shape(suite, ninfer::QType::Q8_G32_FP16, ActivationRecipe::TokenBlocks,
                  make_q8_g32_fp16_weight,
                  {shape.n, shape.k, shape.seed + 1, Comparison::Full, true, full_calls});
    }
    {
        const std::array calls{Invocation{41}, Invocation{49}, Invocation{63}};
        add_shape(suite, ninfer::QType::Q8_G32_FP16, ActivationRecipe::Hashed,
                  make_q8_g32_fp16_weight, {4096, 1024, 1021U, Comparison::Full, true, calls});
    }
#endif
}
} // namespace

void ninfer::test::linear::add_q8_cases(Suite& suite) { q8_a16_conformance(suite); }
