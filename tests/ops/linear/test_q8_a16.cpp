#include "ops/linear/linear_test_common.h"

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
using namespace ninfer::test::linear;

struct Geometry {
    std::int32_t n;
    std::int32_t k;
    std::uint32_t seed;
};

constexpr std::array kGeometries{
    Geometry{1024, 2048, 257U},  Geometry{1024, 5120, 223U},  Geometry{2048, 4096, 251U},
    Geometry{2048, 4608, 271U},  Geometry{2048, 16384, 283U}, Geometry{2560, 6144, 307U},
    Geometry{4608, 4608, 277U},  Geometry{5120, 4608, 281U},  Geometry{5120, 6144, 239U},
    Geometry{5120, 10240, 211U}, Geometry{5120, 17408, 241U}, Geometry{5120, 25600, 293U},
    Geometry{6144, 2560, 347U},  Geometry{6144, 5120, 227U},  Geometry{9216, 2048, 263U},
    Geometry{10240, 2560, 331U}, Geometry{12288, 2048, 269U}, Geometry{12288, 2560, 313U},
    Geometry{14336, 5120, 229U}, Geometry{16384, 2560, 359U}, Geometry{34816, 5120, 233U},
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

int q8_a16_conformance(std::int32_t selected_n = 0, std::int32_t selected_k = 0) {
    if (selected_n != 0 &&
        std::none_of(kGeometries.begin(), kGeometries.end(), [&](const Geometry& shape) {
            return shape.n == selected_n && shape.k == selected_k;
        }))
        throw std::invalid_argument("Q8 test: unknown geometry");
    int failures = 0;
    for (const auto& shape : kGeometries) {
        if (selected_n != 0 && (shape.n != selected_n || shape.k != selected_k)) continue;
        std::vector<Invocation> calls;
        // Cover live-column tails and the transitions from K-split to tiled contractions.
        for (int t : {1,  2,  3,  4,  5,  7,  8,  9,  15,  16,  17,  23,  24,  25,
                      31, 32, 33, 39, 40, 41, 44, 47, 48,  49,  55,  56,  57,  63,
                      64, 65, 79, 80, 81, 95, 96, 97, 127, 128, 129, 256, 1024}) {
            calls.push_back({t});
        }
        if (shape.n == 2048 && shape.k == 4096) {
            for (int t : {895, 896, 897}) calls.push_back({t});
        }
        if (shape.n == 2560 && shape.k == 6144) {
            for (int t : {50, 51, 52, 512, 1025}) calls.push_back({t});
            for (int t : {4, 33, 49, 50, 51, 129, 512, 1024, 1025})
                calls.push_back({t, CallForm::Policy, ninfer::ops::LinearPolicy::A16Only, true});
            for (int t : {40, 512, 1024, 1025}) calls.push_back({t, CallForm::A16Convenience});
        }
        if (shape.n == 12288 && shape.k == 2560) {
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
            for (int t : {94, 112, 113, 512, 513, 1025}) calls.push_back({t});
            for (int t : {17, 24, 25,  32,  33,  40,  41,  64,  65,  80,   81,  95,
                          96, 97, 112, 113, 127, 128, 129, 512, 513, 1024, 1025})
                calls.push_back({t, CallForm::Policy, ninfer::ops::LinearPolicy::A16Only, true});
            for (int t : {24, 32, 40, 64, 80, 96, 112, 127, 128, 512, 1024, 1025})
                calls.push_back({t, CallForm::A16Convenience});
        }
        if (shape.n == 16384 && shape.k == 2560) {
            for (int t : {512, 513, 1025}) calls.push_back({t});
            for (int t : {1,  8,  9,  16, 17, 24, 25,  32,  33,  40,  41,   48,
                          49, 63, 64, 65, 96, 97, 128, 129, 512, 513, 1024, 1025})
                calls.push_back({t, CallForm::Policy, ninfer::ops::LinearPolicy::A16Only, true});
            for (int t : {8, 16, 24, 32, 40, 48, 63, 64, 96, 128, 512, 1024})
                calls.push_back({t, CallForm::A16Convenience});
        }
        if ((shape.n == 2560 && shape.k == 6144) || (shape.n == 12288 && shape.k == 2560) ||
            (shape.n == 10240 && shape.k == 2560) || (shape.n == 6144 && shape.k == 2560) ||
            (shape.n == 16384 && shape.k == 2560)) {
            failures += q8_workspace_domain(shape.n, shape.k);
            failures += verify_workspace_envelopes(ninfer::QType::Q8_G32_FP16, shape.n, shape.k);
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
        failures += run_shape("Q8_A16", ActivationCompute::A16, make_q8_g32_fp16_weight,
                              {shape.n, shape.k, shape.seed, Comparison::Sampled, true, calls});
    }
    const std::array full_calls{Invocation{1}, Invocation{4}, Invocation{8}};
    for (const Geometry shape :
         {Geometry{2560, 6144, 311U}, Geometry{12288, 2560, 317U}, Geometry{10240, 2560, 337U},
          Geometry{6144, 2560, 349U}, Geometry{16384, 2560, 367U}}) {
        if (selected_n != 0 && (shape.n != selected_n || shape.k != selected_k)) continue;
        failures += run_shape("Q8_A16 full", ActivationCompute::A16, make_q8_g32_fp16_weight,
                              {shape.n, shape.k, shape.seed, Comparison::Full, true, full_calls});
    }
    return failures;
}
} // namespace

int main(int argc, char** argv) {
    if (!ninfer::test::linear::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        std::int32_t n = 0, k = 0;
        if (argc == 4 && std::string_view(argv[1]) == "--shape") {
            n = std::stoi(argv[2]);
            k = std::stoi(argv[3]);
            if (n <= 0 || k <= 0) throw std::invalid_argument("Q8 test: positive N/K required");
        } else if (argc != 1) {
            throw std::invalid_argument("usage: ninfer_linear_q8_a16_test [--shape N K]");
        }
        const int failures = q8_a16_conformance(n, k);
        std::cout << (failures == 0 ? "OK" : "FAIL") << " Q8_A16 Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Q8_A16 Linear: " << error.what() << '\n';
        return 1;
    }
}
