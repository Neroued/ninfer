#include "ops/linear/linear_test_common.h"

#include <array>
#include <exception>
#include <iostream>
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
    Geometry{2048, 4608, 271U},  Geometry{2048, 16384, 283U}, Geometry{4608, 4608, 277U},
    Geometry{5120, 4608, 281U},  Geometry{5120, 6144, 239U},  Geometry{5120, 10240, 211U},
    Geometry{5120, 17408, 241U}, Geometry{5120, 25600, 293U}, Geometry{6144, 5120, 227U},
    Geometry{9216, 2048, 263U},  Geometry{12288, 2048, 269U}, Geometry{14336, 5120, 229U},
    Geometry{34816, 5120, 233U}, Geometry{248320, 5120, 197U}};

int q8_a16_conformance() {
    int failures = 0;
    for (const auto& shape : kGeometries) {
        std::vector<Invocation> calls;
        // Cover live-column tails and the transitions from K-split to tiled contractions.
        for (int t : {1,  2,  3,  4,  5,  7,  8,  9,  15,  16,  17,  23,  24,  25,
                      31, 32, 33, 39, 40, 41, 44, 47, 48,  49,  55,  56,  57,  63,
                      64, 65, 79, 80, 81, 95, 96, 97, 127, 128, 129, 256, 1024}) {
            calls.push_back({t});
        }
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
    return failures;
}
} // namespace

int main() {
    if (!ninfer::test::linear::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        const int failures = q8_a16_conformance();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " Q8_A16 Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Q8_A16 Linear: " << error.what() << '\n';
        return 1;
    }
}
