#include "ops/linear/linear_test_common.h"

#include <array>
#include <exception>
#include <iostream>

namespace {
using namespace ninfer::test::linear;

constexpr Invocation a16(std::int32_t t) { return {t}; }

constexpr Invocation convenience(std::int32_t t) { return {t, CallForm::A16Convenience}; }

constexpr Invocation graph(std::int32_t t) {
    return {t, CallForm::Policy, ninfer::ops::LinearPolicy::A16Only, true};
}

int q6_a16_conformance() {
    int failures = 0;
    constexpr std::array kN248320K5120{
        a16(1),    a16(2),   a16(3),   a16(4),   a16(7),    a16(8),    a16(9),     a16(12),
        a16(15),   a16(16),  a16(17),  a16(24),  a16(31),   a16(32),   a16(33),    a16(40),
        a16(47),   a16(48),  a16(49),  a16(55),  a16(56),   a16(57),   a16(63),    a16(64),
        a16(65),   a16(72),  a16(79),  a16(80),  a16(81),   a16(88),   a16(95),    a16(96),
        a16(97),   a16(104), a16(111), a16(112), a16(113),  a16(128),  a16(129),   a16(512),
        a16(1024), graph(2), graph(3), graph(9), graph(17), graph(48), graph(112),
    };
    failures += run_shape("Q6_A16", ActivationCompute::A16, make_q6_g64_fp16_weight,
                          {248320, 5120, 191U, Comparison::Sampled, false, kN248320K5120});

    constexpr std::array kN248320K2048{
        a16(1),   a16(2),    a16(3),   a16(4),   a16(8),    a16(15),   a16(16),    a16(17),
        a16(24),  a16(31),   a16(32),  a16(33),  a16(39),   a16(40),   a16(41),    a16(47),
        a16(48),  a16(49),   a16(55),  a16(56),  a16(57),   a16(63),   a16(64),    a16(65),
        a16(71),  a16(72),   a16(73),  a16(79),  a16(80),   a16(81),   a16(95),    a16(96),
        a16(97),  a16(104),  a16(111), a16(112), a16(113),  a16(127),  a16(128),   a16(129),
        a16(512), a16(1024), graph(1), graph(2), graph(17), graph(57), graph(128),
    };
    failures += run_shape("Q6_A16", ActivationCompute::A16, make_q6_g64_fp16_weight,
                          {248320, 2048, 193U, Comparison::Sampled, false, kN248320K2048});

    constexpr std::array kVisionFull{
        convenience(4), a16(8),  a16(16),  a16(20),   a16(24),   a16(28),
        a16(80),        a16(84), graph(8), graph(24), graph(28), graph(84),
    };
    failures += run_shape("Q6_A16", ActivationCompute::A16, make_q6_g64_fp16_weight,
                          {1152, 1536, 197U, Comparison::Full, true, kVisionFull});

    constexpr std::array kVisionLarge{
        a16(128),  a16(132),  a16(192),  a16(256),    a16(380),   a16(384),    a16(388),
        a16(512),  a16(704),  a16(708),  a16(1024),   a16(1028),  a16(1148),   a16(1152),
        a16(1156), a16(1280), a16(2048), a16(131072), graph(388), graph(1024), graph(1156),
    };
    failures += run_shape("Q6_A16", ActivationCompute::A16, make_q6_g64_fp16_weight,
                          {1152, 1536, 197U, Comparison::Sampled, false, kVisionLarge});
    return failures;
}
} // namespace

int main() {
    if (!ninfer::test::linear::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        const int failures = q6_a16_conformance();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " Q6_A16 Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Q6_A16 Linear: " << error.what() << '\n';
        return 1;
    }
}
