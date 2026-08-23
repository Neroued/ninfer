#include "ops/linear/linear_test_common.h"

#include <array>
#include <exception>
#include <iostream>

namespace {

using namespace ninfer::test::linear;

constexpr Invocation a16(std::int32_t t) { return {t}; }

constexpr Invocation convenience(std::int32_t t) { return {t, CallForm::A16Convenience}; }

int w8_a16_conformance() {
    int failures = 0;

    // 9 and 96 are the Volta fused route's band edges, 24 the C8 MTP operating point.
    constexpr std::array kN248320K5120{
        a16(1),  a16(6),  a16(9),  a16(16), a16(17), a16(24), a16(32),
        a16(33), a16(34), a16(48), a16(49), a16(64), a16(65), a16(96),
    };
    failures += run_shape("W8_A16", ActivationCompute::A16, make_w8g32_f16s_weight,
                          {248320, 5120, 197U, Comparison::Sampled, false, kN248320K5120});

    constexpr std::array kN5120K10240{
        a16(1),  a16(4),  a16(5),  a16(8),  a16(9),  a16(16), a16(17), a16(24),
        a16(25), a16(32), a16(33), a16(40), a16(41), a16(48), a16(49), a16(128),
    };
    failures += run_shape("W8_A16", ActivationCompute::A16, make_w8g32_f16s_weight,
                          {5120, 10240, 211U, Comparison::Sampled, false, kN5120K10240});

    constexpr std::array kN1024K5120{
        a16(1), a16(4), a16(5), a16(16), a16(17), a16(128),
    };
    failures += run_shape("W8_A16", ActivationCompute::A16, make_w8g32_f16s_weight,
                          {1024, 5120, 223U, Comparison::Sampled, false, kN1024K5120});

    constexpr std::array kN6144K5120{
        a16(1), a16(4), a16(5), a16(16), a16(17), a16(128),
    };
    failures += run_shape("W8_A16", ActivationCompute::A16, make_w8g32_f16s_weight,
                          {6144, 5120, 227U, Comparison::Sampled, false, kN6144K5120});

    constexpr std::array kN14336K5120{
        a16(1),  a16(4),  a16(5),  a16(8),  a16(9),  a16(16), a16(17), a16(24),
        a16(25), a16(32), a16(33), a16(40), a16(41), a16(48), a16(49), a16(128),
    };
    failures += run_shape("W8_A16", ActivationCompute::A16, make_w8g32_f16s_weight,
                          {14336, 5120, 229U, Comparison::Sampled, false, kN14336K5120});

    constexpr std::array kN34816K5120{
        a16(1),  a16(4),  a16(5),  a16(8),  a16(9),  a16(15), a16(16), a16(17), a16(21),  a16(22),
        a16(24), a16(25), a16(32), a16(33), a16(40), a16(41), a16(48), a16(49), a16(128),
    };
    failures += run_shape("W8_A16", ActivationCompute::A16, make_w8g32_f16s_weight,
                          {34816, 5120, 233U, Comparison::Sampled, false, kN34816K5120});

    constexpr std::array kN5120K6144{
        a16(1),  a16(4),  a16(5),  a16(8),  a16(9),  a16(16), a16(17), a16(24),
        a16(25), a16(32), a16(33), a16(40), a16(41), a16(48), a16(49), a16(128),
    };
    failures += run_shape("W8_A16", ActivationCompute::A16, make_w8g32_f16s_weight,
                          {5120, 6144, 239U, Comparison::Sampled, false, kN5120K6144});

    constexpr std::array kN5120K17408{
        a16(1),  a16(4),  a16(5),  a16(8),  a16(9),  a16(16), a16(17), a16(24), a16(25),
        a16(30), a16(31), a16(32), a16(33), a16(40), a16(41), a16(48), a16(49), a16(128),
    };
    failures += run_shape("W8_A16", ActivationCompute::A16, make_w8g32_f16s_weight,
                          {5120, 17408, 241U, Comparison::Sampled, false, kN5120K17408});

    constexpr std::array kN2048K4096{
        a16(1),  a16(4),  a16(5),  a16(8),   a16(9),   a16(16),   a16(17), a16(24),
        a16(25), a16(26), a16(27), a16(32),  a16(33),  a16(40),   a16(41), a16(48),
        a16(49), a16(56), a16(57), a16(895), a16(896), a16(1024),
    };
    failures += run_shape("W8_A16", ActivationCompute::A16, make_w8g32_f16s_weight,
                          {2048, 4096, 251U, Comparison::Sampled, false, kN2048K4096});

    constexpr std::array kN1024K2048{
        convenience(1), a16(4), a16(5), a16(16), a16(17), a16(128),
    };
    failures += run_shape("W8_A16", ActivationCompute::A16, make_w8g32_f16s_weight,
                          {1024, 2048, 257U, Comparison::Full, true, kN1024K2048});

    constexpr std::array kN9216K2048{
        a16(1), a16(13), a16(14), a16(128), a16(129), a16(256),
    };
    failures += run_shape("W8_A16", ActivationCompute::A16, make_w8g32_f16s_weight,
                          {9216, 2048, 263U, Comparison::Sampled, false, kN9216K2048});

    constexpr std::array kN12288K2048{
        a16(1),
        a16(16),
        a16(17),
        a16(128),
    };
    failures += run_shape("W8_A16", ActivationCompute::A16, make_w8g32_f16s_weight,
                          {12288, 2048, 269U, Comparison::Sampled, false, kN12288K2048});

    constexpr std::array kN2048K4608{
        a16(1),  a16(14), a16(15), a16(16),  a16(17),  a16(19),    a16(20),
        a16(21), a16(23), a16(24), a16(25),  a16(27),  a16(28),    a16(29),
        a16(31), a16(32), a16(33), a16(871), a16(872), a16(32768),
    };
    failures += run_shape("W8_A16", ActivationCompute::A16, make_w8g32_f16s_weight,
                          {2048, 4608, 271U, Comparison::Sampled, false, kN2048K4608});

    constexpr std::array kN4608K4608{
        a16(1), a16(8), a16(9), a16(11), a16(12), a16(13), a16(256), a16(257), a16(32768),
    };
    failures += run_shape("W8_A16", ActivationCompute::A16, make_w8g32_f16s_weight,
                          {4608, 4608, 277U, Comparison::Sampled, false, kN4608K4608});

    constexpr std::array kN5120K4608{
        a16(1), a16(4), a16(5), a16(6), a16(128), a16(32768),
    };
    failures += run_shape("W8_A16", ActivationCompute::A16, make_w8g32_f16s_weight,
                          {5120, 4608, 281U, Comparison::Sampled, false, kN5120K4608});

    constexpr std::array kN2048K16384{
        a16(1),    a16(2),    a16(4),    a16(5),    a16(8),    a16(9),    a16(16),   a16(17),
        a16(24),   a16(25),   a16(32),   a16(33),   a16(36),   a16(37),   a16(40),   a16(41),
        a16(47),   a16(48),   a16(49),   a16(64),   a16(65),   a16(66),   a16(72),   a16(73),
        a16(80),   a16(81),   a16(96),   a16(97),   a16(104),  a16(105),  a16(112),  a16(113),
        a16(120),  a16(121),  a16(125),  a16(126),  a16(128),  a16(129),  a16(144),  a16(145),
        a16(255),  a16(256),  a16(384),  a16(385),  a16(480),  a16(481),  a16(482),  a16(640),
        a16(641),  a16(668),  a16(669),  a16(672),  a16(673),  a16(674),  a16(704),  a16(705),
        a16(784),  a16(785),  a16(896),  a16(897),  a16(912),  a16(913),  a16(960),  a16(961),
        a16(1007), a16(1008), a16(1009), a16(1119), a16(1120), a16(1121), a16(1280), a16(1281),
        a16(1313), a16(1314), a16(1344), a16(1345), a16(1440), a16(1441), a16(1500), a16(1501),
        a16(1680), a16(1681), a16(1745), a16(1746), a16(1791), a16(1792), a16(1793), a16(1919),
        a16(1920), a16(1921), a16(1953), a16(1954), a16(2016), a16(2017), a16(2048), a16(2049),
        a16(2112), a16(2113), a16(4096),
    };
    failures += run_shape("W8_A16", ActivationCompute::A16, make_w8g32_f16s_weight,
                          {2048, 16384, 283U, Comparison::Sampled, false, kN2048K16384});

    return failures;
}

} // namespace

int main() {
    if (!ninfer::test::linear::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    try {
        const int failures = w8_a16_conformance();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " W8_A16 Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "W8_A16 Linear: " << error.what() << '\n';
        return 1;
    }
}
