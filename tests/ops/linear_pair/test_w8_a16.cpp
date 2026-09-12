#include "ops/linear_pair/linear_pair_test_common.h"

#include <array>
#include <exception>
#include <iostream>

namespace {

using ninfer::test::linear_pair::ShapeCase;

int w8_a16_conformance() {
    int failures = 0;

    constexpr std::array<std::int32_t, 2> kK5120RouteStarts{86, 961};
    constexpr std::array<std::int32_t, 3> kK5120RouteInteriors{24, 512, 1024};
    failures += ninfer::test::linear_pair::run_w8_a16_shape(
        "W8_A16 LinearPair", ShapeCase{5120, 431U, kK5120RouteStarts, kK5120RouteInteriors});

    constexpr std::array<std::int32_t, 36> kK2048RouteStarts{
        2,    33,   49,   65,   81,   89,   97,   105,  113,  129,  161,  193,
        385,  481,  641,  642,  673,  681,  785,  897,  961,  977,  1281, 1317,
        1345, 1346, 1441, 1467, 1681, 1709, 1921, 1923, 2017, 2019, 2209, 2271,
    };
    constexpr std::array<std::int32_t, 37> kK2048RouteInteriors{
        1,    16,   40,   56,   72,   84,   92,   100,  108,  120,  144,  176,  288,
        432,  560,  641,  656,  676,  736,  840,  928,  968,  1120, 1296, 1332, 1345,
        1392, 1456, 1576, 1696, 1816, 1921, 1968, 2017, 2112, 2240, 4096,
    };
    failures += ninfer::test::linear_pair::run_w8_a16_shape(
        "W8_A16 LinearPair", ShapeCase{2048, 433U, kK2048RouteStarts, kK2048RouteInteriors});
    return failures;
}

} // namespace

int main() {
    if (!ninfer::test::linear_pair::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    try {
        const int failures = w8_a16_conformance();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " W8_A16 LinearPair\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "W8_A16 LinearPair: " << error.what() << '\n';
        return 1;
    }
}
