#include "ops/linear_add/linear_add_test_common.h"

#include <array>
#include <exception>
#include <iostream>

namespace {

using ninfer::test::linear_add::ShapeCase;
using ninfer::test::linear_add::WeightFormat;

int q5_a16_conformance() {
    // Cover the k-split capacity tiles, the wide-route crossover, and the narrow-tail composite.
    constexpr std::array<std::int32_t, 15> kInteriors{1, 2, 3, 8, 24, 40, 56, 64,
                                                      96, 128, 129, 256, 640, 768, 1024};
    constexpr std::array<std::int32_t, 7> kK6144RouteStarts{9, 17, 25, 33, 61, 193, 513};
    constexpr std::array<std::int32_t, 6> kK6144GraphTokens{513, 526, 545, 561, 705, 1025};

    int failures = 0;
    failures += ninfer::test::linear_add::run_shape(
        "Q5_A16 LinearAdd", WeightFormat::Q5G64F16S,
        ShapeCase{5120, 6144, 401U, kK6144RouteStarts, kInteriors, kK6144GraphTokens, false, 512});
    constexpr std::array<std::int32_t, 7> kK17408RouteStarts{9, 17, 25, 33, 65, 193, 513};
    constexpr std::array<std::int32_t, 6> kK17408GraphTokens{513, 529, 545, 561, 705, 1025};
    failures += ninfer::test::linear_add::run_shape(
        "Q5_A16 LinearAdd", WeightFormat::Q5G64F16S,
        ShapeCase{5120, 17408, 409U, kK17408RouteStarts, kInteriors, kK17408GraphTokens, false, 512});
    return failures;
}

} // namespace

int main() {
    if (!ninfer::test::linear_add::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    try {
        const int failures = q5_a16_conformance();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " Q5_A16 LinearAdd\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Q5_A16 LinearAdd: " << error.what() << '\n';
        return 1;
    }
}
