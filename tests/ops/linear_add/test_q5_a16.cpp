#include "ops/linear_add/linear_add_test_common.h"

#include <array>
#include <exception>
#include <iostream>

namespace {

using ninfer::test::linear_add::ShapeCase;
using ninfer::test::linear_add::WeightFormat;

int q5_a16_conformance() {
    // Exercise the production transitions, including the same transitions in
    // a composite tail and the 192-token limit on that tail.
    constexpr std::array<std::int32_t, 13> starts{5,  9,   17,  25,  33,  49, 65,
                                                  97, 129, 161, 193, 257, 513};
    constexpr std::array<std::int32_t, 13> interiors{1,  2,   3,   7,   13,  20,  40,
                                                     80, 112, 144, 224, 768, 1024};
    constexpr std::array<std::int32_t, 9> graphs{13, 40, 80, 112, 192, 513, 545, 609, 705};
    constexpr std::array<std::int32_t, 0> no_starts{};
    constexpr std::array<std::int32_t, 7> full{8, 13, 32, 40, 80, 112, 192};
    int failures = 0;
    for (const auto k : {6144, 17408}) {
        failures += ninfer::test::linear_add::run_shape(
            "Q5_A16 LinearAdd", WeightFormat::Q5G64F16S,
            ShapeCase{5120, k, 401U, starts, interiors, graphs, false, 512});
        failures += ninfer::test::linear_add::run_shape(
            "Q5_A16 LinearAdd", WeightFormat::Q5G64F16S,
            ShapeCase{5120, k, 401U, no_starts, full, {}, true, 0});
    }
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
