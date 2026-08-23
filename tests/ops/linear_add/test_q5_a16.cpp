#include "ops/linear_add/linear_add_test_common.h"

#include <array>
#include <exception>
#include <iostream>

namespace {

using ninfer::test::linear_add::ShapeCase;
using ninfer::test::linear_add::WeightFormat;

int q5_a16_conformance() {
    // Starts of the registered positive-T regions. run_shape checks b-1/b/b+1 for every start,
    // plus one interior point for every region, through the public Op.
    constexpr std::array<std::int32_t, 6> kK6144RouteStarts{2, 14, 17, 33, 49, 129};
    constexpr std::array<std::int32_t, 6> kK6144RouteInteriors{1, 8, 24, 40, 96, 256};

    int failures = 0;
    failures += ninfer::test::linear_add::run_shape(
        "Q5_A16 LinearAdd", WeightFormat::Q5G64F16S,
        ShapeCase{5120, 6144, 401U, kK6144RouteStarts, kK6144RouteInteriors});
    constexpr std::array<std::int32_t, 5> kK17408RouteStarts{2, 17, 33, 49, 129};
    constexpr std::array<std::int32_t, 6> kK17408RouteInteriors{1, 8, 24, 40, 96, 256};
    failures += ninfer::test::linear_add::run_shape(
        "Q5_A16 LinearAdd", WeightFormat::Q5G64F16S,
        ShapeCase{5120, 17408, 409U, kK17408RouteStarts, kK17408RouteInteriors});
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
