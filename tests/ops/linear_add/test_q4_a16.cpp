#include "ops/linear_add/linear_add_test_common.h"

#include <array>
#include <exception>
#include <iostream>

int main() {
    using namespace ninfer::test::linear_add;
    if (!cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        constexpr std::array<std::int32_t, 8> route_starts{2, 5, 9, 17, 25, 33, 97, 193};
        constexpr std::array<std::int32_t, 9> interiors{1, 4, 8, 64, 128, 512, 513, 1024, 1025};
        constexpr std::array<std::int32_t, 6> graph_tokens{1, 4, 33, 97, 193, 512};
        constexpr std::array<std::int32_t, 3> full_tokens{1, 4, 8};
        int failures = run_shape("Q4_A16 LinearAdd", WeightFormat::Q4G64F16S,
                                 {5120, 6144, 429U, route_starts, interiors, graph_tokens});
        failures += run_shape("Q4_A16 LinearAdd full", WeightFormat::Q4G64F16S,
                              {5120, 6144, 431U, {}, full_tokens, {}, true});
        std::cout << (failures == 0 ? "OK" : "FAIL") << " Q4_A16 LinearAdd\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Q4_A16 LinearAdd: " << error.what() << '\n';
        return 1;
    }
}
