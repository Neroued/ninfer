#pragma once

#include <cstdint>
#include <span>
#include <string_view>

namespace ninfer::test::linear_pair {

struct ShapeCase {
    std::int32_t k;
    std::uint32_t seed;
    std::span<const std::int32_t> route_starts;
    std::span<const std::int32_t> route_interiors;
};

bool cuda_available();

int run_q8_a16_shape(std::string_view label, const ShapeCase& shape);

} // namespace ninfer::test::linear_pair
