#pragma once

#include "core/tensor.h"

#include <cstdint>
#include <span>
#include <string_view>

namespace ninfer::test::linear_add {

enum class WeightFormat : std::uint8_t {
    BF16,
    Q4G64F16S,
    Q5G64F16S,
    Q8G32F16S,
};

// Append new fields at the end. The suites in this directory initialize this aggregate
// positionally, so a field inserted in the middle silently re-targets their arguments.
struct ShapeCase {
    std::int32_t n;
    std::int32_t k;
    std::uint32_t seed;
    std::span<const std::int32_t> route_starts;
    std::span<const std::int32_t> route_interiors;
    std::span<const std::int32_t> graph_tokens{};
    bool full_output = false;
    // Non-zero for a shape whose widest route splits the columns into whole waves plus a narrow
    // tail that is resolved through the same route table: the harness then probes every route
    // start again one whole wave later, which is where the tail lands on it.
    std::int32_t composite_offset = 0;
};

bool cuda_available();

int run_shape(std::string_view label, WeightFormat format, const ShapeCase& shape);

} // namespace ninfer::test::linear_add
