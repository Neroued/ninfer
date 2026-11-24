#pragma once

#include "ops/dynamic_grouped_conv/operands.h"

namespace ninfer::ops::detail::dynamic_grouped_conv::prepare {
struct Projection {
    int rows, columns, splits;
};

struct Plan {
    Projection projection;
    int reduction_capacity;
    std::size_t workspace_bytes;
};

void launch(Plan plan, const Tensor& base, const Weight& projection, float* partials,
            Tensor& prepared, Tensor& finish, cudaStream_t stream);
} // namespace ninfer::ops::detail::dynamic_grouped_conv::prepare
