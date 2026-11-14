#pragma once

#include "ops/dynamic_grouped_conv/operands.h"

namespace ninfer::ops::detail::dynamic_grouped_conv::prepare {
struct Projection {
    int rows, columns, splits;
};

struct Plan {
    Projection projection;
    int reduction_capacity;
    std::size_t workspace_bytes, inverse_offset;
    bool fused = false;
};

struct NormSource {
    const Tensor& residual;
    const Tensor& weight;
    const Tensor& inverse;
};

void launch(Plan plan, const NormSource& source, const Tensor& base, const Weight& projection,
            float* partials, Tensor& prepared, Tensor& finish, cudaStream_t stream);
void launch_fused(const NormSource& source, const Tensor& normalized, const Tensor& base,
                  const Weight& projection, Tensor& prepared, Tensor& finish, cudaStream_t stream);
} // namespace ninfer::ops::detail::dynamic_grouped_conv::prepare
