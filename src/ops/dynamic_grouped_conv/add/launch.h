#pragma once

#include "ops/dynamic_grouped_conv/operands.h"

namespace ninfer::ops::detail::dynamic_grouped_conv::add {
enum class Projection { TiledMma, MmaK128 };

struct Plan {
    Projection projection;
    int tile_index;
    std::size_t workspace_bytes;
};

void launch(Plan plan, const Tensor& x, const Weight& projection, const Tensor& base,
            const Tensor& delta, Tensor& residual, Tensor& projected, cudaStream_t stream);
} // namespace ninfer::ops::detail::dynamic_grouped_conv::add
