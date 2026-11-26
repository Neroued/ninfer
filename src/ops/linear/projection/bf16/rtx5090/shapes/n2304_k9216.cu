#include "ops/linear/projection/bf16/bf16_shapes.h"
#include "ops/linear/projection/bf16/rtx5090/shapes/n2304_k9216.cuh"
#include "ops/linear/projection/bf16/shapes/contiguous_routes.cuh"

namespace ninfer::ops::detail {

Bf16Launch select_bf16_n2304_k9216(std::int32_t tokens) {
    return bf16_n2304_k9216::select(tokens, Bf16ContiguousRoutes{});
}

} // namespace ninfer::ops::detail
