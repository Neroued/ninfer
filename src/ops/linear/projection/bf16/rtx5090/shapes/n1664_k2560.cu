#include "ops/linear/projection/bf16/bf16_shapes.h"
#include "ops/linear/projection/bf16/shapes/contiguous_routes.cuh"
#include "ops/linear/projection/bf16/rtx5090/shapes/n1664_k2560.cuh"

namespace ninfer::ops::detail {

Bf16Launch select_bf16_n1664_k2560(std::int32_t tokens) {
    return bf16_n1664_k2560::select(tokens, Bf16ContiguousRoutes{});
}

} // namespace ninfer::ops::detail
