#include "ops/linear/projection/bf16/bf16_shapes.h"
#include "ops/linear/projection/bf16/rtx5090/shapes/n1024_k1024.cuh"
#include "ops/linear/projection/bf16/shapes/contiguous_routes.cuh"

namespace ninfer::ops::detail {

Bf16Launch select_bf16_n1024_k1024(std::int32_t tokens) {
    return bf16_n1024_k1024::select(tokens, Bf16ContiguousRoutes{});
}

} // namespace ninfer::ops::detail
