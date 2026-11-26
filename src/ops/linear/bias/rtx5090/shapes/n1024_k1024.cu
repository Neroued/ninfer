#include "ops/linear/bias/rtx5090/routes.cuh"
#include "ops/linear/projection/bf16/rtx5090/shapes/n1024_k1024.cuh"

namespace ninfer::ops::detail::linear_bias {

Launch select_n1024_k1024(int columns) {
    return bf16_n1024_k1024::select(columns, Bf16BiasRoutes{});
}

} // namespace ninfer::ops::detail::linear_bias
