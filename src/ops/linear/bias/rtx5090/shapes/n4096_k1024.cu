#include "ops/linear/bias/rtx5090/routes.cuh"
#include "ops/linear/projection/bf16/rtx5090/shapes/n4096_k1024.cuh"

namespace ninfer::ops::detail::linear_bias {

Launch select_n4096_k1024(int columns) {
    return bf16_n4096_k1024::select(columns, Bf16BiasRoutes{});
}

} // namespace ninfer::ops::detail::linear_bias
