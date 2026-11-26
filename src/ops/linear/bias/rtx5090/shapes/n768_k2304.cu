#include "ops/linear/bias/rtx5090/routes.cuh"
#include "ops/linear/projection/bf16/rtx5090/shapes/n768_k2304.cuh"

namespace ninfer::ops::detail::linear_bias {

Launch select_n768_k2304(int columns) { return bf16_n768_k2304::select(columns, Bf16BiasRoutes{}); }

} // namespace ninfer::ops::detail::linear_bias
