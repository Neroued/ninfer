#include "ops/linear/split/q4_q5/shapes/n7168_n7168_k5120/launch.h"

namespace ninfer::ops::detail::linear_split::q4_q5_n7168_n7168_k5120 {
namespace {
void launch(const Tensor& x, Weights weights, std::span<const Tensor> outputs, MemoryArena*,
            cudaStream_t stream) {
    if (x.ne[1] <= 12)
        launch_small(x, weights.first, *weights.second, outputs, stream);
    else
        launch_matrix(x, weights.first, *weights.second, outputs, stream);
}
} // namespace
} // namespace ninfer::ops::detail::linear_split::q4_q5_n7168_n7168_k5120

namespace ninfer::ops::detail::linear_split {
const SplitShape kQ4Q5SplitN7168N7168K5120{
    7168, 5120, kSplitRows<6144, 1024, 6144, 1024>, q4_q5_n7168_n7168_k5120::launch, nullptr, 7168};
} // namespace ninfer::ops::detail::linear_split
