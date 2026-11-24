#include "ops/linear/split/fp8/shapes/n16384_k5120/launch.h"

namespace ninfer::ops::detail::linear_split::fp8_n16384_k5120 {
namespace {
bool quantized(LinearPolicy policy, int columns) { return allows_a8(policy) && columns >= 17; }

std::size_t capacity(LinearPolicy policy, std::int32_t, std::int32_t last) {
    return quantized(policy, last)
               ? fp8_a8_workspace_capacity_bytes(last, 5120, partial_capacity(last))
               : 0;
}

void launch(const Tensor& x, Weights weights, std::span<const Tensor> outputs,
            MemoryArena* workspace, cudaStream_t stream) {
    if (!quantized(weights.policy, x.ne[1])) {
        launch_a16(x, weights.first, outputs, stream);
        return;
    }
    const auto scratch =
        allocate_fp8_a8_workspace(*workspace, x.ne[1], 5120, partial_capacity(x.ne[1]));
    launch_a8(x, weights.first, outputs, scratch, stream);
}
} // namespace
} // namespace ninfer::ops::detail::linear_split::fp8_n16384_k5120

namespace ninfer::ops::detail::linear_split {
const SplitShape kFp8SplitN16384K5120{16384, 5120, kSplitRows<10240, 6144>,
                                      fp8_n16384_k5120::launch, fp8_n16384_k5120::capacity};
} // namespace ninfer::ops::detail::linear_split
