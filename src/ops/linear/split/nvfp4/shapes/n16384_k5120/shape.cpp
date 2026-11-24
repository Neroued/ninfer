#include "ops/linear/split/nvfp4/shapes/n16384_k5120/launch.h"

namespace ninfer::ops::detail::linear_split::nvfp4_n16384_k5120 {
namespace {
bool quantized(LinearPolicy policy, int) { return allows_a4(policy); }

std::size_t capacity(LinearPolicy policy, std::int32_t, std::int32_t last) {
    return quantized(policy, last) ? nvfp4_a4_workspace_capacity_bytes(last, 5120) : 0;
}

void launch(const Tensor& x, Weights weights, std::span<const Tensor> outputs,
            MemoryArena* workspace, cudaStream_t stream) {
    if (!quantized(weights.policy, x.ne[1])) {
        launch_a16(x, weights.first, outputs, stream);
        return;
    }
    const auto scratch = allocate_nvfp4_a4_workspace(*workspace, x.ne[1], 5120);
    launch_a4(x, weights.first, outputs, scratch, stream);
}
} // namespace
} // namespace ninfer::ops::detail::linear_split::nvfp4_n16384_k5120

namespace ninfer::ops::detail::linear_split {
const SplitShape kNvfp4SplitN16384K5120{16384, 5120, kSplitRows<10240, 6144>,
                                        nvfp4_n16384_k5120::launch, nvfp4_n16384_k5120::capacity};
} // namespace ninfer::ops::detail::linear_split
