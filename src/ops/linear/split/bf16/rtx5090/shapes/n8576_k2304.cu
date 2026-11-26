#include "ops/linear/projection/bf16/rtx5090/shapes/n8576_k2304.cuh"
#include "ops/linear/split/bf16/launch.cuh"
#include "ops/linear/split/dispatch.h"

namespace ninfer::ops::detail::linear_split {
namespace {

void launch_split(const Tensor& x, Weights weights, std::span<const Tensor> out, MemoryArena*,
                  cudaStream_t stream) {
    using Output = LinearBf16SegmentedOutput<1024, 1024, 3072, 3072, 384>;
    const Output output{
        {static_cast<__nv_bfloat16*>(out[0].data), static_cast<__nv_bfloat16*>(out[1].data),
         static_cast<__nv_bfloat16*>(out[2].data), static_cast<__nv_bfloat16*>(out[3].data),
         static_cast<__nv_bfloat16*>(out[4].data)}};
    bf16_n8576_k2304::select(
        x.ne[1], Bf16SplitRoutes<Output>{bf16_a16_operands(x, weights.first), output, stream});
}

} // namespace

const SplitShape kBf16SplitN8576K2304{8576, 2304, kSplitRows<1024, 1024, 3072, 3072, 384>,
                                      launch_split};

} // namespace ninfer::ops::detail::linear_split
