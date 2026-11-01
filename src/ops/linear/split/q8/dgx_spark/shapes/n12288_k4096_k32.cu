#include "ops/linear/kernels/q8/q8_a16_tma_k32.cuh"
#include "ops/linear/split/q8/dgx_spark/launch.cuh"

namespace ninfer::ops::detail::linear_split {
void launch_q8_n12288_k4096_full(const Q8LinearOperands& operands,
                                 LinearBf16SegmentedOutput<4096, 4096, 4096> output,
                                 cudaStream_t stream) {
    if (operands.tokens < 4096)
        launch_q8_a16_tma_k32<Q8A16TmaK32Schedule<4, 4096>>(operands, output,
                                                            LinearIdentityEpilogue{}, stream);
    else
        launch_q8_a16_tma_k32<Q8A16TmaK32Schedule<16, 4096>>(operands, output,
                                                             LinearIdentityEpilogue{}, stream);
}
} // namespace ninfer::ops::detail::linear_split
