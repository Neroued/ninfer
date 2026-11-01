#include "ops/linear/kernels/q8/q8_a16_tma_k32.cuh"
#include "ops/linear/split/q8/dgx_spark/launch.cuh"

namespace ninfer::ops::detail::linear_split {
namespace {
struct Schedule : Q8A16TmaK32Schedule<8, 2560> {};
} // namespace

void launch_q8_n16384_k2560_full(const Q8LinearOperands& operands,
                                 LinearBf16SegmentedOutput<10240, 6144> output,
                                 cudaStream_t stream) {
    launch_q8_a16_tma_k32<Schedule>(operands, output, LinearIdentityEpilogue{}, stream);
}
} // namespace ninfer::ops::detail::linear_split
