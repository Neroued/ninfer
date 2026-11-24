#include "ops/linear_attention/gated_delta_net/input_conv/q8/dgx_spark/launch.cuh"
#include "ops/linear/kernels/q8/q8_a16_tma_k32.cuh"

namespace ninfer::ops::detail {
namespace {
struct Schedule : Q8A16TmaK32Schedule<8, 2560> {};
} // namespace

void launch_q8_gdn_n16384_k2560_full(const Q8LinearOperands& operands,
                                     LinearBf16SegmentedOutput<10240, 6144> output,
                                     cudaStream_t stream) {
    launch_q8_a16_tma_k32<Schedule>(operands, output, LinearIdentityEpilogue{}, stream);
}
} // namespace ninfer::ops::detail
