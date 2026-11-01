#include "ops/linear/kernels/q8/q8_a16_tma_k32.cuh"
#include "ops/linear/projection/q8/dgx_spark/launch.h"

namespace ninfer::ops::detail {
namespace {
struct Schedule : Q8A16TmaK32Schedule<8, 3072> {};
} // namespace

void launch_q8_n2560_k3072_bulk_full(const Tensor& x, const Weight& weight, Tensor& out,
                                     cudaStream_t stream) {
    launch_q8_a16_tma_k32<Schedule>(
        q8_linear_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
        LinearIdentityEpilogue{}, stream);
}
} // namespace ninfer::ops::detail
