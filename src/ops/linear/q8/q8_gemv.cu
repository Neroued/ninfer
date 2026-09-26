#include "ops/linear/q8/q8_launch.h"
#include "ops/linear/q8/q8_gemv_launch.cuh"

namespace ninfer::ops::detail {
void launch_q8_a16_gemv_r4_w1_k16384(const Tensor& x, const Weight& w, Tensor& out,
                                     cudaStream_t stream) {
    launch_q8_a16_gemv<Q8A16GemvSchedule<4, 1, 2, 16384>>(
        q8_linear_operands(x, w),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
        LinearIdentityEpilogue{}, stream);
}
} // namespace ninfer::ops::detail
