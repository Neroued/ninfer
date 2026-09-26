#include "ops/linear/q6/q6_launch.h"
#include "ops/linear/q6/q6_simt_launch.cuh"
#include "ops/linear/q6/q6_instances.cuh"

namespace ninfer::ops::detail {

void launch_q6_a16_simt_r8_t4(const Tensor& x, const Weight& weight, Tensor& out,
                              cudaStream_t stream) {
    launch_q6_a16_simt<q6_instances::SimtR8T4>(
        q6_linear_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n}, LinearIdentityEpilogue{},
        stream);
}

void launch_q6_a16_gemv_r4_w2_g16(const Tensor& x, const Weight& weight, Tensor& out,
                                  cudaStream_t stream) {
    launch_q6_a16_gemv<q6_instances::GemvR4W2G16>(
        q6_linear_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n}, LinearIdentityEpilogue{},
        stream);
}

} // namespace ninfer::ops::detail
