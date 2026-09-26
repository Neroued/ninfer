#include "ops/linear/q8/q8_simt_launch.cuh"
#include "ops/linear/q8/q8_instances.cuh"
#include "ops/linear/q8/q8_launch.h"

namespace ninfer::ops::detail {
void launch_q8_a16_simt_r8_t4(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_q8_a16_simt<q8_instances::SimtR8T4>(
        q8_linear_operands(x, w),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
        LinearIdentityEpilogue{}, stream);
}

void launch_q8_a16_simt_r8_t8(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_q8_a16_simt<q8_instances::SimtR8T8>(
        q8_linear_operands(x, w),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
        LinearIdentityEpilogue{}, stream);
}

void launch_q8_a16_simt_r4_t4_w2_g16_s2(const Tensor& x, const Weight& w, Tensor& out,
                                        cudaStream_t stream) {
    launch_q8_a16_simt<q8_instances::SimtR4T4W2G16S2>(
        q8_linear_operands(x, w),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
        LinearIdentityEpilogue{}, stream);
}
} // namespace ninfer::ops::detail
