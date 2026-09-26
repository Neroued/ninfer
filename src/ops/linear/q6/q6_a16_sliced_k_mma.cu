#include "ops/linear/q6/q6_launch.h"
#include "ops/linear/q6/q6_sliced_k_launch.cuh"
#include "ops/linear/q6/q6_instances.cuh"

namespace ninfer::ops::detail {

void launch_q6_a16_sliced_r16_t8_w4_s2(const Tensor& x, const Weight& weight, Tensor& out,
                                       cudaStream_t stream) {
    launch_q6_a16_sliced_k_mma<q6_instances::SlicedR16T8W4S2>(
        q6_linear_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n}, LinearIdentityEpilogue{},
        stream);
}

void launch_q6_a16_sliced_r32_t16_w4_s2(const Tensor& x, const Weight& weight, Tensor& out,
                                        cudaStream_t stream) {
    launch_q6_a16_sliced_k_mma<q6_instances::SlicedR32T16W4S2>(
        q6_linear_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n}, LinearIdentityEpilogue{},
        stream);
}

void launch_q6_a16_sliced_r32_t32_w4_s1(const Tensor& x, const Weight& weight, Tensor& out,
                                        cudaStream_t stream) {
    launch_q6_a16_sliced_k_mma<q6_instances::SlicedR32T32W4S1>(
        q6_linear_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n}, LinearIdentityEpilogue{},
        stream);
}

void launch_q6_a16_sliced_r32_t64_w2_s1(const Tensor& x, const Weight& weight, Tensor& out,
                                        cudaStream_t stream) {
    launch_q6_a16_sliced_k_mma<q6_instances::SlicedR32T64W2S1>(
        q6_linear_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n}, LinearIdentityEpilogue{},
        stream);
}

void launch_q6_a16_sliced_r16_t24_w4_s2(const Tensor& x, const Weight& weight, Tensor& out,
                                        cudaStream_t stream) {
    launch_q6_a16_sliced_k_mma<q6_instances::SlicedR16T24W4S2>(
        q6_linear_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n}, LinearIdentityEpilogue{},
        stream);
}

void launch_q6_a16_sliced_r16_t32_w4_s2(const Tensor& x, const Weight& weight, Tensor& out,
                                        cudaStream_t stream) {
    launch_q6_a16_sliced_k_mma<q6_instances::SlicedR16T32W4S2>(
        q6_linear_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n}, LinearIdentityEpilogue{},
        stream);
}

void launch_q6_a16_sliced_r32_t32_w4_s2(const Tensor& x, const Weight& weight, Tensor& out,
                                        cudaStream_t stream) {
    launch_q6_a16_sliced_k_mma<q6_instances::SlicedR32T32W4S2>(
        q6_linear_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n}, LinearIdentityEpilogue{},
        stream);
}

} // namespace ninfer::ops::detail
