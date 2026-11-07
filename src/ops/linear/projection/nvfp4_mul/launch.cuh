#pragma once

#include "ops/linear/kernels/nvfp4/nvfp4_template_launch.cuh"

namespace ninfer::ops::detail {

template <class Schedule>
void nvfp4_mul_linear_gemv(const Tensor& x, const Weight& weight, Tensor& out,
                           cudaStream_t stream) {
    launch_nvfp4_a16_gemv<Schedule>(
        nvfp4_mul_a16_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n}, LinearIdentityEpilogue{},
        stream);
}

template <class Schedule>
void nvfp4_mul_linear_simt(const Tensor& x, const Weight& weight, Tensor& out,
                           cudaStream_t stream) {
    launch_nvfp4_a16_simt<Schedule>(
        nvfp4_mul_a16_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n}, LinearIdentityEpilogue{},
        stream);
}

template <class Schedule>
void nvfp4_mul_linear_sliced(const Tensor& x, const Weight& weight, Tensor& out,
                             cudaStream_t stream) {
    launch_nvfp4_a16_sliced_k_mma<Schedule>(
        nvfp4_mul_a16_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n}, LinearIdentityEpilogue{},
        stream);
}

template <class Schedule>
void nvfp4_mul_linear_mma(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_nvfp4_a16_mma<Schedule>(
        nvfp4_mul_a16_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n}, LinearIdentityEpilogue{},
        stream);
}

} // namespace ninfer::ops::detail
