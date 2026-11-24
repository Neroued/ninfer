#pragma once

#include "ops/linear/add/q8/epilogue.cuh"
#include "ops/linear/kernels/q8/q8_gemv_launch.cuh"
#include "ops/linear/kernels/q8/q8_simt_launch.cuh"
#include "ops/linear/kernels/q8/q8_mma_launch.cuh"
#include "ops/linear/kernels/q8/q8_sliced_k_launch.cuh"
#include "ops/linear/kernels/q8/q8_grouped_sliced_k_launch.cuh"

namespace ninfer::ops::detail {
template <class Schedule>
void launch_q8_add_simt(const Tensor& x, const Weight& weight, Tensor& residual,
                        cudaStream_t stream) {
    const LinearBf16Output output{static_cast<__nv_bfloat16*>(residual.data), residual.ne[0]};
    launch_q8_a16_simt<Schedule>(q8_linear_operands(x, weight), output,
                                 LinearResidualAddEpilogue{{output.data, output.rows, 0}}, stream);
}

template <class Schedule>
void launch_q8_add_gemv(const Tensor& x, const Weight& weight, Tensor& residual,
                        cudaStream_t stream) {
    const LinearBf16Output output{static_cast<__nv_bfloat16*>(residual.data), residual.ne[0]};
    launch_q8_a16_gemv<Schedule>(q8_linear_operands(x, weight), output,
                                 LinearResidualAddEpilogue{{output.data, output.rows, 0}}, stream);
}

template <class Schedule>
void launch_q8_add_sliced(const Tensor& x, const Weight& weight, Tensor& residual,
                          cudaStream_t stream) {
    const LinearBf16Output output{static_cast<__nv_bfloat16*>(residual.data), residual.ne[0]};
    launch_q8_a16_sliced_k_mma<Schedule>(q8_linear_operands(x, weight), output,
                                         LinearResidualAddEpilogue{{output.data, output.rows, 0}},
                                         stream);
}

template <class Schedule>
void launch_q8_add_grouped(const Tensor& x, const Weight& weight, Tensor& residual,
                           cudaStream_t stream) {
    const LinearBf16Output output{static_cast<__nv_bfloat16*>(residual.data), residual.ne[0]};
    launch_q8_a16_grouped_sliced_k_mma<Schedule>(
        q8_linear_operands(x, weight), output,
        LinearResidualAddEpilogue{{output.data, output.rows, 0}}, stream);
}

template <class Schedule>
void launch_q8_add_mma(const Tensor& x, const Weight& weight, Tensor& residual,
                       cudaStream_t stream) {
    const LinearBf16Output output{static_cast<__nv_bfloat16*>(residual.data), residual.ne[0]};
    launch_q8_a16_mma<Schedule>(q8_linear_operands(x, weight), output, Q8AddMmaEpilogue{}, stream);
}

} // namespace ninfer::ops::detail
