#pragma once
#include "ops/linear/q5/q5_gemv_launch.cuh"
#include "ops/linear/q5/q5_simt_launch.cuh"
#include "ops/linear/q5/q5_mma_launch.cuh"
#include "ops/linear/q5/q5_sliced_k_launch.cuh"
#include "ops/linear/q5/q5_instances.cuh"
#include "ops/linear/q5/q5_launch.h"

namespace ninfer::ops::detail {
template <class Schedule>
void launch_q5_a16_gemv_instance(const Tensor& x, const Weight& w, Tensor& out,
                                 cudaStream_t stream) {
    launch_q5_a16_gemv<Schedule>(q5_linear_operands(x, w),
                                 LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
                                 LinearIdentityEpilogue{}, stream);
}

template <class Schedule>
void launch_q5_a16_simt_instance(const Tensor& x, const Weight& w, Tensor& out,
                                 cudaStream_t stream) {
    launch_q5_a16_simt<Schedule>(q5_linear_operands(x, w),
                                 LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
                                 LinearIdentityEpilogue{}, stream);
}

template <class Schedule>
void launch_q5_a16_direct_simt_instance(const Tensor& x, const Weight& w, Tensor& out,
                                        cudaStream_t stream) {
    launch_q5_a16_direct_simt<Schedule>(
        q5_linear_operands(x, w),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
        LinearIdentityEpilogue{}, stream);
}

template <class Schedule>
void launch_q5_a16_mma_instance(const Tensor& x, const Weight& w, Tensor& out,
                                cudaStream_t stream) {
    launch_q5_a16_mma<Schedule>(q5_linear_operands(x, w),
                                LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
                                LinearIdentityEpilogue{}, stream);
}

template <class Schedule>
void launch_q5_a16_sliced_k_mma_instance(const Tensor& x, const Weight& w, Tensor& out,
                                         cudaStream_t stream) {
    launch_q5_a16_sliced_k_mma<Schedule>(
        q5_linear_operands(x, w),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
        LinearIdentityEpilogue{}, stream);
}
} // namespace ninfer::ops::detail
