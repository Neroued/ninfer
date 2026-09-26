#pragma once
#include "ops/linear/q4/q4_gemv_launch.cuh"
#include "ops/linear/q4/q4_simt_launch.cuh"
#include "ops/linear/q4/q4_mma_launch.cuh"
#include "ops/linear/q4/q4_sliced_k_launch.cuh"
#include "ops/linear/q4/q4_instances.cuh"
#include "ops/linear/q4/q4_launch.h"

namespace ninfer::ops::detail {
template <class Schedule>
void launch_q4_a16_gemv_instance(const Tensor& x, const Weight& w, Tensor& out,
                                 cudaStream_t stream) {
    launch_q4_a16_gemv<Schedule>(q4_linear_operands(x, w),
                                 LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
                                 LinearIdentityEpilogue{}, stream);
}

template <class Schedule>
void launch_q4_a16_simt_instance(const Tensor& x, const Weight& w, Tensor& out,
                                 cudaStream_t stream) {
    launch_q4_a16_simt<Schedule>(q4_linear_operands(x, w),
                                 LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
                                 LinearIdentityEpilogue{}, stream);
}

template <class Schedule>
void launch_q4_a16_mma_instance(const Tensor& x, const Weight& w, Tensor& out,
                                cudaStream_t stream) {
    launch_q4_a16_mma<Schedule>(q4_linear_operands(x, w),
                                LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
                                LinearIdentityEpilogue{}, stream);
}

template <class Schedule>
void launch_q4_a16_sliced_instance(const Tensor& x, const Weight& w, Tensor& out,
                                   cudaStream_t stream) {
    launch_q4_a16_sliced_k_mma<Schedule>(
        q4_linear_operands(x, w), LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
        LinearIdentityEpilogue{}, stream);
}
} // namespace ninfer::ops::detail
