#pragma once
#include "ops/linear/swiglu/dispatch.h"
#include "ops/linear/swiglu/row_major_mma_epilogue.cuh"
#include "ops/linear/kernels/q8/q8_mma_launch.cuh"
#include "ops/linear/kernels/q8/q8_sliced_k_launch.cuh"
#include "ops/linear/swiglu/q8/output.cuh"

namespace ninfer::ops::detail {
template <class Schedule>
void launch_q8_swiglu_mma(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_q8_a16_mma<Schedule, SwiGluRowMajorMmaRows<Schedule>>(
        q8_linear_operands(x, w),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
        SwiGluRowMajorMmaEpilogue{}, stream);
}
} // namespace ninfer::ops::detail
