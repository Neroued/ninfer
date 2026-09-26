#pragma once
#include "ops/linear/bf16/bf16_launch.h"
#include "ops/linear/bf16/bf16_template_launch.cuh"

namespace ninfer::ops::detail {
template <class Schedule>
void launch_bf16_gemv(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_bf16_a16_gemv<Schedule>(bf16_a16_operands(x, w),
                                   LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
                                   LinearIdentityEpilogue{}, stream);
}

template <class Schedule>
void launch_bf16_simt(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_bf16_a16_simt<Schedule>(bf16_a16_operands(x, w),
                                   LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
                                   LinearIdentityEpilogue{}, stream);
}

template <class Schedule>
void launch_bf16_mma(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_bf16_a16_mma<Schedule>(bf16_a16_operands(x, w),
                                  LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
                                  LinearIdentityEpilogue{}, stream);
}

template <class Schedule>
void launch_bf16_sliced_k_mma(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_bf16_a16_sliced_k_mma<Schedule>(
        bf16_a16_operands(x, w), LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
        LinearIdentityEpilogue{}, stream);
}

template <class Schedule>
void launch_bf16_tma_mma(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_bf16_a16_tma_mma<Schedule>(bf16_a16_operands(x, w),
                                      LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
                                      LinearIdentityEpilogue{}, stream);
}
} // namespace ninfer::ops::detail
