#pragma once

#include "ops/linear/kernels/bf16/bf16_template_launch.cuh"
#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"

namespace ninfer::ops::detail::linear_bias {

template <class Schedule>
void gemv(const Tensor& x, const Weight& w, const Tensor& bias, Tensor& out, cudaStream_t stream) {
    launch_bf16_a16_gemv<Schedule>(
        bf16_a16_operands(x, w), LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
        LinearBiasEpilogue{static_cast<const __nv_bfloat16*>(bias.data)}, stream);
}

template <class Schedule>
void sliced(const Tensor& x, const Weight& w, const Tensor& bias, Tensor& out,
            cudaStream_t stream) {
    launch_bf16_a16_sliced_k_mma<Schedule>(
        bf16_a16_operands(x, w), LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
        LinearBiasEpilogue{static_cast<const __nv_bfloat16*>(bias.data)}, stream);
}

template <class Schedule>
void mma(const Tensor& x, const Weight& w, const Tensor& bias, Tensor& out, cudaStream_t stream) {
    launch_bf16_a16_mma<Schedule>(
        bf16_a16_operands(x, w), LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
        LinearBiasEpilogue{static_cast<const __nv_bfloat16*>(bias.data)}, stream);
}

template <class Schedule>
void tma(const Tensor& x, const Weight& w, const Tensor& bias, Tensor& out, cudaStream_t stream) {
    launch_bf16_a16_tma_mma<Schedule>(
        bf16_a16_operands(x, w), LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
        LinearBiasEpilogue{static_cast<const __nv_bfloat16*>(bias.data)}, stream);
}

template <class Schedule>
void warp(const Tensor& x, const Weight& w, const Tensor& bias, Tensor& out, cudaStream_t stream) {
    launch_bf16_a16_tma_mma<Schedule>(
        bf16_a16_operands(x, w), LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
        Bf16BiasWarpStoreEpilogue{{}, static_cast<const __nv_bfloat16*>(bias.data)}, stream);
}

} // namespace ninfer::ops::detail::linear_bias
