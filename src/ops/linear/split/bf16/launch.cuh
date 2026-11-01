#pragma once

#include "ops/linear/kernels/bf16/bf16_template_launch.cuh"
#include "ops/linear/kernels/bf16/bf16_a16_tma_mma.cuh"

namespace ninfer::ops::detail::linear_split {

template <class Output>
struct Bf16SplitRoutes {
    Bf16A16Operands operands;
    Output output;
    cudaStream_t stream;

    template <class S>
    void gemv() const {
        launch_bf16_a16_gemv<S>(operands, output, LinearIdentityEpilogue{}, stream);
    }

    template <class S>
    void sliced() const {
        launch_bf16_a16_sliced_k_mma<S>(operands, output, LinearIdentityEpilogue{}, stream);
    }

    template <class S>
    void mma() const {
        launch_bf16_a16_mma<S>(operands, output, LinearIdentityEpilogue{}, stream);
    }

    template <class S, class Epilogue = LinearIdentityEpilogue>
    void tma() const {
        launch_bf16_a16_tma_mma<S>(operands, output, Epilogue{}, stream);
    }
};

} // namespace ninfer::ops::detail::linear_split
