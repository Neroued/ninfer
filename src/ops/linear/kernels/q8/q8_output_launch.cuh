#pragma once

#include "ops/linear/kernels/q8/q8_gemv_launch.cuh"
#include "ops/linear/kernels/q8/q8_sliced_k_launch.cuh"
#include "ops/linear/kernels/q8/q8_grouped_sliced_k_launch.cuh"
#include "ops/linear/kernels/q8/q8_mma_launch.cuh"

namespace ninfer::ops::detail {

template <class Output>
struct Q8A16OutputLaunch {
    Q8LinearOperands operands;
    Output output;
    cudaStream_t stream;

    template <class S>
    void gemv() const {
        launch_q8_a16_gemv<S>(operands, output, LinearIdentityEpilogue{}, stream);
    }

    template <class S>
    void sliced() const {
        launch_q8_a16_sliced_k_mma<S>(operands, output, LinearIdentityEpilogue{}, stream);
    }

    template <class S>
    void grouped() const {
        launch_q8_a16_grouped_sliced_k_mma<S>(operands, output, LinearIdentityEpilogue{}, stream);
    }

    template <class S>
    void mma() const {
        launch_q8_a16_mma<S>(operands, output, LinearIdentityEpilogue{}, stream);
    }
};

} // namespace ninfer::ops::detail
