#pragma once

#include "ops/linear/kernels/q8/q8_output_launch.cuh"
#include "ops/linear/kernels/q8/q8_a16_tma.cuh"

namespace ninfer::ops::detail::linear_split {

template <class Output>
struct Q8SplitRoutes : Q8A16OutputLaunch<Output> {
    template <class Schedule>
    void tma() const {
        launch_q8_a16_tma_mma<Schedule>(this->operands, this->output, LinearIdentityEpilogue{},
                                        this->stream);
    }
};

void launch_q8_n16384_k2560_full(const Q8LinearOperands& operands,
                                 LinearBf16SegmentedOutput<10240, 6144> output,
                                 cudaStream_t stream);

void launch_q8_n12288_k2560_full(const Q8LinearOperands& operands,
                                 LinearBf16SegmentedOutput<6144, 6144> output, cudaStream_t stream);

} // namespace ninfer::ops::detail::linear_split
