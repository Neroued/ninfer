#include "ops/linear_swiglu/row_major_mma_epilogue.cuh"
#include "core/weight.h"
#include "ops/linear_swiglu/q8/q8_linear_swiglu_kernels.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/q8/q8_mma_launch.cuh"

namespace ninfer::ops::detail {
namespace {

template <class Schedule>
void launch_route(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_q8_a16_mma<Schedule, SwiGluRowMajorMmaRows<Schedule>>(
        q8_linear_operands(x, w),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
        SwiGluRowMajorMmaEpilogue{}, stream);
}

} // namespace

void q8_linear_swiglu_mma_r32_c32_launch(const Tensor& x, const Weight& w, Tensor& out,
                                         cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<32, 32, 64, 32, 16, 2, 4>;
    launch_route<Schedule>(x, w, out, stream);
}

void q8_linear_swiglu_mma_r32_c48_launch(const Tensor& x, const Weight& w, Tensor& out,
                                         cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<32, 48, 64, 32, 16, 2, 4>;
    launch_route<Schedule>(x, w, out, stream);
}

void q8_linear_swiglu_mma_r32_c64_launch(const Tensor& x, const Weight& w, Tensor& out,
                                         cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<32, 64, 64, 32, 16, 2, 3>;
    launch_route<Schedule>(x, w, out, stream);
}

void q8_linear_swiglu_mma_r32_c80_launch(const Tensor& x, const Weight& w, Tensor& out,
                                         cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<32, 80, 64, 32, 16, 2, 3>;
    launch_route<Schedule>(x, w, out, stream);
}

void q8_linear_swiglu_mma_r32_c96_launch(const Tensor& x, const Weight& w, Tensor& out,
                                         cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<32, 96, 64, 32, 16, 2, 2>;
    launch_route<Schedule>(x, w, out, stream);
}

void q8_linear_swiglu_mma_r32_c128_launch(const Tensor& x, const Weight& w, Tensor& out,
                                          cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<32, 128, 64, 32, 16, 2, 2>;
    launch_route<Schedule>(x, w, out, stream);
}

void q8_linear_swiglu_mma_r64_c64_launch(const Tensor& x, const Weight& w, Tensor& out,
                                         cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<64, 64, 64, 64, 16, 2, 2>;
    launch_route<Schedule>(x, w, out, stream);
}

void q8_linear_swiglu_mma_r64_c96_launch(const Tensor& x, const Weight& w, Tensor& out,
                                         cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>;
    launch_route<Schedule>(x, w, out, stream);
}

void q8_linear_swiglu_mma_r64_c128_launch(const Tensor& x, const Weight& w, Tensor& out,
                                          cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>;
    launch_route<Schedule>(x, w, out, stream);
}

void q8_linear_swiglu_mma_r128_c64_launch(const Tensor& x, const Weight& w, Tensor& out,
                                          cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<128, 64, 64, 64, 16, 2, 2>;
    launch_route<Schedule>(x, w, out, stream);
}

void q8_linear_swiglu_mma_r128_c80_launch(const Tensor& x, const Weight& w, Tensor& out,
                                          cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<128, 80, 64, 64, 16, 2, 2>;
    launch_route<Schedule>(x, w, out, stream);
}

void q8_dflash2_linear_swiglu_mma_r32_c64_k128_launch(const Tensor& x, const Weight& w, Tensor& out,
                                                      cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>;
    launch_route<Schedule>(x, w, out, stream);
}

void q8_dflash2_linear_swiglu_mma_r64_c64_k128_launch(const Tensor& x, const Weight& w, Tensor& out,
                                                      cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<64, 64, 128, 32, 16, 1, 2>;
    launch_route<Schedule>(x, w, out, stream);
}

void q8_dflash2_linear_swiglu_mma_r64_c80_k128_launch(const Tensor& x, const Weight& w, Tensor& out,
                                                      cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<64, 80, 128, 64, 8, 1, 2>;
    launch_route<Schedule>(x, w, out, stream);
}

void q8_dflash2_linear_swiglu_mma_r64_c96_k128_launch(const Tensor& x, const Weight& w, Tensor& out,
                                                      cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<64, 96, 128, 64, 8, 1, 2>;
    launch_route<Schedule>(x, w, out, stream);
}

} // namespace ninfer::ops::detail
