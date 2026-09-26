#include "ops/linear_add/q8/q8_add_epilogue.cuh"
#include "core/weight.h"
#include "ops/linear_add/q8/q8_linear_add_kernels.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/q8/q8_mma_launch.cuh"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

template <class Schedule>
void launch_variant(const Tensor& x, const Weight& w, Tensor& residual_out, cudaStream_t stream) {
    launch_q8_a16_mma<Schedule>(
        q8_linear_operands(x, w),
        LinearBf16Output{static_cast<__nv_bfloat16*>(residual_out.data), residual_out.ne[0]},
        Q8AddMmaEpilogue{}, stream);
}

} // namespace

void q8_linear_add_mma_r32_c32_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<32, 32, 64, 32, 16, 2, 4>;
    launch_variant<Schedule>(x, w, residual_out, stream);
}

void q8_linear_add_mma_r32_c48_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<32, 48, 64, 32, 16, 2, 4>;
    launch_variant<Schedule>(x, w, residual_out, stream);
}

void q8_linear_add_mma_r32_c64_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<32, 64, 64, 32, 16, 2, 3>;
    launch_variant<Schedule>(x, w, residual_out, stream);
}

void q8_linear_add_mma_r32_c80_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<32, 80, 64, 32, 16, 2, 3>;
    launch_variant<Schedule>(x, w, residual_out, stream);
}

void q8_linear_add_mma_r32_c96_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<32, 96, 64, 32, 16, 2, 2>;
    launch_variant<Schedule>(x, w, residual_out, stream);
}

void q8_linear_add_mma_r32_c112_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                       cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<32, 112, 64, 32, 16, 2, 2>;
    launch_variant<Schedule>(x, w, residual_out, stream);
}

void q8_linear_add_mma_r32_c128_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                       cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<32, 128, 64, 32, 16, 2, 2>;
    launch_variant<Schedule>(x, w, residual_out, stream);
}

void q8_linear_add_mma_r48_c64_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<48, 64, 64, 48, 16, 2, 3>;
    launch_variant<Schedule>(x, w, residual_out, stream);
}

void q8_linear_add_mma_r48_c80_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<48, 80, 64, 48, 16, 2, 3>;
    launch_variant<Schedule>(x, w, residual_out, stream);
}

void q8_linear_add_mma_r48_c96_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<48, 96, 64, 48, 16, 2, 2>;
    launch_variant<Schedule>(x, w, residual_out, stream);
}

void q8_linear_add_mma_r48_c112_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                       cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<48, 112, 64, 48, 16, 2, 2>;
    launch_variant<Schedule>(x, w, residual_out, stream);
}

void q8_linear_add_mma_r48_c128_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                       cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<48, 128, 64, 48, 16, 2, 2>;
    launch_variant<Schedule>(x, w, residual_out, stream);
}

void q8_linear_add_mma_r64_c32_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<64, 32, 64, 64, 16, 2, 3>;
    launch_variant<Schedule>(x, w, residual_out, stream);
}

void q8_linear_add_mma_r64_c48_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<64, 48, 64, 64, 16, 2, 3>;
    launch_variant<Schedule>(x, w, residual_out, stream);
}

void q8_linear_add_mma_r64_c64_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<64, 64, 64, 64, 16, 2, 2>;
    launch_variant<Schedule>(x, w, residual_out, stream);
}

void q8_linear_add_mma_r64_c80_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<64, 80, 64, 64, 16, 2, 2>;
    launch_variant<Schedule>(x, w, residual_out, stream);
}

void q8_linear_add_mma_r64_c96_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>;
    launch_variant<Schedule>(x, w, residual_out, stream);
}

void q8_linear_add_mma_r64_c112_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                       cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<64, 112, 64, 64, 16, 2, 2>;
    launch_variant<Schedule>(x, w, residual_out, stream);
}

void q8_linear_add_mma_r64_c128_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                       cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>;
    launch_variant<Schedule>(x, w, residual_out, stream);
}

void q8_linear_add_mma_r128_c64_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                       cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<128, 64, 64, 64, 16, 2, 2>;
    launch_variant<Schedule>(x, w, residual_out, stream);
}

void q8_linear_add_mma_r128_c80_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                       cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<128, 80, 64, 64, 16, 2, 2>;
    launch_variant<Schedule>(x, w, residual_out, stream);
}

} // namespace ninfer::ops::detail
