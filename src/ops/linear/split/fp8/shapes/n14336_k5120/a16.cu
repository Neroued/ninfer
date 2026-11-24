#include "ops/linear/split/fp8/shapes/n14336_k5120/launch.h"
#include "ops/linear/kernels/fp8/fp8_template_launch.cuh"
#include "core/weight.h"
#include "core/device.h"
#include "ops/linear/kernels/fp8/fp8_schedule.cuh"
#include "ops/linear/kernels/fp8/fp8_a16_gemv.cuh"
#include <cuda_bf16.h>
#include "ops/linear/kernels/fp8/fp8_instances.cuh"
#include "ops/linear/kernels/fp8/fp8_a16_mma.cuh"

namespace ninfer::ops::detail::linear_split::fp8_n14336_k5120 {
namespace {

void launch_decode(const Tensor& x, const Weight& weight, std::span<const Tensor> outputs,
                   cudaStream_t stream) {
    using Geometry = Fp8N14336K5120;
    // Keep six CTAs resident with the four-segment output (39 registers, no spills).
    using Schedule = Fp8A16GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 6>;
    const LinearBf16SegmentedOutput<6144, 1024, 6144, 1024> output{
        static_cast<__nv_bfloat16*>(outputs[0].data),
        static_cast<__nv_bfloat16*>(outputs[1].data),
        static_cast<__nv_bfloat16*>(outputs[2].data),
        static_cast<__nv_bfloat16*>(outputs[3].data),
    };
    launch_fp8_a16_gemv<Fp8ScheduleInstance<Schedule, Geometry::kInputRows>>(
        fp8_a16_operands(x, weight), output, LinearIdentityEpilogue{}, stream);
}

void launch_small(const Tensor& x, const Weight& weight, std::span<const Tensor> outputs,
                  cudaStream_t stream) {
    const LinearBf16SegmentedOutput<6144, 1024, 6144, 1024> output{
        static_cast<__nv_bfloat16*>(outputs[0].data), static_cast<__nv_bfloat16*>(outputs[1].data),
        static_cast<__nv_bfloat16*>(outputs[2].data), static_cast<__nv_bfloat16*>(outputs[3].data)};
    const auto launch = [&]<class Schedule>() {
        launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Schedule, 5120>>(
            fp8_a16_operands(x, weight), output, LinearIdentityEpilogue{}, stream);
    };
    if (x.ne[1] <= 16) return launch.template operator()<Fp8SlicedInstance<16, 4, 1>>();
    if (x.ne[1] <= 24) return launch.template operator()<Fp8SlicedInstance<32, 4, 2>>();
    launch.template operator()<Fp8SlicedInstance<32, 4, 1>>();
}

template <class S>
void run(const Tensor& x, const Weight& weight, std::span<const Tensor> outputs,
         cudaStream_t stream) {
    const LinearBf16SegmentedOutput<6144, 1024, 6144, 1024> output{
        static_cast<__nv_bfloat16*>(outputs[0].data), static_cast<__nv_bfloat16*>(outputs[1].data),
        static_cast<__nv_bfloat16*>(outputs[2].data), static_cast<__nv_bfloat16*>(outputs[3].data)};
    launch_fp8_a16_mma<Fp8ScheduleInstance<S, 5120>>(fp8_a16_operands(x, weight), output,
                                                     LinearIdentityEpilogue{}, stream);
}

void launch_matrix(const Tensor& x, const Weight& weight, std::span<const Tensor> outputs,
                   cudaStream_t stream) {
    // Column tiles follow the measured whole-Op envelope, including the 129..160 tail.
    if (x.ne[1] <= 64)
        run<Fp8A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>>(x, weight, outputs, stream);
    else if (x.ne[1] <= 80 || (x.ne[1] >= 129 && x.ne[1] <= 160))
        run<Fp8A16MmaSchedule<32, 80, 128, 32, 16, 1, 3>>(x, weight, outputs, stream);
    else if (x.ne[1] <= 96)
        run<Fp8A16MmaSchedule<64, 96, 128, 64, 16, 1, 2>>(x, weight, outputs, stream);
    else
        run<Fp8A16MmaSchedule<64, 128, 64, 32, 16, 2, 2>>(x, weight, outputs, stream);
}

} // namespace

void launch_a16(const Tensor& x, const Weight& weight, std::span<const Tensor> outputs,
                cudaStream_t stream) {
    if (x.ne[1] == 1)
        launch_decode(x, weight, outputs, stream);
    else if (x.ne[1] <= 32)
        launch_small(x, weight, outputs, stream);
    else
        launch_matrix(x, weight, outputs, stream);
}

} // namespace ninfer::ops::detail::linear_split::fp8_n14336_k5120
