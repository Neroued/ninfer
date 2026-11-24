#include "ops/linear/kernels/fp8/fp8_template_launch.cuh"
#include "core/weight.h"
#include "ops/linear_attention/gated_delta_net/input_conv/fp8/launch.h"
#include "core/device.h"
#include "ops/linear/common/output.cuh"
#include "ops/linear/kernels/fp8/fp8_schedule.cuh"
#include "ops/linear/kernels/fp8/fp8_a16_gemv.cuh"
#include <cuda_bf16.h>

namespace ninfer::ops::detail {

void launch_fp8_gdn_input_decode(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                 cudaStream_t stream) {
    using Geometry = Fp8N16384K5120;
    using Schedule = Fp8A16GemvSchedule<4, 4, 16, 4, Fp8CodeCache::Default, 1, 1>;
    const LinearBf16SegmentedOutput<10240, 6144> output{static_cast<__nv_bfloat16*>(qkv.data),
                                                        static_cast<__nv_bfloat16*>(z.data)};
    launch_fp8_a16_gemv<Fp8ScheduleInstance<Schedule, Geometry::kInputRows>>(
        fp8_a16_operands(x, weight), output, LinearIdentityEpilogue{}, stream);
}

} // namespace ninfer::ops::detail
