#include "ops/gdn_input_proj/fp8/fp8_gdn_input_plan.h"
#include "ops/gdn_input_proj/fp8/fp8_gdn_input_output.cuh"
#include "ops/linear/fp8/fp8_template_launch.cuh"
#include "ops/linear/fp8/fp8_instances.cuh"

namespace ninfer::ops::detail {
void fp8_gdn_input_a8_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                             Fp8A8Workspace workspace, cudaStream_t stream) {
    launch_fp8_a8_quantize(x, weight, workspace, stream);
    const Fp8GdnInputOutput output{static_cast<__nv_bfloat16*>(qkv.data),
                                   static_cast<__nv_bfloat16*>(z.data)};
    launch_fp8_a8_mma<Fp8ScheduleInstance<Fp8A8T64R128K128, 5120>>(
        fp8_a8_operands(weight, workspace, x.ne[1]), output, LinearIdentityEpilogue{}, stream);
}
} // namespace ninfer::ops::detail
