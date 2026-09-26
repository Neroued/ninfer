#include "ops/linear/bf16/bf16_template_launch.cuh"
#include "ops/linear_add/bf16/bf16_linear_add_plan.h"

namespace ninfer::ops::detail {
void bf16_linear_add_decode_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                   cudaStream_t stream) {
    using Schedule = Bf16ScheduleInstance<
        Bf16A16GemvSchedule<8, 2, 2, 8, 4, Bf16ActivationAccess::Direct, Bf16WeightCache::Default,
                            Bf16PhaseOrder::RowSwizzled, 1, 2, 1, 1>,
        6144>;
    auto* data = static_cast<__nv_bfloat16*>(residual.data);
    launch_bf16_a16_gemv<Schedule>(bf16_a16_operands(x, weight), LinearBf16Output{data, weight.n},
                                   LinearResidualAddEpilogue{{data, weight.n}}, stream);
}
} // namespace ninfer::ops::detail
