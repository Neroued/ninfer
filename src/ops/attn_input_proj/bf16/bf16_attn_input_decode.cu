#include "ops/linear/bf16/bf16_template_launch.cuh"
#include "ops/attn_input_proj/bf16/bf16_attn_input_plan.h"

namespace ninfer::ops::detail {
void bf16_attn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                   Tensor& k, Tensor& v, cudaStream_t stream) {
    using Schedule = Bf16ScheduleInstance<
        Bf16A16GemvSchedule<4, 1, 8, 8, 4, Bf16ActivationAccess::Direct, Bf16WeightCache::Default,
                            Bf16PhaseOrder::RowSwizzled, 1, 1, 1, 2>,
        5120>;
    const LinearBf16SegmentedOutput<6144, 1024, 6144, 1024> output{
        {static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
         static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(v.data)}};
    launch_bf16_a16_gemv<Schedule>(bf16_a16_operands(x, weight), output, LinearIdentityEpilogue{},
                                   stream);
}
} // namespace ninfer::ops::detail
