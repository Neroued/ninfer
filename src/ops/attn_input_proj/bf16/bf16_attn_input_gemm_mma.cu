#include "ops/linear/bf16/bf16_instances.cuh"
#include "ops/linear/bf16/bf16_template_launch.cuh"
#include "ops/attn_input_proj/bf16/bf16_attn_input_plan.h"

namespace ninfer::ops::detail {
void bf16_attn_input_mma_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                Tensor& k, Tensor& v, cudaStream_t stream) {
    const LinearBf16SegmentedOutput<6144, 1024, 6144, 1024> output{
        {static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
         static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(v.data)}};

    const auto p   = bf16_a16_operands(x, weight);
    const auto mma = [&]<class S>() {
        launch_bf16_a16_mma<Bf16ScheduleInstance<S, 5120>>(p, output, LinearIdentityEpilogue{},
                                                           stream);
    };
    const auto tma = [&]<class S>() {
        launch_bf16_a16_tma_mma<Bf16ScheduleInstance<S, 5120>>(p, output, LinearIdentityEpilogue{},
                                                               stream);
    };
    if (p.tokens <= 16)
        launch_bf16_a16_sliced_k_mma<Bf16ScheduleInstance<Bf16A16SlicedR32T16W4, 5120>>(
            p, output, LinearIdentityEpilogue{}, stream);
    else if (p.tokens <= 32)
        mma.template operator()<Bf16A16MmaR32T32K128S3>();
    else if (p.tokens <= 64)
        tma.template operator()<Bf16A16TmaR64T64K64S3>();
    else if (p.tokens <= 96)
        mma.template operator()<Bf16A16MmaR64T32K64S3>();
    else if (p.tokens <= 128)
        tma.template operator()<Bf16A16TmaR64T128K64S2>();
    else if (p.tokens <= 192)
        tma.template operator()<Bf16A16TmaR64T64K64S3>();
    else
        tma.template operator()<Bf16A16TmaR64T128K64S2>();
}
} // namespace ninfer::ops::detail
