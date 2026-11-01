#include "gemv.h"
#include "ops/gated_residual/mix/kernel/output.cuh"
#include "ops/gated_residual/mix/kernel/up_gemv_mix.cuh"
#include "ops/linear/kernels/bf16/bf16_template_launch.cuh"

namespace ninfer::ops::detail::hc::mix {
namespace {
using Schedule = Bf16ScheduleInstance<
    Bf16A16GemvSchedule<4, 4, 1, 16, 4, Bf16ActivationAccess::Direct, Bf16WeightCache::Default,
                        Bf16PhaseOrder::Sequential, 1, 1, 2, 1>,
    kResidualWidth>;

template <Kind Form>
void down(Operands o, Scratch scratch, cudaStream_t stream) {
    const Bf16A16Operands p{static_cast<const __nv_bfloat16*>(o.normalized),
                            static_cast<const __nv_bfloat16*>(o.down), down_rows(Form),
                            kResidualWidth, o.columns};
    launch_bf16_a16_gemv<Schedule>(
        p, DownOutput<Form>{static_cast<__nv_bfloat16*>(scratch.activation), o.injection},
        LinearIdentityEpilogue{}, stream);
}
} // namespace

void gemv_down(Kind kind, Operands o, Scratch scratch, cudaStream_t stream) {
    if (kind == Kind::Block)
        down<Kind::Block>(o, scratch, stream);
    else
        down<Kind::Final>(o, scratch, stream);
}

void gemv_up(Operands o, Scratch scratch, cudaStream_t stream) {
    up_gemv_mix_kernel<<<kWidth / 16, 128, 0, stream>>>(
        o, static_cast<const __nv_bfloat16*>(scratch.activation));
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops::detail::hc::mix
