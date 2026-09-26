#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_plan.h"
#include "ops/linear/fp8/fp8_template_launch.cuh"
#include "ops/linear/fp8/fp8_instances.cuh"
#include "ops/linear_swiglu/token_major_mma_epilogue.cuh"

namespace ninfer::ops::detail {
void fp8_linear_swiglu_a8_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                 WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope         = workspace.scope();
    const auto scratch = allocate_fp8_a8_workspace(workspace, x.ne[1], weight.k);
    launch_fp8_a8_quantize(x, weight, scratch, stream);
    const auto operands = fp8_a8_operands(weight, scratch, x.ne[1]);
    const LinearBf16Output output{static_cast<__nv_bfloat16*>(out.data), weight.n / 2};
    const auto launch = [&]<class Schedule>() {
        using S = Fp8ScheduleInstance<Schedule, 5120>;
        launch_fp8_a8_mma<S>(operands, output, SwiGluTokenMajorMmaEpilogue{}, stream,
                             SwiGluTokenMajorMmaRows<S>{});
    };
    if (x.ne[1] <= 16) return launch.template operator()<Fp8A8T16R64K128>();
    if (x.ne[1] <= 32) return launch.template operator()<Fp8A8T32R128K128>();
    if (x.ne[1] <= 64) return launch.template operator()<Fp8A8T64R128K128>();
    if (x.ne[1] <= 96) return launch.template operator()<Fp8A8T32R128K128>();
    launch.template operator()<Fp8A8T64R128K128>();
}
} // namespace ninfer::ops::detail
