#include "ops/linear/swiglu/fp8/dispatch.h"
#include "ops/linear/kernels/fp8/fp8_template_launch.cuh"
#include "ops/linear/kernels/fp8/fp8_instances.cuh"
#include "ops/linear/swiglu/token_major_mma_epilogue.cuh"

namespace ninfer::ops::detail {
namespace {
using Tma64x128 = Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 2, 1>;
using Tma64x256 = Fp8A8TmaMmaSchedule<64, 256, 128, 2, 4, 2, 1>;
using Bulk      = Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, 170, 4, 8>;
} // namespace

std::size_t fp8_linear_swiglu_partial_capacity_bytes(std::int32_t max_tokens) {
    return max_tokens > 256 ? Bulk::kPartialBytes : 0;
}

void launch_fp8_swiglu_a8(const Tensor& x, const Weight& weight, Tensor& out,
                          MemoryArena& workspace, cudaStream_t stream) {
    auto scope         = workspace.scope();
    const auto scratch = allocate_fp8_a8_workspace(
        workspace, x.ne[1], weight.k, fp8_linear_swiglu_partial_capacity_bytes(x.ne[1]));
    launch_fp8_a8_quantize(x, weight, scratch, stream);
    const auto operands = fp8_a8_operands(weight, scratch, x.ne[1]);
    const LinearBf16Output output{static_cast<__nv_bfloat16*>(out.data), weight.n / 2};
    const auto launch = [&]<class Schedule>() {
        using S = Fp8ScheduleInstance<Schedule, 5120>;
        if constexpr (S::kTmaSwizzle)
            launch_fp8_a8_tma_mma<S>(operands, output, SwiGluTokenMajorMmaEpilogue{}, stream,
                                     scratch.partials, SwiGluTokenMajorMmaRows<S>{});
        else
            launch_fp8_a8_mma<S>(operands, output, SwiGluTokenMajorMmaEpilogue{}, stream,
                                 SwiGluTokenMajorMmaRows<S>{});
    };
    if (x.ne[1] <= 16) return launch.template operator()<Fp8A8T16R64K128>();
    if (x.ne[1] <= 32) return launch.template operator()<Fp8A8T32R128K128>();
    if (x.ne[1] <= 64) return launch.template operator()<Fp8A8T64R128K256>();
    if (x.ne[1] <= 128) return launch.template operator()<Tma64x128>();
    if (x.ne[1] <= 192) return launch.template operator()<Tma64x256>();
    launch.template operator()<Bulk>();
}
} // namespace ninfer::ops::detail
