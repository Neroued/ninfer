#include "ops/linear_attention/gated_delta_net/input_conv/fp8/launch.h"
#include "ops/linear/common/output.cuh"
#include "ops/linear/kernels/fp8/fp8_template_launch.cuh"
#include "ops/linear/kernels/fp8/fp8_instances.cuh"

namespace ninfer::ops::detail {
namespace {
using Tma64x128  = Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 2, 1>;
using Tma192x128 = Fp8A8TmaMmaSchedule<192, 128, 128, 3, 4, 2, 1>;
using MidBulk = Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 128, 128, 2, 4, 2, 1>, 170, 4, 8>;
using Bulk    = Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, 170, 4, 8>;

using Output = LinearBf16SegmentedOutput<10240, 6144>;

template <class Schedule>
void launch_projection(const Fp8A8Operands& operands, Output output, float* partials,
                       cudaStream_t stream) {
    using S = Fp8ScheduleInstance<Schedule, 5120>;
    if constexpr (S::kTmaSwizzle)
        launch_fp8_a8_tma_mma<S>(operands, output, LinearIdentityEpilogue{}, stream, partials);
    else
        launch_fp8_a8_mma<S>(operands, output, LinearIdentityEpilogue{}, stream);
}

using Launcher = void (*)(const Fp8A8Operands&, Output, float*, cudaStream_t);

struct Plan {
    Launcher launch;
    std::size_t partial_bytes;
};

Plan select_projection(int columns) {
    if (columns <= 32) return {launch_projection<Fp8A8T32R32K128>, 0};
    if (columns <= 64) return {launch_projection<Fp8A8T64R128K256>, 0};
    if (columns <= 128) return {launch_projection<Tma64x128>, 0};
    if (columns <= 192) return {launch_projection<Tma192x128>, 0};
    const auto partial_bytes = columns > 256 ? Bulk::kPartialBytes : 0;
    if (columns > 384 && columns <= 512) return {launch_projection<MidBulk>, partial_bytes};
    return {launch_projection<Bulk>, partial_bytes};
}
} // namespace

std::size_t fp8_gdn_input_partial_capacity_bytes(std::int32_t max_tokens) {
    return select_projection(max_tokens).partial_bytes;
}

void launch_fp8_gdn_input_a8(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                             Fp8A8Workspace workspace, cudaStream_t stream) {
    const auto plan = select_projection(x.ne[1]);
    launch_fp8_a8_quantize(x, weight, workspace, stream);
    const Output output{static_cast<__nv_bfloat16*>(qkv.data), static_cast<__nv_bfloat16*>(z.data)};
    plan.launch(fp8_a8_operands(weight, workspace, x.ne[1]), output, workspace.partials, stream);
}
} // namespace ninfer::ops::detail
