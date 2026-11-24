#include "core/weight.h"
#include "ops/linear_attention/gated_delta_net/input_conv/nvfp4/launch.h"
#include "core/device.h"
#include "ops/linear/common/output.cuh"
#include "ops/linear/kernels/nvfp4/nvfp4_schedule.cuh"
#include "ops/linear/kernels/nvfp4/nvfp4_template_launch.cuh"
#include "ops/linear_attention/gated_delta_net/input_conv/nvfp4/tma_launch.h"

namespace ninfer::ops::detail {
namespace {

using Geometry = Nvfp4N16384K5120;

using M32N64            = Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>;
using M32N128           = Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>;
using M64N128           = Nvfp4A4MmaSchedule<64, 128, 256, 4, 2, 2, 1>;
using M128N128Pipelined = Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 2, 1>;
using M128N128Resident  = Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 1, 2>;

template <class Schedule>
void launch_gemm(const Weight& weight, Tensor& qkv, Tensor& z, Nvfp4A4Workspace workspace,
                 std::int32_t tokens, cudaStream_t stream) {
    launch_nvfp4_a4_mma<Nvfp4ScheduleInstance<Schedule, Geometry::kInputRows>>(
        nvfp4_a4_operands(weight, workspace, tokens, Nvfp4ScaleLayout::RowMajor),
        LinearBf16SegmentedOutput<10240, 6144>{static_cast<__nv_bfloat16*>(qkv.data),
                                               static_cast<__nv_bfloat16*>(z.data)},
        LinearIdentityEpilogue{}, stream);
}

void launch_tma(const Weight& weight, Tensor& qkv, Tensor& z, Nvfp4A4Workspace workspace,
                std::int32_t tokens, cudaStream_t stream) {
    launch_nvfp4_a4_tma_gdn(
        nvfp4_a4_operands(weight, workspace, tokens, Nvfp4ScaleLayout::Tiled256),
        static_cast<__nv_bfloat16*>(qkv.data), static_cast<__nv_bfloat16*>(z.data), stream);
}

using Launcher = void (*)(const Weight&, Tensor&, Tensor&, Nvfp4A4Workspace, std::int32_t,
                          cudaStream_t);

struct Plan {
    Nvfp4ScaleLayout scales;
    Launcher launch;
};

Plan select_projection(int tokens) {
    if (tokens >= 512) return {Nvfp4ScaleLayout::Tiled256, launch_tma};
    if (tokens <= 64) return {Nvfp4ScaleLayout::RowMajor, launch_gemm<M32N64>};
    if (tokens <= 96) return {Nvfp4ScaleLayout::RowMajor, launch_gemm<M32N128>};
    if (tokens <= 128) return {Nvfp4ScaleLayout::RowMajor, launch_gemm<M128N128Pipelined>};
    if (tokens <= 192) return {Nvfp4ScaleLayout::RowMajor, launch_gemm<M64N128>};
    return {Nvfp4ScaleLayout::RowMajor, launch_gemm<M128N128Resident>};
}
} // namespace

void launch_nvfp4_gdn_input_a4(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                               Nvfp4A4Workspace workspace, cudaStream_t stream) {
    const auto plan = select_projection(x.ne[1]);
    launch_nvfp4_a4_quantize(x, weight, workspace, plan.scales, stream);
    plan.launch(weight, qkv, z, workspace, x.ne[1], stream);
}
} // namespace ninfer::ops::detail
