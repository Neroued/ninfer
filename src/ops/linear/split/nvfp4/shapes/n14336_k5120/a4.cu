#include "ops/linear/split/nvfp4/shapes/n14336_k5120/launch.h"
#include "core/weight.h"
#include "core/device.h"
#include "ops/linear/kernels/nvfp4/nvfp4_schedule.cuh"
#include "ops/linear/kernels/nvfp4/nvfp4_template_launch.cuh"
#include <cuda_bf16.h>
#include <cstdint>

namespace ninfer::ops::detail::linear_split::nvfp4_n14336_k5120 {

namespace {

using Geometry = Nvfp4N14336K5120;

using M32N64            = Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>;
using M32N128           = Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>;
using M64N128           = Nvfp4A4MmaSchedule<64, 128, 256, 4, 2, 2, 1>;
using M128N128Pipelined = Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 2, 1>;
using M128N128Resident  = Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 1, 2>;

// The quantizer and contraction must select the same activation scale layout.
constexpr bool uses_tma(std::int32_t tokens) { return tokens >= 512; }

template <class Schedule>
void launch_gemm(const Weight& weight, std::span<const Tensor> outputs, Nvfp4A4Workspace workspace,
                 std::int32_t tokens, cudaStream_t stream) {
    launch_nvfp4_a4_mma<Nvfp4ScheduleInstance<Schedule, Geometry::kInputRows>>(
        nvfp4_a4_operands(weight, workspace, tokens, Nvfp4ScaleLayout::RowMajor),
        LinearBf16SegmentedOutput<6144, 1024, 6144, 1024>{
            static_cast<__nv_bfloat16*>(outputs[0].data),
            static_cast<__nv_bfloat16*>(outputs[1].data),
            static_cast<__nv_bfloat16*>(outputs[2].data),
            static_cast<__nv_bfloat16*>(outputs[3].data)},
        LinearIdentityEpilogue{}, stream);
}

} // namespace

void launch_a4(const Tensor& x, const Weight& weight, std::span<const Tensor> outputs,
               Nvfp4A4Workspace workspace, cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    const auto layout =
        uses_tma(tokens) ? (tokens < 1024 ? Nvfp4ScaleLayout::Tiled128 : Nvfp4ScaleLayout::Tiled256)
                         : Nvfp4ScaleLayout::RowMajor;
    launch_nvfp4_a4_quantize(x, weight, workspace, layout, stream);
    if (uses_tma(tokens)) {
        launch_a4_tma(nvfp4_a4_operands(weight, workspace, tokens, layout), outputs, stream);
    } else if (tokens <= 64) {
        launch_gemm<M32N64>(weight, outputs, workspace, tokens, stream);
    } else if (tokens <= 96) {
        launch_gemm<M32N128>(weight, outputs, workspace, tokens, stream);
    } else if (tokens <= 128) {
        launch_gemm<M128N128Pipelined>(weight, outputs, workspace, tokens, stream);
    } else if (tokens <= 192) {
        launch_gemm<M64N128>(weight, outputs, workspace, tokens, stream);
    } else if (tokens <= 384) {
        launch_gemm<M128N128Resident>(weight, outputs, workspace, tokens, stream);
    } else {
        launch_gemm<M128N128Pipelined>(weight, outputs, workspace, tokens, stream);
    }
}

} // namespace ninfer::ops::detail::linear_split::nvfp4_n14336_k5120
