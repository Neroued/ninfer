#include "core/weight.h"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"

#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_schedule.cuh"
#include "ops/linear/nvfp4/nvfp4_template_launch.cuh"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_a4_tma_launch.h"
#include "ops/linear/common/epilogue.cuh"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

using M32N64            = Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>;
using M32N128           = Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>;
using M64N128           = Nvfp4A4MmaSchedule<64, 128, 256, 4, 2, 2, 1>;
using M128N128Pipelined = Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 2, 1>;
using M128N128Resident  = Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 1, 2>;

// This projection selects its own route, so the layout the quantizer writes below must be derived
// from the same predicate; the two are read together at the call site for that reason.
constexpr bool uses_tma(std::int32_t tokens) { return tokens >= 1024; }

template <class Geometry, class Schedule>
void launch_gemm(const Weight& weight, Tensor& residual, Nvfp4A4Workspace workspace,
                 std::int32_t tokens, cudaStream_t stream) {
    launch_nvfp4_a4_mma<Nvfp4ScheduleInstance<Schedule, Geometry::kInputRows>>(
        nvfp4_a4_operands(weight, workspace, tokens, Nvfp4ScaleLayout::RowMajor),
        LinearBf16Output{static_cast<__nv_bfloat16*>(residual.data), weight.n},
        LinearResidualAddEpilogue{{static_cast<__nv_bfloat16*>(residual.data), weight.n}}, stream);
}

template <class Geometry>
void launch_problem(const Weight& weight, Tensor& residual, Nvfp4A4Workspace workspace,
                    std::int32_t tokens, cudaStream_t stream) {
    if (tokens <= 64) {
        launch_gemm<Geometry, M32N64>(weight, residual, workspace, tokens, stream);
    } else if (tokens <= 128) {
        launch_gemm<Geometry, M32N128>(weight, residual, workspace, tokens, stream);
    } else if (tokens <= 192) {
        launch_gemm<Geometry, M64N128>(weight, residual, workspace, tokens, stream);
    } else if (tokens <= 384) {
        launch_gemm<Geometry, M128N128Resident>(weight, residual, workspace, tokens, stream);
    } else if (tokens <= 512) {
        launch_gemm<Geometry, M128N128Pipelined>(weight, residual, workspace, tokens, stream);
    } else {
        launch_gemm<Geometry, M128N128Resident>(weight, residual, workspace, tokens, stream);
    }
}

} // namespace

void nvfp4_linear_add_a4_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                Nvfp4A4Workspace workspace, cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    const auto layout = uses_tma(tokens) ? Nvfp4ScaleLayout::Tiled256 : Nvfp4ScaleLayout::RowMajor;
    launch_nvfp4_a4_quantize(x, weight, workspace, layout, stream);
    if (uses_tma(tokens)) {
        launch_nvfp4_a4_tma_linear_add(nvfp4_a4_operands(weight, workspace, tokens, layout),
                                       static_cast<__nv_bfloat16*>(residual.data), stream);
        return;
    }
    switch (resolve_nvfp4_geometry(weight.n, weight.k)) {
    case Nvfp4GeometryId::N5120K6144:
        launch_problem<Nvfp4N5120K6144>(weight, residual, workspace, tokens, stream);
        return;
    case Nvfp4GeometryId::N5120K17408:
        launch_problem<Nvfp4N5120K17408>(weight, residual, workspace, tokens, stream);
        return;
    case Nvfp4GeometryId::N14336K5120:
    case Nvfp4GeometryId::N16384K5120:
    case Nvfp4GeometryId::N34816K5120:
        break;
    }
    throw std::invalid_argument("nvfp4 linear_add: unsupported problem");
}

} // namespace ninfer::ops::detail
