#include "ops/linear/fp8/fp8_template_launch.cuh"
#include "core/weight.h"
#include "ops/linear_add/fp8/fp8_linear_add_plan.h"

#include "core/device.h"
#include "ops/linear/fp8/fp8_a8_mma.cuh"
#include "ops/linear/fp8/fp8_a8_plan.h"
#include "ops/linear/fp8/fp8_instances.cuh"
#include "ops/linear/fp8/fp8_schedule.cuh"
#include "ops/linear/common/epilogue.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

template <class Geometry>
void launch_problem(const Weight& weight, Tensor& residual, Fp8A8Workspace workspace, int tokens,
                    cudaStream_t stream) {
    auto* data = static_cast<__nv_bfloat16*>(residual.data);
    const LinearBf16Output output{data, weight.n};
    const LinearResidualAddEpilogue epilogue{{data, weight.n}};
    const auto launch = [&]<class Schedule>() {
        launch_fp8_a8_mma<Fp8ScheduleInstance<Schedule, Geometry::kInputRows>>(
            fp8_a8_operands(weight, workspace, tokens), output, epilogue, stream);
    };
    if (tokens <= 64) return launch.template operator()<Fp8A8T32R32K128>();
    if (tokens <= 128) return launch.template operator()<Fp8A8T64R64K128>();
    launch.template operator()<Fp8A8T64R128K128>();
}

} // namespace

void fp8_linear_add_a8_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                              WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope                   = workspace.scope();
    const Fp8A8Workspace scratch = allocate_fp8_a8_workspace(workspace, x.ne[1], weight.k);
    launch_fp8_a8_quantize(x, weight, scratch, stream);
    switch (resolve_fp8_geometry(weight.n, weight.k)) {
    case Fp8GeometryId::N5120K6144:
        launch_problem<Fp8N5120K6144>(weight, residual, scratch, x.ne[1], stream);
        return;
    case Fp8GeometryId::N5120K17408:
        launch_problem<Fp8N5120K17408>(weight, residual, scratch, x.ne[1], stream);
        return;
    case Fp8GeometryId::N14336K5120:
    case Fp8GeometryId::N16384K5120:
    case Fp8GeometryId::N34816K5120:
    case Fp8GeometryId::N248320K5120:
        break;
    }
    throw std::invalid_argument("fp8 linear_add: unsupported problem");
}

} // namespace ninfer::ops::detail
