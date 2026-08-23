#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_plan.h"

#include "core/device.h"
#include "ops/linear/fp8/fp8_a8_mma.cuh"
#include "ops/linear/fp8/fp8_a8_plan.h"
#include "ops/linear/fp8/fp8_a8_schedule.cuh"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_output.cuh"
#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_output.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

using Geometry = Fp8MlpGateUpGeometry;
using Schedule = typename Fp8LinearA8ProductionSchedule<Geometry>::Type;

constexpr int kIntermediate = Geometry::kOutputRows / 2;
using Rows                  = Fp8SwiGluRows<Schedule::kBlockRows / 2, kIntermediate>;
static_assert((Schedule::kBlockRows % 2) == 0);

template <bool FullTokens>
void launch_mma(const Weight& weight, Tensor& out, Fp8A8Workspace workspace, std::int32_t tokens,
                cudaStream_t stream) {
    constexpr int kRowTiles = Geometry::kOutputRows / Schedule::kBlockRows;
    const int token_tiles   = (tokens + Schedule::kBlockTokens - 1) / Schedule::kBlockTokens;
    const int blocks        = kRowTiles * token_tiles;
    const Rows rows{};
    const Fp8SwiGluOutput output{static_cast<__nv_bfloat16*>(out.data), kIntermediate};

    if constexpr (Schedule::kSharedBytes > 48 * 1024) {
        static const cudaError_t attribute = cudaFuncSetAttribute(
            fp8_mma_kernel<Geometry, Schedule, FullTokens, Fp8IdentityEpilogue, Fp8SwiGluOutput,
                           Rows, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, Schedule::kSharedBytes);
        CUDA_CHECK(attribute);
    }
    fp8_mma_kernel<Geometry, Schedule, FullTokens, Fp8IdentityEpilogue, Fp8SwiGluOutput, Rows, true>
        <<<blocks, Schedule::kThreads, Schedule::kSharedBytes, stream>>>(
            workspace.codes, workspace.scales, static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales), tokens, Fp8IdentityEpilogue{}, output,
            rows);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void fp8_linear_swiglu_a8_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                 WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope = workspace.scope();
    const Fp8A8Workspace scratch =
        allocate_fp8_a8_workspace(workspace, x.ne[1], Geometry::kInputRows);
    launch_fp8_a8_quantize(x, weight, scratch, stream);
    if ((x.ne[1] % Schedule::kBlockTokens) == 0) {
        launch_mma<true>(weight, out, scratch, x.ne[1], stream);
    } else {
        launch_mma<false>(weight, out, scratch, x.ne[1], stream);
    }
}

} // namespace ninfer::ops::detail
