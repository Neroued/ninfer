#include "ops/attn_input_proj/fp8/fp8_attn_input_plan.h"

#include "core/device.h"
#include "ops/attn_input_proj/fp8/fp8_attn_input_output.cuh"
#include "ops/linear/fp8/fp8_a8_schedule.cuh"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_output.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

using Geometry = Fp8AttnInputGeometry;
using Schedule = typename Fp8LinearA8ProductionSchedule<Geometry>::Type;

static_assert((kFp8AttnInputQueryRows % Schedule::kBlockRows) == 0);
static_assert((kFp8AttnInputKeyRows % Schedule::kBlockRows) == 0);
static_assert((kFp8AttnInputGateRows % Schedule::kBlockRows) == 0);

template <bool FullTokens>
void launch_mma(const Weight& weight, Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                Fp8A8Workspace workspace, std::int32_t tokens, cudaStream_t stream) {
    constexpr int kRowTiles = Geometry::kOutputRows / Schedule::kBlockRows;
    const int token_tiles   = (tokens + Schedule::kBlockTokens - 1) / Schedule::kBlockTokens;
    const int blocks        = kRowTiles * token_tiles;
    const Fp8AttentionInputOutput output{
        static_cast<__nv_bfloat16*>(q.data),
        static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(gate.data),
        static_cast<__nv_bfloat16*>(v.data),
    };

    if constexpr (Schedule::kSharedBytes > 48 * 1024) {
        static const cudaError_t attribute = cudaFuncSetAttribute(
            fp8_mma_kernel<Geometry, Schedule, FullTokens, Fp8IdentityEpilogue,
                           Fp8AttentionInputOutput>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, Schedule::kSharedBytes);
        CUDA_CHECK(attribute);
    }
    fp8_mma_kernel<Geometry, Schedule, FullTokens>
        <<<blocks, Schedule::kThreads, Schedule::kSharedBytes, stream>>>(
            workspace.codes, workspace.scales, static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales), tokens, Fp8IdentityEpilogue{},
            output);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void fp8_attn_input_a8_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                              Tensor& k, Tensor& v, Fp8A8Workspace workspace, cudaStream_t stream) {
    launch_fp8_a8_quantize(x, weight, workspace, stream);
    if ((x.ne[1] % Schedule::kBlockTokens) == 0) {
        launch_mma<true>(weight, q, gate, k, v, workspace, x.ne[1], stream);
    } else {
        launch_mma<false>(weight, q, gate, k, v, workspace, x.ne[1], stream);
    }
}

} // namespace ninfer::ops::detail
