#include "ops/attn_input_proj/fp8/fp8_attn_input_plan.h"

#include "core/device.h"
#include "ops/attn_input_proj/fp8/fp8_attn_input_output.cuh"
#include "ops/linear/fp8/fp8_a8_mma.cuh"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_output.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

using Geometry = Fp8AttnInputGeometry;

template <class Schedule, bool FullTokens>
void launch_mma(const Weight& weight, Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                Fp8A8Workspace workspace, std::int32_t tokens, cudaStream_t stream) {
    static_assert((kFp8AttnInputQueryRows % Schedule::kBlockRows) == 0);
    static_assert((kFp8AttnInputKeyRows % Schedule::kBlockRows) == 0);
    constexpr int kRowTiles = Geometry::kOutputRows / Schedule::kBlockRows;
    const int token_tiles   = (tokens + Schedule::kBlockTokens - 1) / Schedule::kBlockTokens;
    const int blocks        = kRowTiles * token_tiles;
    const Fp8AttentionInputOutput output{
        static_cast<__nv_bfloat16*>(q.data),
        static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(gate.data),
        static_cast<__nv_bfloat16*>(v.data),
    };

    static_assert(Schedule::kSharedBytes <= 48 * 1024);
    fp8_mma_kernel<Geometry, Schedule, FullTokens>
        <<<blocks, Schedule::kThreads, Schedule::kSharedBytes, stream>>>(
            workspace.codes, workspace.scales, static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales), tokens, Fp8IdentityEpilogue{},
            output);
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule>
void run(const Weight& weight, Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
         Fp8A8Workspace workspace, int tokens, cudaStream_t stream) {
    if (tokens % Schedule::kBlockTokens == 0)
        launch_mma<Schedule, true>(weight, q, gate, k, v, workspace, tokens, stream);
    else
        launch_mma<Schedule, false>(weight, q, gate, k, v, workspace, tokens, stream);
}
} // namespace

void fp8_attn_input_a8_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                              Tensor& k, Tensor& v, Fp8A8Workspace workspace, cudaStream_t stream) {
    launch_fp8_a8_quantize(x, weight, workspace, stream);
    // This Op owns its tile choices; the generic Linear schedules do not describe four-output
    // projection's short-column cost. All variants share the same activation representation.
    using Small32   = Fp8MmaSchedule<32, 64, 128, 1, 2, 3, 2, Cache::cg, Cache::cg,
                                     Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
    using Small64   = Fp8MmaSchedule<64, 64, 128, 2, 2, 3, 2, Cache::cg, Cache::cg,
                                     Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
    using ShortTail = Fp8MmaSchedule<32, 128, 128, 2, 4, 2, 2, Cache::cg, Cache::cg,
                                     Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
    using Wide128   = Fp8MmaSchedule<64, 64, 128, 2, 2, 2, 3, Cache::cg, Cache::cg,
                                     Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
    using Tail144   = Fp8MmaSchedule<48, 128, 128, 3, 4, 2, 2, Cache::cg, Cache::cg,
                                     Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
    using Prefill   = Fp8MmaSchedule<64, 128, 128, 2, 4, 2, 2, Cache::cg, Cache::cg,
                                     Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
    if (x.ne[1] <= 32)
        run<Small32>(weight, q, gate, k, v, workspace, x.ne[1], stream);
    else if (x.ne[1] <= 64)
        run<Small64>(weight, q, gate, k, v, workspace, x.ne[1], stream);
    else if (x.ne[1] <= 96)
        run<ShortTail>(weight, q, gate, k, v, workspace, x.ne[1], stream);
    else if (x.ne[1] <= 128)
        run<Wide128>(weight, q, gate, k, v, workspace, x.ne[1], stream);
    else if (x.ne[1] <= 144)
        run<Tail144>(weight, q, gate, k, v, workspace, x.ne[1], stream);
    else
        run<Prefill>(weight, q, gate, k, v, workspace, x.ne[1], stream);
}
} // namespace ninfer::ops::detail
