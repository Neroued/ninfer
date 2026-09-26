#include "ops/linear/fp8/fp8_template_launch.cuh"
#include "core/weight.h"
#include "ops/attn_input_proj/fp8/fp8_attn_input_plan.h"

#include "core/device.h"
#include "ops/attn_input_proj/fp8/fp8_attn_input_output.cuh"
#include "ops/linear/fp8/fp8_a8_mma.cuh"
#include "ops/linear/fp8/fp8_schedule.cuh"
#include "ops/linear/common/epilogue.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

using Geometry = Fp8N14336K5120;

template <class Schedule>
void run(const Weight& weight, Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
         Fp8A8Workspace workspace, int tokens, cudaStream_t stream) {
    const Fp8AttentionInputOutput output{
        static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(v.data)};
    launch_fp8_a8_mma<Fp8ScheduleInstance<Schedule, 5120>>(
        fp8_a8_operands(weight, workspace, tokens), output, LinearIdentityEpilogue{}, stream);
}
} // namespace

void fp8_attn_input_a8_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                              Tensor& k, Tensor& v, Fp8A8Workspace workspace, cudaStream_t stream) {
    launch_fp8_a8_quantize(x, weight, workspace, stream);
    // This Op owns its tile choices; the generic Linear schedules do not describe four-output
    // projection's short-column cost. All variants share the same activation representation.
    using Small32   = Fp8A8MmaSchedule<32, 64, 128, 1, 2, 3, 2, Cache::cg, Cache::cg,
                                       Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
    using Small64   = Fp8A8MmaSchedule<64, 64, 128, 2, 2, 3, 2, Cache::cg, Cache::cg,
                                       Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
    using ShortTail = Fp8A8MmaSchedule<32, 128, 128, 2, 4, 2, 2, Cache::cg, Cache::cg,
                                       Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
    using Wide128   = Fp8A8MmaSchedule<64, 64, 128, 2, 2, 2, 3, Cache::cg, Cache::cg,
                                       Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
    using Tail144   = Fp8A8MmaSchedule<48, 128, 128, 3, 4, 2, 2, Cache::cg, Cache::cg,
                                       Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
    using Prefill   = Fp8A8MmaSchedule<64, 128, 128, 2, 4, 2, 2, Cache::cg, Cache::cg,
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
