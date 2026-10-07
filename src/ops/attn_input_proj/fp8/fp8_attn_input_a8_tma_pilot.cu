#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ops/attn_input_proj/fp8/fp8_attn_input_output.cuh"
#include "ops/attn_input_proj/fp8/fp8_attn_input_plan.h"
#include "ops/common/token_slices.h"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/fp8/fp8_a8_mma_tma.cuh"
#include "ops/linear/fp8/fp8_a8_plan.h"
#include "ops/linear/fp8/fp8_operands.h"
#include "ops/linear/fp8/fp8_schedule.cuh"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// PILOT schedule: the Prefill production schedule, restated locally so the pilot
// translation unit does not reach into another .cu file's anonymous namespace.
using Fp8MmaTmaPrefillSchedule =
    Fp8A8MmaSchedule<64, 128, 128, 2, 4, 2, 2, Cache::cg, Cache::cg,
                     Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
using Schedule = Fp8ScheduleInstance<Fp8MmaTmaPrefillSchedule, 5120>;

} // namespace

// PILOT: TMA activation staging for the Prefill route. Prefill band (T > 144, matching
// production dispatch) with full 64-wide token tiles; not wired into production dispatch.
// The caller (test/bench flag) selects exact-multiple shapes.
void fp8_attn_input_a8_tma_pilot_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                       Tensor& gate, Tensor& k, Tensor& v,
                                       Fp8A8Workspace workspace, cudaStream_t stream) {
    static_assert((kFp8AttnInputQueryRows % Schedule::kBlockRows) == 0);
    static_assert((kFp8AttnInputKeyRows % Schedule::kBlockRows) == 0);
    const std::int32_t tokens = x.ne[1];
    if (tokens <= 144 || tokens % Schedule::kBlockTokens != 0) {
        throw std::invalid_argument("fp8 TMA pilot needs Prefill-band full 64-wide token tiles");
    }
    if (tokens > Schedule::kBlockTokens * kCudaGridYLimit) {
        throw std::invalid_argument("fp8 TMA pilot needs a single-launch token count");
    }
    // Validate before quantizing: the quantizer faults on misaligned codes instead of
    // throwing, so the operand checks must run before any kernel launch.
    const Fp8A8Operands operands = fp8_a8_operands(weight, workspace, tokens);
    validate_fp8_operands<Schedule>(operands);
    if (operands.rows % Schedule::kBlockRows != 0 ||
        operands.k % Schedule::kBlockK != 0 ||
        operands.k / Schedule::kBlockK < Schedule::kStages) {
        throw std::invalid_argument("fp8 TMA pilot needs complete row/K tiles and enough K stages");
    }
    launch_fp8_a8_quantize(x, weight, workspace, stream);
    const Fp8AttentionInputOutput output{
        static_cast<__nv_bfloat16*>(q.data),
        static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(gate.data),
        static_cast<__nv_bfloat16*>(v.data),
    };
    Fp8MmaTmaDescriptors descriptors{};
    descriptors.a_tile = fp8_mma_tma_activation_2d(
        workspace.codes, static_cast<std::uint64_t>(operands.k),
        static_cast<std::uint64_t>(tokens), Schedule::kBlockK, Schedule::kBlockTokens);
    constexpr auto kernel = fp8_a8_mma_tma_kernel<Schedule, true, LinearIdentityEpilogue,
                                                 Fp8AttentionInputOutput, Fp8IdentityRows>;
    constexpr int kSharedBytes = fp8_mma_shared_bytes<Schedule, LinearIdentityEpilogue> +
                                 Schedule::kStages * static_cast<int>(sizeof(std::uint64_t));
    static const cudaError_t attribute = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes);
    CUDA_CHECK(attribute);
    const int blocks = operands.rows / Schedule::kBlockRows * (tokens / Schedule::kBlockTokens);
    kernel<<<blocks, Schedule::kThreads, 0, stream>>>(
        operands, output, LinearIdentityEpilogue{}, Fp8IdentityRows{}, descriptors, 0, tokens);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
