#include "ops/linear/swiglu/nvfp4/dispatch.h"
#include "ops/linear/swiglu/nvfp4/tma_launch.h"
#include "ops/linear/kernels/nvfp4/nvfp4_a4_plan.h"
#include "ops/linear/kernels/nvfp4/nvfp4_instances.cuh"
#include "ops/linear/kernels/nvfp4/nvfp4_template_launch.cuh"
#include "ops/linear/swiglu/paired_output.cuh"
#include "ops/linear/swiglu/token_major_mma_epilogue.cuh"

namespace ninfer::ops::detail {
namespace {
template <class Schedule, int Rows = 28672>
struct FusedSchedule : Nvfp4ScheduleInstance<Schedule, 5120> {};

using Gemv =
    Nvfp4A16GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 2>;
using T32  = Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>;
using T64  = Nvfp4A4MmaSchedule<64, 128, 256, 2, 4, 2, 1>;
using T128 = Nvfp4A4MmaSchedule<128, 64, 256, 4, 2, 2, 1>;

bool use_a4(LinearPolicy policy, int tokens) {
    if (tokens <= 0 || !valid_linear_policy(policy))
        throw std::invalid_argument("NVFP4 SwiGLU: invalid policy or token extent");
    if (policy != LinearPolicy::AllowA4 && tokens > 16)
        throw std::invalid_argument("NVFP4 SwiGLU A16 is registered only through T=16");
    return policy == LinearPolicy::AllowA4 && tokens >= 8;
}

template <class Schedule>
void launch_cp(const Nvfp4A4Operands& p, LinearBf16Output output, cudaStream_t stream) {
    using S = FusedSchedule<Schedule>;
    launch_nvfp4_a4_mma<S>(p, output, SwiGluTokenMajorMmaEpilogue{}, stream,
                           SwiGluTokenMajorMmaRows<S>{});
}
} // namespace

std::size_t nvfp4_swiglu_n28672_k5120_capacity(LinearPolicy policy, int min_tokens,
                                               int max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens)
        throw std::invalid_argument("NVFP4 SwiGLU workspace: invalid interval");
    (void)use_a4(policy, min_tokens);
    return use_a4(policy, max_tokens) ? nvfp4_a4_workspace_capacity_bytes(max_tokens, 5120) : 0;
}

void launch_nvfp4_swiglu_n28672_k5120(const Tensor& x, const Weight& w, Tensor& y,
                                      LinearPolicy policy, MemoryArena& workspace,
                                      cudaStream_t stream) {
    const int tokens = x.ne[1];
    const LinearBf16Output output{static_cast<__nv_bfloat16*>(y.data), 14336};
    if (!use_a4(policy, tokens)) {
        const auto p = nvfp4_a16_operands(x, w);
        if (tokens == 1)
            launch_nvfp4_a16_gemv<FusedSchedule<Gemv>>(p, output, SwiGluPairEpilogue{}, stream,
                                                       SwiGluBranchRows<1>{});
        else if (tokens <= 8)
            launch_nvfp4_a16_sliced_k_mma<FusedSchedule<Nvfp4SlicedInstance<8, 8, 2>>>(
                p, output, SwiGluPairEpilogue{}, stream, SwiGluBranchRows<8>{});
        else
            launch_nvfp4_a16_sliced_k_mma<FusedSchedule<Nvfp4SlicedInstance<16, 8, 2>>>(
                p, output, SwiGluPairEpilogue{}, stream, SwiGluBranchRows<8>{});
        return;
    }
    auto scope         = workspace.scope();
    const auto scratch = allocate_nvfp4_a4_workspace(workspace, tokens, 5120);
    const auto layout  = tokens <= 128   ? Nvfp4ScaleLayout::RowMajor
                         : tokens <= 384 ? Nvfp4ScaleLayout::Tiled128
                                         : Nvfp4ScaleLayout::Tiled256;
    launch_nvfp4_a4_quantize(x, w, scratch, layout, stream);
    const auto p = nvfp4_a4_operands(w, scratch, tokens, layout);
    if (tokens <= 32)
        launch_cp<T32>(p, output, stream);
    else if (tokens <= 64)
        launch_cp<T64>(p, output, stream);
    else if (tokens <= 128)
        launch_cp<T128>(p, output, stream);
    else
        launch_nvfp4_swiglu_n28672_k5120_tma(p, output.data, stream);
}
} // namespace ninfer::ops::detail
