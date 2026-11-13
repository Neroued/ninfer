#include "ops/linear/swiglu/nvfp4/dispatch.h"
#include "ops/linear/swiglu/nvfp4/tma_launch.h"
#include "ops/linear/kernels/nvfp4/nvfp4_a4_plan.h"
#include "ops/linear/kernels/nvfp4/nvfp4_instances.cuh"
#include "ops/linear/kernels/nvfp4/nvfp4_template_launch.cuh"
#include "ops/linear/swiglu/paired_output.cuh"
#include "ops/linear/swiglu/token_major_mma_epilogue.cuh"
#include "ops/linear/swiglu/row_major_mma_epilogue.cuh"

namespace ninfer::ops::detail {
namespace {
template <class Schedule>
struct FusedSchedule : Nvfp4ScheduleInstance<Schedule, 5120> {};

using Gemv =
    Nvfp4A16GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 2>;
template <int Capacity>
using Small =
    Nvfp4ScheduleInstance<Nvfp4A16SlicedKMmaSchedule<8, 8, 2, Cache::ca, Cache::cg,
                                                     Nvfp4ActivationStage::ActiveOnly, 2, true>,
                          5120, Capacity>;
using T16  = Nvfp4A4MmaSchedule<16, 64, 512, 1, 4, 3, 2>;
using T32  = Nvfp4A4MmaSchedule<32, 64, 512, 2, 4, 2, 1>;
using T64  = Nvfp4A4MmaSchedule<64, 64, 512, 2, 4, 2, 1>;
using T128 = Nvfp4A4MmaSchedule<128, 64, 256, 4, 2, 2, 1>;

bool use_a4(LinearPolicy policy, int tokens) {
    if (tokens <= 0 || !valid_linear_policy(policy))
        throw std::invalid_argument("NVFP4 SwiGLU: invalid policy or token extent");
    return policy == LinearPolicy::AllowA4 && tokens >= 8;
}

template <class Schedule>
void launch_cp(const Nvfp4A4Operands& p, LinearBf16Output output, cudaStream_t stream) {
    using S = FusedSchedule<Schedule>;
    launch_nvfp4_a4_mma<S>(p, output, SwiGluTokenMajorMmaEpilogue{}, stream,
                           SwiGluTokenMajorMmaRows<S>{});
}

template <class Schedule>
void launch_a16_mma(const Nvfp4A16Operands& p, LinearBf16Output output, cudaStream_t stream) {
    using S = FusedSchedule<Schedule>;
    launch_nvfp4_a16_mma<S>(p, output, SwiGluRowMajorMmaEpilogue{}, stream,
                            SwiGluRowMajorMmaRows<S>{});
}

void launch_large_a16(const Nvfp4A16Operands& p, LinearBf16Output output, cudaStream_t stream) {
    if (p.tokens <= 32)
        launch_a16_mma<Nvfp4A16MmaSchedule<64, 32, 128, 32, 16, 1, 3>>(p, output, stream);
    else if (p.tokens <= 64 || (p.tokens > 112 && p.tokens <= 128))
        launch_a16_mma<Nvfp4A16MmaSchedule<64, 64, 64, 64, 16, 2, 2>>(p, output, stream);
    else if (p.tokens <= 80)
        launch_a16_mma<Nvfp4A16MmaSchedule<64, 80, 64, 64, 16, 2, 3>>(p, output, stream);
    else if (p.tokens <= 96)
        launch_a16_mma<Nvfp4A16MmaSchedule<64, 96, 64, 32, 16, 2, 2>>(p, output, stream);
    else if (p.tokens > 128 && p.tokens <= 192)
        launch_a16_mma<Nvfp4A16MmaSchedule<128, 64, 64, 64, 16, 2, 2>>(p, output, stream);
    else {
        using S = Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2, Cache::cg, Cache::cg,
                                      Nvfp4MmaFragmentPipeline::PingPong, false, 4>;
        launch_a16_mma<S>(p, output, stream);
    }
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
        if (tokens > 16)
            launch_large_a16(p, output, stream);
        else if (tokens == 1)
            launch_nvfp4_a16_gemv<FusedSchedule<Gemv>>(p, output, SwiGluPairEpilogue{}, stream,
                                                       SwiGluBranchRows<1>{});
        else if (tokens <= 4)
            launch_nvfp4_a16_sliced_k_mma<Small<4>>(p, output, SwiGluPairEpilogue{}, stream,
                                                    SwiGluBranchRows<8>{});
        else if (tokens <= 8)
            launch_nvfp4_a16_sliced_k_mma<Small<8>>(p, output, SwiGluPairEpilogue{}, stream,
                                                    SwiGluBranchRows<8>{});
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
    if (tokens <= 16)
        launch_cp<T16>(p, output, stream);
    else if (tokens <= 32)
        launch_cp<T32>(p, output, stream);
    else if (tokens <= 64)
        launch_cp<T64>(p, output, stream);
    else if (tokens <= 128)
        launch_cp<T128>(p, output, stream);
    else
        launch_nvfp4_swiglu_n28672_k5120_tma(p, output.data, stream);
}
} // namespace ninfer::ops::detail
