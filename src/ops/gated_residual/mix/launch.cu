#include "ops/gated_residual/mix/launch.h"
#include "ops/common/host/launch.h"
#include "ops/gated_residual/mix/kernel/output.cuh"
#include "ops/gated_residual/mix/kernel/up_mma_mix.cuh"
#include "ops/gated_residual/mix/kernel/up_tma_mix.cuh"
#include "ops/linear/kernels/bf16/bf16_template_launch.cuh"
#include "ops/linear/kernels/bf16/bf16_split_k_launch.cuh"

#include <type_traits>

namespace ninfer::ops::detail::hc::mix {
namespace {
template <Kind Form, class Schedule>
using DownRows = std::conditional_t<Form == Kind::Block, Bf16RowTailSchedule<Schedule>, Schedule>;

template <Kind Form, class Schedule>
using DownSchedule = DownRows<Form, Bf16ScheduleInstance<Schedule, kResidualWidth>>;

template <int Rows, int Tokens, int K, int MinBlocks>
using SplitSchedule =
    Bf16A16MmaSchedule<Rows, Tokens, K, 16, 8, 3, MinBlocks, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>;

using DownGemv = Bf16ScheduleInstance<
    Bf16A16GemvSchedule<2, 1, 1, 16, 4, Bf16ActivationAccess::Direct, Bf16WeightCache::Default,
                        Bf16PhaseOrder::Sequential, 1, 1, 4, 2>,
    kResidualWidth>;
using DownSimt = Bf16ScheduleInstance<
    Bf16A16SimtSchedule<8, 2, 1, 8, 4, 2, Bf16SimtActivationAccess::DirectStream,
                        Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 4, 1, 2, 2>,
    kResidualWidth>;

template <Kind Form>
void down(Operands o, Down schedule, Scratch scratch, cudaStream_t stream) {
    const Bf16A16Operands p{static_cast<const __nv_bfloat16*>(o.normalized),
                            static_cast<const __nv_bfloat16*>(o.down), down_rows(Form),
                            kResidualWidth, o.columns};
    const DownOutput<Form> output{static_cast<__nv_bfloat16*>(scratch.activation), o.injection};
    const LinearIdentityEpilogue identity;
    const Bf16SplitKWorkspace partials{static_cast<float*>(scratch.partials.data),
                                       scratch.partials.bytes};
    switch (schedule) {
    case Down::Gemv:
        launch_bf16_a16_gemv<DownGemv>(p, output, identity, stream);
        break;
    case Down::Simt:
        launch_bf16_a16_simt<DownSimt>(p, output, identity, stream);
        break;
    case Down::Split8x16:
        launch_bf16_a16_split_k_mma<DownSchedule<Form, SplitSchedule<16, 16, 128, 2>>, 8>(
            p, output, identity, partials, stream);
        break;
    case Down::Split8x32:
        launch_bf16_a16_split_k_mma<DownSchedule<Form, SplitSchedule<32, 32, 64, 1>>, 8>(
            p, output, identity, partials, stream);
        break;
    case Down::TmaSplit2:
        launch_bf16_a16_tma_split_k_mma<
            DownSchedule<Form, Bf16A16TmaMmaSchedule<32, 64, 64, 16, 8, 4, 1>>, 2>(
            p, output, identity, partials, stream);
        break;
    case Down::TmaSplit4:
        launch_bf16_a16_tma_split_k_mma<
            DownSchedule<Form, Bf16A16TmaMmaSchedule<32, 64, 64, 16, 8, 4, 1>>, 4>(
            p, output, identity, partials, stream);
        break;
    case Down::Tma96Split4:
        launch_bf16_a16_tma_split_k_mma<
            DownSchedule<Form, Bf16A16TmaMmaSchedule<32, 96, 64, 16, 16, 3, 1>>, 4>(
            p, output, identity, partials, stream);
        break;
    case Down::Tma96Split8:
        launch_bf16_a16_tma_split_k_mma<
            DownSchedule<Form, Bf16A16TmaMmaSchedule<32, 96, 64, 16, 16, 3, 1>>, 8>(
            p, output, identity, partials, stream);
        break;
    case Down::Tma80:
        launch_bf16_a16_tma_mma<DownSchedule<Form, Bf16A16TmaMmaSchedule<32, 80, 64, 16, 8, 5, 1>>>(
            p, output, identity, stream);
        break;
    }
}

template <int Features, int Tokens>
void folded_up(Operands o, Scratch scratch, cudaStream_t stream) {
    using S               = FoldedUpSchedule<Features, Tokens>;
    constexpr auto kernel = up_mma_mix_kernel<Features, Tokens>;
    const int bytes       = common::prepare_shared<S::kSharedBytes, kernel, true>();
    common::for_each_token_slice(o.columns, Tokens, [&](int offset, int count) {
        const unsigned blocks = (kWidth / Features) * common::div_up(count, Tokens);
#if defined(_MSC_VER)
        // MSVC's cudafe cannot launch through the constexpr function-pointer variable
        // captured by this lambda (C2326), so reference the kernel template directly.
        up_mma_mix_kernel<Features, Tokens>
            <<<blocks, S::kThreads, bytes, stream>>>(
                o, static_cast<const __nv_bfloat16*>(scratch.activation), offset);
#else
        kernel<<<blocks, S::kThreads, bytes, stream>>>(
            o, static_cast<const __nv_bfloat16*>(scratch.activation), offset);
#endif
        CUDA_CHECK(cudaGetLastError());
    });
}

template <int Features>
void tma_folded_up(Operands o, Scratch scratch, cudaStream_t stream) {
    using S = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<kStreams * Features, 64, 64, 32, 16, 2, 1>,
                                   kRank>;
    const Bf16A16Operands p{static_cast<const __nv_bfloat16*>(scratch.activation),
                            static_cast<const __nv_bfloat16*>(o.up), kResidualWidth, kRank,
                            o.columns};
    launch_bf16_a16_tma_mma<S, 1, LinearBf16Output, TmaMixEpilogue, StripedTmaWeightAccess>(
        p, {static_cast<__nv_bfloat16*>(o.output), kWidth},
        {static_cast<const __nv_bfloat16*>(o.normalized)}, stream);
}

} // namespace

void launch(Kind kind, Operands o, Plan p, Scratch scratch, CudaExecutionView execution) {
    if (kind == Kind::Block)
        down<Kind::Block>(o, p.down, scratch, execution.stream());
    else
        down<Kind::Final>(o, p.down, scratch, execution.stream());
    switch (p.up) {
    case Up::Folded16x16:
        folded_up<16, 16>(o, scratch, execution.stream());
        break;
    case Up::Folded32x32:
        folded_up<32, 32>(o, scratch, execution.stream());
        break;
    case Up::TmaFolded16x64:
        tma_folded_up<16>(o, scratch, execution.stream());
        break;
    case Up::TmaFolded32x64:
        tma_folded_up<32>(o, scratch, execution.stream());
        break;
    }
}
} // namespace ninfer::ops::detail::hc::mix
