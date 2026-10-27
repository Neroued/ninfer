#include "ops/softmax_attention/kpool_mla/prepare/projection.cuh"
#include "ops/softmax_attention/kpool_mla/prepare/post.cuh"

namespace ninfer::ops::detail::kpool {
namespace {
template <class S>
using Static = Bf16ScheduleInstance<S, 4096>;
template <class S>
using Rows = Static<Bf16RowTailSchedule<S>>;
using Gemv =
    Static<Bf16A16GemvSchedule<4, 4, 1, 8, 4, Bf16ActivationAccess::Direct,
                               Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 1, 4, 2>>;
template <int Columns>
using Simt = Static<
    Bf16A16SimtSchedule<4, 4, 1, 8, 4, 2, Bf16SimtActivationAccess::DirectStream,
                        Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 4, 1, 2, Columns>>;
using Short        = Static<Bf16A16SlicedKMmaSchedule<16, 8, 16, 64, 2>>;
using Small        = Static<Bf16A16SlicedKMmaSchedule<32, 16, 4, 64, 2>>;
using HistorySmall = Static<Bf16A16SlicedKMmaSchedule<16, 8, 4, 64, 2>>;
using Medium =
    Static<Bf16A16MmaSchedule<32, 32, 128, 16, 8, 3, 1, Cache::cg, Cache::cg,
                              Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>>;
using TmaSmall   = Rows<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 32, 3>>;
using TmaHistory = Rows<Bf16A16TmaMmaSchedule<128, 64, 64, 64, 32, 2>>;
using TmaLarge   = Rows<Bf16A16TmaMmaSchedule<64, 128, 64, 32, 64, 2>>;
using SplitSmall = Rows<Bf16A16TmaMmaSchedule<32, 32, 64, 16, 16, 2>>;
using SplitLarge = Rows<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 32, 2>>;
} // namespace

void launch_prepare(const PrepareArguments& inputs, int window_columns, cudaStream_t stream) {
    auto a           = inputs;
    const auto total = std::int64_t(a.tokens) * a.batch;
    for (std::int64_t begin = 0; begin < total; begin += window_columns) {
        const int columns = static_cast<int>(std::min<std::int64_t>(window_columns, total - begin));
        const auto route  = select_projection(columns, a.rows);
        a.splits          = projection_splits(route);
        set_window(a, begin, columns);
        const Bf16A16Operands p{a.x + begin * 4096, a.weight, a.rows, 4096, columns};
        const ProjectionOutput output{a, begin};
        const ProjectionPartialOutput partial{a.projected, a.rows, columns};
        switch (route) {
        case ProjectionRoute::Gemv:
            launch_bf16_a16_gemv<Gemv>(p, output, LinearIdentityEpilogue{}, stream);
            break;
        case ProjectionRoute::Simt2:
            launch_bf16_a16_simt<Simt<2>>(p, output, LinearIdentityEpilogue{}, stream);
            break;
        case ProjectionRoute::Simt4:
            launch_bf16_a16_simt<Simt<4>>(p, output, LinearIdentityEpilogue{}, stream);
            break;
        case ProjectionRoute::Sliced8:
            launch_bf16_a16_sliced_k_mma<Short>(p, output, LinearIdentityEpilogue{}, stream);
            break;
        case ProjectionRoute::Sliced16:
            launch_bf16_a16_sliced_k_mma<Small>(p, output, LinearIdentityEpilogue{}, stream);
            break;
        case ProjectionRoute::SlicedHistory:
            launch_bf16_a16_sliced_k_mma<HistorySmall>(p, output, LinearIdentityEpilogue{}, stream);
            break;
        case ProjectionRoute::Mma32:
            launch_bf16_a16_mma<Medium>(p, output, LinearIdentityEpilogue{}, stream);
            break;
        case ProjectionRoute::TmaSmall:
            launch_bf16_a16_tma_mma<TmaSmall>(p, output, ProjectionTmaEpilogue{}, stream);
            break;
        case ProjectionRoute::TmaHistory:
            launch_bf16_a16_tma_mma<TmaHistory>(p, output, ProjectionTmaEpilogue{}, stream);
            break;
        case ProjectionRoute::Tma128:
            launch_bf16_a16_tma_mma<TmaLarge>(p, output, ProjectionTmaEpilogue{}, stream);
            break;
        case ProjectionRoute::Split32:
            launch_bf16_a16_tma_mma<SplitSmall, 4>(p, partial, ProjectionTmaEpilogue{}, stream);
            break;
        case ProjectionRoute::Split64:
            launch_bf16_a16_tma_mma<SplitLarge, 4>(p, partial, ProjectionTmaEpilogue{}, stream);
            break;
        }
        int jobs = columns * (1 + (a.query_rank != 0));
        for (int b = a.first_row; b <= a.last_row; ++b) {
            const int start = b == a.first_row ? a.first_column : 0;
            const int end   = b == a.last_row ? a.last_stop : a.tokens;
            jobs += (end - start + 6) / 4;
        }
        if (a.fp8_index)
            prepare_post_kernel<true><<<jobs, 128, 0, stream>>>(a, begin, columns);
        else
            prepare_post_kernel<false><<<jobs, 128, 0, stream>>>(a, begin, columns);
        CUDA_CHECK(cudaGetLastError());
    }
}
} // namespace ninfer::ops::detail::kpool
