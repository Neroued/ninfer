#include "ops/softmax_attention/kpool_mla/prepare/projection.cuh"
#include "ops/softmax_attention/kpool_mla/prepare/post.cuh"
#include "ops/linear/kernels/fp8/native_a16.cuh"
#include "ops/linear/kernels/mxfp8/a8.cuh"
#include "ops/linear/kernels/fp8_block/operands.h"
#include "ops/linear/kernels/fp8/native_a16_pair.cuh"
#include "ops/linear/kernels/mxfp8/a8_tma.cuh"

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

template <class S>
using Native = NativeFp8ScheduleInstance<S, 4096>;

void launch_encoded_projection(const PrepareArguments& a, int parent, std::int64_t begin,
                               int columns, ProjectionRoute route, cudaStream_t stream) {
    const auto& weight = a.weights[parent];
    const auto* x      = a.x + begin * 4096;
    const auto* codes  = static_cast<const std::uint8_t*>(weight.codes);
    const ProjectionOutput output{a, begin, projection_row_map(a.mapping, parent)};
    using Medium   = Native<NativeFp8MmaSchedule<32, 32, 128, 16, 16, 2>>;
    using Large    = Native<NativeFp8MmaSchedule<128, 64, 128, 32, 32, 2>>;
    const auto a16 = [&](const auto& p) {
        switch (route) {
        case ProjectionRoute::Fp8Simt1:
            launch_native_fp8_a16_simt<Native<NativeFp8A16SimtSchedule<1, 8, 1, 16>>>(
                p, output, LinearIdentityEpilogue{}, stream, Fp8IdentityRows{});
            break;
        case ProjectionRoute::Fp8Simt4:
            launch_native_fp8_a16_simt<Native<NativeFp8A16SimtSchedule<4, 8, 1, 16>>>(
                p, output, LinearIdentityEpilogue{}, stream, Fp8IdentityRows{});
            break;
        case ProjectionRoute::Fp8Sliced8:
            launch_native_fp8_a16_sliced_k<Native<NativeFp8A16SlicedKSchedule<16, 8, 8, 128, 1>>>(
                p, output, LinearIdentityEpilogue{}, stream);
            break;
        case ProjectionRoute::Fp8Sliced16:
            launch_native_fp8_a16_sliced_k<Native<NativeFp8A16SlicedKSchedule<16, 16, 4, 128, 2>>>(
                p, output, LinearIdentityEpilogue{}, stream);
            break;
        case ProjectionRoute::Fp8Mma32:
            launch_native_fp8_a16_mma<Medium>(p, output, ProjectionTileEpilogue{}, stream,
                                              Fp8IdentityRows{});
            break;
        case ProjectionRoute::Fp8Mma64:
            launch_native_fp8_a16_mma<Large>(p, output, ProjectionTileEpilogue{}, stream,
                                             Fp8IdentityRows{});
            break;
        default:
            throw std::logic_error("Kpool Prepare: invalid A16 projection route");
        }
    };
    if (weight.format == QType::MXFP8) {
        const auto* scales = static_cast<const std::uint8_t*>(weight.scales);
        if (projection_uses_a8(route)) {
            const Mxfp8A8Operands p{
                a.activation_codes, a.activation_scales, codes, scales, weight.rows, 4096, columns};
            switch (route) {
            case ProjectionRoute::Mxfp8A8Mma8:
                launch_mxfp8_a8_mma<Native<NativeFp8MmaSchedule<64, 8, 128, 16, 8, 3>>>(
                    p, output, ProjectionTileEpilogue{}, stream);
                break;
            case ProjectionRoute::Mxfp8A8Mma8Deep:
                launch_mxfp8_a8_mma<Native<NativeFp8MmaSchedule<64, 8, 128, 16, 8, 4>>>(
                    p, output, ProjectionTileEpilogue{}, stream);
                break;
            case ProjectionRoute::Mxfp8A8Mma16:
                launch_mxfp8_a8_mma<Native<NativeFp8MmaSchedule<64, 16, 128, 32, 8, 3>>>(
                    p, output, ProjectionTileEpilogue{}, stream);
                break;
            case ProjectionRoute::Mxfp8A8Mma32:
                launch_mxfp8_a8_mma<Native<NativeFp8MmaSchedule<64, 32, 128, 32, 16, 2>>>(
                    p, output, ProjectionTileEpilogue{}, stream);
                break;
            case ProjectionRoute::Mxfp8A8Mma32Deep:
                launch_mxfp8_a8_mma<Native<NativeFp8MmaSchedule<64, 32, 128, 32, 16, 4>>>(
                    p, output, ProjectionTileEpilogue{}, stream);
                break;
            case ProjectionRoute::Mxfp8A8Tma64:
                launch_mxfp8_a8_tma<Native<Mxfp8A8TmaSchedule<64, 64, 128, 32, 16, 3>>>(
                    p, output, ProjectionTileEpilogue{}, stream);
                break;
            case ProjectionRoute::Mxfp8A8Tma128:
                launch_mxfp8_a8_tma<Native<Mxfp8A8TmaSchedule<128, 128, 128, 64, 32, 2>>>(
                    p, output, ProjectionTileEpilogue{}, stream);
                break;
            case ProjectionRoute::Mxfp8A8Tma128Wide:
                launch_mxfp8_a8_tma<Native<Mxfp8A8TmaSchedule<128, 128, 128, 64, 64, 2>>>(
                    p, output, ProjectionTileEpilogue{}, stream);
                break;
            default:
                throw std::logic_error("Kpool Prepare: invalid A8 projection route");
            }
        } else
            a16(Mxfp8A16Operands{x, codes, scales, weight.rows, 4096, columns});
    } else
        a16(Fp8BlockA16Operands{x, codes, static_cast<const float*>(weight.scales), weight.rows,
                                4096, columns});
}

using PairedDenseSimt = Static<
    Bf16A16SimtSchedule<8, 4, 1, 16, 4, 2, Bf16SimtActivationAccess::DirectStream,
                        Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 4, 1, 1, 1>>;

void launch_paired_projection(const PrepareArguments& a, std::int64_t begin, int columns,
                              PairedProjectionRoute route, cudaStream_t stream) {
    const int encoded_parent = a.weights[0].format == QType::BF16 ? 1 : 0;
    const int dense_parent   = 1 - encoded_parent;
    const auto& weight       = a.weights[encoded_parent];
    const auto& dense_weight = a.weights[dense_parent];
    const auto* x            = a.x + begin * 4096;
    const Bf16A16Operands dense{x, static_cast<const __nv_bfloat16*>(dense_weight.codes),
                                dense_weight.rows, 4096, columns};
    const ProjectionOutput encoded_out{a, begin, projection_row_map(a.mapping, encoded_parent)};
    const ProjectionOutput dense_out{a, begin, projection_row_map(a.mapping, dense_parent)};
    const auto submit = [&](const auto& encoded) {
        if (route == PairedProjectionRoute::Simt1)
            launch_native_fp8_bf16_pair<Native<NativeFp8A16SimtSchedule<1, 8, 1, 16>>,
                                        PairedDenseSimt, false>(encoded, dense, encoded_out,
                                                                dense_out, LinearIdentityEpilogue{},
                                                                LinearIdentityEpilogue{}, stream);
        else if (route == PairedProjectionRoute::Sliced16)
            launch_native_fp8_bf16_pair<Native<NativeFp8A16SlicedKSchedule<16, 16, 4, 128, 2>>,
                                        Static<Bf16A16SlicedKMmaSchedule<16, 16, 4, 64, 2>>, true>(
                encoded, dense, encoded_out, dense_out, LinearIdentityEpilogue{},
                LinearIdentityEpilogue{}, stream);
        else
            launch_native_fp8_bf16_pair<Native<NativeFp8A16SlicedKSchedule<16, 8, 8, 128, 1>>,
                                        Static<Bf16A16SlicedKMmaSchedule<16, 8, 8, 64, 1>>, true>(
                encoded, dense, encoded_out, dense_out, LinearIdentityEpilogue{},
                LinearIdentityEpilogue{}, stream);
    };
    const auto* codes = static_cast<const std::uint8_t*>(weight.codes);
    if (weight.format == QType::MXFP8)
        submit(Mxfp8A16Operands{x, codes, static_cast<const std::uint8_t*>(weight.scales),
                                weight.rows, 4096, columns});
    else
        submit(Fp8BlockA16Operands{x, codes, static_cast<const float*>(weight.scales), weight.rows,
                                   4096, columns});
}

void launch_bf16_projection(const PrepareArguments& a, int parent, std::int64_t begin, int columns,
                            ProjectionRoute route, cudaStream_t stream) {
    const auto& weight = a.weights[parent];
    const Bf16A16Operands p{a.x + begin * 4096, static_cast<const __nv_bfloat16*>(weight.codes),
                            weight.rows, 4096, columns};
    const auto mapping = projection_row_map(a.mapping, parent);
    const ProjectionOutput output{a, begin, mapping};
    const ProjectionPartialOutput partial{a.projected, a.rows, columns, mapping};
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
        if (weight.rows <= 288)
            launch_bf16_a16_mma<Medium>(p, output, ProjectionTileEpilogue{}, stream);
        else
            launch_bf16_a16_mma<Medium>(p, output, LinearIdentityEpilogue{}, stream);
        break;
    case ProjectionRoute::TmaSmall:
        launch_bf16_a16_tma_mma<TmaSmall>(p, output, ProjectionTileEpilogue{}, stream);
        break;
    case ProjectionRoute::TmaHistory:
        launch_bf16_a16_tma_mma<TmaHistory>(p, output, ProjectionTileEpilogue{}, stream);
        break;
    case ProjectionRoute::Tma128:
        launch_bf16_a16_tma_mma<TmaLarge>(p, output, ProjectionTileEpilogue{}, stream);
        break;
    case ProjectionRoute::Split32:
        launch_bf16_a16_tma_mma<SplitSmall, 4>(p, partial, ProjectionTileEpilogue{}, stream);
        break;
    case ProjectionRoute::Split64:
        launch_bf16_a16_tma_mma<SplitLarge, 4>(p, partial, ProjectionTileEpilogue{}, stream);
        break;
    default:
        throw std::logic_error("Kpool Prepare: invalid BF16 projection route");
    }
}
} // namespace

void launch_prepare(const PrepareArguments& inputs, const PrepareGeometry& geometry,
                    int window_columns, cudaStream_t stream) {
    auto a           = inputs;
    const auto total = std::int64_t(a.tokens) * a.batch;
    for (std::int64_t begin = 0; begin < total; begin += window_columns) {
        const int columns = static_cast<int>(std::min<std::int64_t>(window_columns, total - begin));
        const auto plan   = prepare_projection_plan(geometry, columns);
        a.splits          = plan.splits;
        set_window(a, begin, columns);
        if (plan.quantize)
            launch_mxfp8_a8_quantize(a.x + begin * 4096, 4096, columns,
                                     {a.activation_codes, a.activation_scales}, stream);
        if (plan.paired != PairedProjectionRoute::None) {
            launch_paired_projection(a, begin, columns, plan.paired, stream);
        } else {
            for (int parent = 0; parent < 2 && a.weights[parent].rows; ++parent) {
                if (a.weights[parent].format == QType::BF16)
                    launch_bf16_projection(a, parent, begin, columns,
                                           parent ? plan.second : plan.first, stream);
                else
                    launch_encoded_projection(a, parent, begin, columns,
                                              parent ? plan.second : plan.first, stream);
            }
        }
        int jobs = columns * (1 + (a.query_rank != 0));
        for (int b = a.first_row; b <= a.last_row; ++b) {
            const int start = b == a.first_row ? a.first_column : 0;
            const int end   = b == a.last_row ? a.last_stop : a.tokens;
            jobs += (end - start + 6) / 4;
        }
        const auto post = [&]<bool Fp8Latent>() {
            if (a.fp8_index)
                prepare_post_kernel<true, Fp8Latent><<<jobs, 128, 0, stream>>>(a, begin, columns);
            else
                prepare_post_kernel<false, Fp8Latent><<<jobs, 128, 0, stream>>>(a, begin, columns);
        };
        if (a.fp8_latent)
            post.template operator()<true>();
        else
            post.template operator()<false>();
        CUDA_CHECK(cudaGetLastError());
    }
}
} // namespace ninfer::ops::detail::kpool
