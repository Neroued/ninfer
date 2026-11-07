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

template <class S>
struct RegisterWeights : Native<S> {
    static constexpr bool kRegisterWeights = true;
};

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
            const auto tma = [&]<class S>() {
                launch_mxfp8_a8_tma<Native<S>>(p, output, ProjectionTileEpilogue{}, stream);
            };
            if (route == ProjectionRoute::Mxfp8A8Short) {
                if (weight.rows == 2176 && columns > 48 && columns <= 64)
                    launch_mxfp8_a8_mma<Native<NativeFp8MmaSchedule<64, 16, 128, 32, 8, 3>>>(
                        p, output, ProjectionTileEpilogue{}, stream);
                else if (weight.rows <= 768)
                    tma.template
                    operator()<Mxfp8A8TmaSchedule<32, 32, 256, 16, 16, 2, 1, 2, false, 2>>();
                else
                    tma.template operator()<Mxfp8A8TmaSchedule<48, 64, 256, 16, 32, 2, 1, 2>>();
            } else if (route == ProjectionRoute::Mxfp8A8Medium) {
                if (weight.rows <= 768)
                    tma.template operator()<Mxfp8A8TmaSchedule<64, 64, 128, 32, 16, 3>>();
                else if (weight.rows == 2176)
                    tma.template operator()<Mxfp8A8TmaSchedule<128, 128, 128, 64, 32, 2>>();
                else
                    tma.template operator()<Mxfp8A8TmaSchedule<128, 128, 128, 64, 64, 2>>();
            } else if (weight.rows <= 768) {
                if (columns <= 512)
                    tma.template operator()<Mxfp8A8TmaSchedule<64, 128, 128, 32, 32, 3>>();
                else
                    tma.template
                    operator()<Mxfp8A8TmaSchedule<128, 128, 128, 64, 32, 2, 1, 2, true, 2>>();
            } else
                tma.template
                operator()<Mxfp8A8TmaSchedule<192, 128, 128, 48, 64, 2, 1, 2, true, 1>>();
        } else
            a16(Mxfp8A16Operands{x, codes, scales, weight.rows, 4096, columns});
    } else {
        const Fp8BlockA16Operands p{x,           codes, static_cast<const float*>(weight.scales),
                                    weight.rows, 4096,  columns};
        const auto mma = [&]<class S>() {
            launch_native_fp8_a16_mma<RegisterWeights<S>>(p, output, ProjectionTileEpilogue{},
                                                          stream);
        };
        if (columns <= 32)
            a16(p);
        else if (weight.rows <= 768) {
            if (columns <= 96)
                mma.template operator()<NativeFp8MmaSchedule<32, 16, 128, 16, 8, 2, 2, 2>>();
            else if (columns <= 128)
                mma.template operator()<NativeFp8MmaSchedule<16, 32, 128, 16, 16, 2, 2, 2>>();
            else if (columns <= 192)
                mma.template operator()<NativeFp8MmaSchedule<32, 32, 128, 16, 16, 2, 2, 2>>();
            else
                mma.template operator()<NativeFp8MmaSchedule<64, 64, 128, 32, 32, 2, 2, 8>>();
        } else {
            if (columns <= 64)
                mma.template operator()<NativeFp8MmaSchedule<48, 32, 128, 16, 16, 2, 2, 2>>();
            else if (columns <= 128)
                mma.template operator()<NativeFp8MmaSchedule<96, 64, 128, 32, 16, 2, 1, 2>>();
            else if (columns <= 192)
                mma.template operator()<NativeFp8MmaSchedule<128, 64, 128, 32, 32, 2, 1, 2>>();
            else if (columns <= 256 || (columns > 384 && columns <= 640) ||
                     (columns > 768 && columns <= 896) || (columns > 960 && columns <= 1024))
                mma.template operator()<NativeFp8MmaSchedule<64, 64, 128, 32, 32, 2, 2>>();
            else
                mma.template operator()<NativeFp8MmaSchedule<128, 64, 128, 32, 32, 1, 2>>();
        }
    }
}

using PairedDenseSimt = Static<
    Bf16A16SimtSchedule<8, 4, 1, 16, 4, 2, Bf16SimtActivationAccess::DirectStream,
                        Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 4, 1, 1, 1>>;

template <class E, class D, bool Sliced, class Encoded, class Output>
void paired_projection(const Encoded& encoded, const Bf16A16Operands& dense, Output encoded_out,
                       Output dense_out, LinearIdentityEpilogue, LinearIdentityEpilogue,
                       cudaStream_t stream, bool interleave, bool vector_rows) {
    const auto submit = [&](auto encoded_epilogue) {
        if (interleave)
            launch_native_fp8_bf16_pair<E, D, Sliced, true>(encoded, dense, encoded_out, dense_out,
                                                            encoded_epilogue,
                                                            LinearIdentityEpilogue{}, stream);
        else
            launch_native_fp8_bf16_pair<E, D, Sliced, false>(encoded, dense, encoded_out, dense_out,
                                                             encoded_epilogue,
                                                             LinearIdentityEpilogue{}, stream);
    };
    if constexpr (Sliced)
        if (vector_rows) return submit(ProjectionRowEpilogue{});
    submit(LinearIdentityEpilogue{});
}

void launch_paired_projection(const PrepareArguments& a, std::int64_t begin, int columns,
                              PairedProjectionRoute route, bool skip_empty, cudaStream_t stream) {
    const int encoded_parent = a.weights[0].format == QType::BF16 ? 1 : 0;
    const int dense_parent   = 1 - encoded_parent;
    const auto& weight       = a.weights[encoded_parent];
    const auto& dense_weight = a.weights[dense_parent];
    const auto* x            = a.x + begin * 4096;
    const Bf16A16Operands dense{x, static_cast<const __nv_bfloat16*>(dense_weight.codes),
                                dense_weight.rows, 4096, columns};
    const ProjectionOutput encoded_out{a, begin, projection_row_map(a.mapping, encoded_parent)};
    const ProjectionOutput dense_out{a, begin, projection_row_map(a.mapping, dense_parent)};
    const bool interleave =
        columns > 1 &&
        (a.query_rank ? columns <= 32 : weight.format != QType::MXFP8 && columns > 32);
    const bool query_history_tail =
        weight.format == QType::MXFP8 && a.query_rank && !a.heads && columns > 32 && columns <= 48;
    const bool vector_rows =
        (columns <= 32 || query_history_tail) && (weight.format != QType::MXFP8 || a.query_rank) &&
        (columns > 8 || (columns > 1 && weight.format != QType::MXFP8 && a.query_rank));
    const auto submit = [&](const auto& encoded) {
        const auto project = [&](const auto& encoded_out, const auto& dense_out) {
            if (route == PairedProjectionRoute::Simt1)
                paired_projection<Native<NativeFp8A16SimtSchedule<1, 8, 1, 16>>, PairedDenseSimt,
                                  false>(encoded, dense, encoded_out, dense_out,
                                         LinearIdentityEpilogue{}, LinearIdentityEpilogue{}, stream,
                                         interleave, vector_rows);
            else if (query_history_tail)
                paired_projection<Native<NativeFp8A16SlicedKSchedule<16, 16, 4, 128, 1>>,
                                  Static<Bf16A16SlicedKMmaSchedule<16, 16, 4, 64, 1>>, true>(
                    encoded, dense, encoded_out, dense_out, LinearIdentityEpilogue{},
                    LinearIdentityEpilogue{}, stream, interleave, vector_rows);
            else if (weight.format == QType::MXFP8 && a.query_rank && columns > 8 && columns <= 16)
                paired_projection<Native<NativeFp8A16SlicedKSchedule<16, 16, 8, 128, 1>>,
                                  Static<Bf16A16SlicedKMmaSchedule<16, 16, 8, 64, 1>>, true>(
                    encoded, dense, encoded_out, dense_out, LinearIdentityEpilogue{},
                    LinearIdentityEpilogue{}, stream, interleave, vector_rows);
            else if (weight.format != QType::MXFP8 && columns > 8 && columns <= 24)
                paired_projection<Native<NativeFp8A16SlicedKSchedule<16, 8, 8, 128, 1>>,
                                  Static<Bf16A16SlicedKMmaSchedule<16, 8, 8, 64, 1>>, true>(
                    encoded, dense, encoded_out, dense_out, LinearIdentityEpilogue{},
                    LinearIdentityEpilogue{}, stream, interleave, vector_rows);
            else if (weight.format == QType::MXFP8 && a.query_rank && columns > 16 && columns <= 24)
                paired_projection<Native<NativeFp8A16SlicedKSchedule<16, 8, 4, 128, 1>>,
                                  Static<Bf16A16SlicedKMmaSchedule<16, 8, 4, 64, 1>>, true>(
                    encoded, dense, encoded_out, dense_out, LinearIdentityEpilogue{},
                    LinearIdentityEpilogue{}, stream, interleave, vector_rows);
            else if (route == PairedProjectionRoute::Sliced16)
                paired_projection<Native<NativeFp8A16SlicedKSchedule<16, 16, 4, 128, 2>>,
                                  Static<Bf16A16SlicedKMmaSchedule<16, 16, 4, 64, 2>>, true>(
                    encoded, dense, encoded_out, dense_out, LinearIdentityEpilogue{},
                    LinearIdentityEpilogue{}, stream, interleave, vector_rows);
            else if (weight.format != QType::MXFP8 && weight.rows <= 768)
                paired_projection<Native<NativeFp8A16SlicedKSchedule<16, 8, 4, 128, 2>>,
                                  Static<Bf16A16SlicedKMmaSchedule<16, 8, 4, 64, 2>>, true>(
                    encoded, dense, encoded_out, dense_out, LinearIdentityEpilogue{},
                    LinearIdentityEpilogue{}, stream, interleave, vector_rows);
            else
                paired_projection<Native<NativeFp8A16SlicedKSchedule<16, 8, 8, 128, 1>>,
                                  Static<Bf16A16SlicedKMmaSchedule<16, 8, 8, 64, 1>>, true>(
                    encoded, dense, encoded_out, dense_out, LinearIdentityEpilogue{},
                    LinearIdentityEpilogue{}, stream, interleave, vector_rows);
        };
        if (!skip_empty) {
            project(UnconditionalProjectionOutput{encoded_out},
                    UnconditionalProjectionOutput{dense_out});
            return;
        }
        project(encoded_out, dense_out);
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
        if (weight.rows == 128) {
            // The narrow gate parent needs more independent CTAs than the wide projection.
            if (columns <= 768)
                launch_bf16_a16_tma_mma<Static<
                    Bf16A16TmaMmaSchedule<32, 32, 128, 16, 16, 3, 2, Bf16MmaRaster::RowFast>>>(
                    p, output, ProjectionTileEpilogue{}, stream);
            else
                launch_bf16_a16_tma_mma<Static<
                    Bf16A16TmaMmaSchedule<32, 32, 64, 16, 16, 3, 1, Bf16MmaRaster::RowFast>>>(
                    p, output, ProjectionTileEpilogue{}, stream);
        } else
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
            launch_paired_projection(a, begin, columns, plan.paired, plan.skip_empty, stream);
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
            const auto submit = [&]<bool Fp8Index>() {
                if (a.update && columns <= 64)
                    prepare_post_kernel<Fp8Index, Fp8Latent, true>
                        <<<jobs, 256, 0, stream>>>(a, begin, columns);
                else
                    prepare_post_kernel<Fp8Index, Fp8Latent>
                        <<<jobs, 128, 0, stream>>>(a, begin, columns);
            };
            if (a.fp8_index)
                submit.template operator()<true>();
            else
                submit.template operator()<false>();
        };
        if (a.fp8_latent)
            post.template operator()<true>();
        else
            post.template operator()<false>();
        CUDA_CHECK(cudaGetLastError());
    }
}
} // namespace ninfer::ops::detail::kpool
