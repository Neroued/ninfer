#include "ops/linear_attention/kimi_delta_attention/input_conv/kernels.cuh"
#include "ops/linear_attention/kimi_delta_attention/input_conv/tma.cuh"
#include "ops/linear/kernels/q8/q8_gemv_launch.cuh"
#include "ops/linear/kernels/q8/q8_simt_launch.cuh"
#include "ops/linear/kernels/q8/q8_sliced_k_launch.cuh"

namespace ninfer::ops::detail::kda_input {
namespace {
template <int Width>
void window(const Q8LinearOperands& operands, const Arguments& a, cudaStream_t stream) {
    const LinearBf16Output unused{a.q, kInputFeatures};
    const WindowEpilogue<Width> epilogue{a};
    if constexpr (Width == 1) {
        if (a.batch == 1) {
            launch_q8_a16_gemv<Q8A16GemvSchedule<4, 1, 2, 4096, 8>>(operands, unused, epilogue,
                                                                    stream);
            return;
        }
    }
    if constexpr (Width <= 2) {
        if (operands.tokens == 2) {
            launch_q8_a16_simt<Q8A16SimtSchedule<4, 4, 2, 16, 2, Cache::cg, 2>>(operands, unused,
                                                                                epilogue, stream);
            return;
        }
    }
    if constexpr (Width <= 8 && 8 % Width == 0) {
        if (operands.tokens == 8 && a.batch <= 2) {
            launch_q8_a16_simt<Q8A16SimtSchedule<8, 8, 1, 32, 2, Cache::cg, 1>>(operands, unused,
                                                                                epilogue, stream);
            return;
        }
    }
    if (a.batch >= 4 && Width <= 2) {
        using S =
            Q8A16SlicedKMmaSchedule<16, 4, 1, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                    Q8ActivationStage::PaddedZero, 4096, (16 / Width) * Width>;
        launch_q8_a16_sliced_k_mma<S>(operands, unused, epilogue, stream);
    } else {
        using S =
            Q8A16SlicedKMmaSchedule<32, 4, 2, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                    Q8ActivationStage::PaddedZero, 4096, (32 / Width) * Width>;
        launch_q8_a16_sliced_k_mma<S>(operands, unused, epilogue, stream);
    }
}

template <int Raster, bool K32, bool Prefetch = false>
void tiled(const Q8LinearOperands& operands, TiledConvolution output, cudaStream_t stream) {
    if constexpr (K32) {
        using S = std::conditional_t<Prefetch, Q8A16TmaK32PrefetchSchedule<Raster, 4096>,
                                     Q8A16TmaK32Schedule<Raster, 4096>>;
        launch_q8_a16_tma_k32<S>(operands, output, ConvolutionEpilogue{}, stream);
    } else {
        using S = Q8A16TmaMmaSchedule<64, 128, 64, 16, 2, Raster, 4096>;
        launch_q8_a16_tma_mma<S>(operands, output, ConvolutionEpilogue{}, stream);
    }
}
} // namespace

void launch(const Tensor& x, const Weight& w, const Arguments& a, float* scratch,
            cudaStream_t stream) {
    const auto operands = q8_linear_operands(x.view({kInputFeatures, a.width * a.batch}), w);
    using Window        = void (*)(const Q8LinearOperands&, const Arguments&, cudaStream_t);
    static constexpr Window windows[] = {window<1>, window<2>, window<3>, window<4>,
                                         window<5>, window<6>, window<7>, window<8>};
    if (!scratch) {
        windows[a.width - 1](operands, a, stream);
        return;
    }
    if (operands.tokens <= 32) {
        using S = Q8A16SlicedKMmaSchedule<32, 4, 1, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                          Q8ActivationStage::PaddedZero, 4096>;
        launch_q8_a16_sliced_k_mma<S>(operands, FloatOutput{scratch}, LinearIdentityEpilogue{},
                                      stream);
        conv_kernel<<<dim3(kProjectionFeatures / 128, (a.width + 3) / 4, a.batch), 128, 0,
                      stream>>>(a, scratch);
        if (a.destination)
            history_kernel<<<dim3(kProjectionFeatures / 128, a.batch), 128, 0, stream>>>(a,
                                                                                         scratch);
    } else {
        const auto layout = tiled_workspace(a.width, a.batch, a.destination != nullptr);
        TiledConvolution output{a, scratch,
                                a.destination
                                    ? reinterpret_cast<__nv_bfloat16*>(scratch + layout.edge_words)
                                    : nullptr};
        if (operands.tokens <= 128)
            tiled<4, true, true>(operands, output, stream);
        else if (operands.tokens <= 512)
            tiled<8, false>(operands, output, stream);
        else if (operands.tokens <= 2048)
            tiled<4, true>(operands, output, stream);
        else
            tiled<16, true>(operands, output, stream);
        convolution_boundaries<kTiledTokens><<<kProjectionFeatures / 128, 128, 0, stream>>>(output);
    }
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops::detail::kda_input
