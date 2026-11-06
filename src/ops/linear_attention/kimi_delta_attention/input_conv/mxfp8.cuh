#pragma once
#include "ops/linear_attention/kimi_delta_attention/input_conv/tile_epilogue.cuh"
#include "ops/linear/kernels/fp8/native_a16.cuh"
#include "ops/linear/kernels/mxfp8/a8.cuh"
#include "ops/linear/kernels/mxfp8/a8_tma.cuh"
#include <algorithm>
#include <limits>

namespace ninfer::ops::detail::kda_input {
template <class Schedule>
struct ConvolutionSchedule : Schedule {
    static constexpr int kWarpGridTokens = Schedule::kWarpsTokens;
};

template <class A16Schedule, class A8Schedule, bool Tma, bool CacheColumns = false,
          bool Packed = false, bool FastSilu = false>
void launch_mxfp8_tiled(const Tensor& x, const Weight& w, LinearPolicy policy, const Arguments& a,
                        void* scratch, cudaStream_t stream) {
    const auto plan = mxfp8_workspace(a.width, a.batch, a.destination != nullptr, policy, true);
    const MemorySpan storage{scratch, plan.bytes, x.device};
    const auto convolution = plan.convolution.bind(storage);
    auto* edges            = static_cast<float*>(convolution.data);
    const auto layout      = tiled_workspace(a.width, a.batch, a.destination != nullptr);
    TiledConvolution output{
        a, edges,
        a.destination ? reinterpret_cast<__nv_bfloat16*>(edges + layout.edge_words) : nullptr};
    Mxfp8A8Workspace quantized{};
    if (policy == LinearPolicy::AllowA8) {
        quantized = {static_cast<std::uint8_t*>(plan.codes.bind(storage).data),
                     static_cast<std::uint8_t*>(plan.scales.bind(storage).data)};
    }
    const auto total_columns = std::int64_t(a.width) * a.batch;
    const Mxfp8A16Operands<> base{static_cast<const __nv_bfloat16*>(x.data),
                                  static_cast<const std::uint8_t*>(w.qdata),
                                  static_cast<const std::uint8_t*>(w.scales),
                                  w.n,
                                  w.k,
                                  0};
    // Keep the projection's token grid and quantizer group count representable without
    // changing request-relative convolution coordinates or the final publication boundary.
    constexpr int max_columns = 65535 * kTiledTokens;
    const auto launch         = [&]<class Column>() {
        using Epilogue =
            std::conditional_t<Packed, PackedConvolutionEpilogue<Column, false, FastSilu>,
                                       ConvolutionEpilogue<__nv_bfloat16, Column, false, CacheColumns>>;
        for (std::int64_t begin = 0; begin < total_columns;) {
            const int columns =
                static_cast<int>(std::min<std::int64_t>(max_columns, total_columns - begin));
            output.column_begin = begin;
            auto slice          = base;
            slice.x += std::int64_t(begin) * kInputFeatures;
            slice.tokens = columns;
            if (policy == LinearPolicy::AllowA8) {
                const Mxfp8A8Workspace packed{
                    quantized.codes + std::int64_t(begin) * kInputFeatures,
                    quantized.scales + std::int64_t(begin) * (kInputFeatures / 32)};
                launch_mxfp8_a8_quantize(slice.x, kInputFeatures, columns, packed, stream);
                if constexpr (Tma)
                    launch_mxfp8_a8_tma<A8Schedule>(mxfp8_a8_operands(w, packed, columns), output,
                                                            Epilogue{}, stream);
                else
                    launch_mxfp8_a8_mma<A8Schedule>(mxfp8_a8_operands(w, packed, columns), output,
                                                            Epilogue{}, stream);
            } else {
                launch_native_fp8_a16_mma<A16Schedule>(slice, output, Epilogue{}, stream);
            }
            begin += columns;
        }
        convolution_boundaries<kTiledTokens, __nv_bfloat16, Column>
            <<<kProjectionFeatures / 128, 128, 0, stream>>>(output);
        CUDA_CHECK(cudaGetLastError());
    };
    if (total_columns <= std::numeric_limits<int>::max())
        launch.template operator()<int>();
    else
        launch.template operator()<std::int64_t>();
}
} // namespace ninfer::ops::detail::kda_input
