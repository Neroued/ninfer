#pragma once
#include "ops/linear_attention/kimi_delta_attention/input_conv/tile_epilogue.cuh"
#include "ops/linear/kernels/fp8/native_a16.cuh"
#include "ops/linear/kernels/mxfp8/a8.cuh"
#include "ops/linear/kernels/mxfp8/a8_tma.cuh"
#include <algorithm>
#include <limits>
#include <type_traits>

namespace ninfer::ops::detail::kda_input {
template <class Schedule>
struct ConvolutionSchedule : Schedule {
    static constexpr int kWarpGridTokens = Schedule::kWarpsTokens;
};

template <class Schedule, LinearPolicy Policy, class Epilogue, class Column, bool WholeWindows,
          bool Ragged, bool Tma = false>
void launch_mxfp8_tile_range(const Tensor& x, const Weight& w, const Arguments& a, void* scratch,
                             cudaStream_t stream) {
    static_assert(!Ragged || (!WholeWindows && std::is_same_v<Column, int>));
    const auto plan =
        Ragged ? ragged_mxfp8_workspace(a.width, a.batch, Policy)
               : mxfp8_workspace(a.width, a.batch, a.destination != nullptr, Policy, !WholeWindows);
    const MemorySpan storage{scratch, plan.bytes, x.device};
    auto* edges = WholeWindows ? nullptr : static_cast<float*>(plan.convolution.bind(storage).data);
    const auto layout = Ragged ? ragged_tiled_workspace(a.width, a.batch)
                               : tiled_workspace(a.width, a.batch, a.destination != nullptr);
    TiledConvolution output{a, edges,
                            a.destination && !WholeWindows
                                ? reinterpret_cast<__nv_bfloat16*>(edges + layout.edge_words)
                                : nullptr};
    Mxfp8A8Workspace quantized{};
    if constexpr (Policy == LinearPolicy::AllowA8) {
        quantized = {static_cast<std::uint8_t*>(plan.codes.bind(storage).data),
                     static_cast<std::uint8_t*>(plan.scales.bind(storage).data)};
    }
    const auto total_columns = Ragged ? a.width : std::int64_t(a.width) * a.batch;
    const auto* activation   = static_cast<const __nv_bfloat16*>(x.data);
    // Keep the projection's token grid and quantizer group count representable without
    // changing request-relative convolution coordinates or the final publication boundary.
    constexpr int max_columns = 65535 * kTiledTokens;
    for (std::int64_t begin = 0; begin < total_columns;) {
        const int columns =
            static_cast<int>(std::min<std::int64_t>(max_columns, total_columns - begin));
        output.column_begin = begin;
        const auto* slice   = activation + begin * kInputFeatures;
        if constexpr (Policy == LinearPolicy::AllowA8) {
            const Mxfp8A8Workspace packed{quantized.codes + begin * kInputFeatures,
                                          quantized.scales + begin * (kInputFeatures / 32)};
            launch_mxfp8_a8_quantize(slice, kInputFeatures, columns, packed, stream);
            if constexpr (Tma)
                launch_mxfp8_a8_tma<Schedule>(mxfp8_a8_operands(w, packed, columns), output,
                                              Epilogue{}, stream);
            else
                launch_mxfp8_a8_mma<Schedule>(mxfp8_a8_operands(w, packed, columns), output,
                                              Epilogue{}, stream);
        } else {
            const Mxfp8A16Operands<> operands{slice,
                                              static_cast<const std::uint8_t*>(w.qdata),
                                              static_cast<const std::uint8_t*>(w.scales),
                                              w.n,
                                              w.k,
                                              columns};
            launch_native_fp8_a16_mma<Schedule>(operands, output, Epilogue{}, stream);
        }
        begin += columns;
    }
    if constexpr (!WholeWindows) {
        convolution_boundaries<kTiledTokens, __nv_bfloat16, Column, Ragged>
            <<<kProjectionFeatures / 128, 128, 0, stream>>>(output);
        CUDA_CHECK(cudaGetLastError());
    }
}

template <class A16Schedule, class A8Schedule, bool Tma, bool CacheColumns = false,
          bool Packed = false, bool FastSilu = false>
void launch_mxfp8_tiled(const Tensor& x, const Weight& w, LinearPolicy policy, const Arguments& a,
                        void* scratch, cudaStream_t stream) {
    const auto launch = [&]<class Column, bool WholeWindows, bool Ragged = false>() {
        using Epilogue = std::conditional_t<
            Packed, PackedConvolutionEpilogue<Column, false, FastSilu, WholeWindows, Ragged>,
            ConvolutionEpilogue<__nv_bfloat16, Column, false, CacheColumns || Ragged, WholeWindows,
                                Ragged>>;
        if (policy == LinearPolicy::AllowA8)
            launch_mxfp8_tile_range<A8Schedule, LinearPolicy::AllowA8, Epilogue, Column,
                                    WholeWindows, Ragged, Tma>(x, w, a, scratch, stream);
        else
            launch_mxfp8_tile_range<A16Schedule, LinearPolicy::A16Only, Epilogue, Column,
                                    WholeWindows, Ragged>(x, w, a, scratch, stream);
    };
    const auto columns = [&]<class Column>() {
        if (whole_window_tiles(a.width))
            launch.template operator()<Column, true>();
        else
            launch.template operator()<Column, false>();
    };
    if (a.token_to_seq)
        launch.template operator()<int, false, true>();
    else if (std::int64_t(a.width) * a.batch <= std::numeric_limits<int>::max())
        columns.template operator()<int>();
    else
        columns.template operator()<std::int64_t>();
}
} // namespace ninfer::ops::detail::kda_input
