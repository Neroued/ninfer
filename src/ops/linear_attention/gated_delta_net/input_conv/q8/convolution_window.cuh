#pragma once

#include "ops/linear_attention/gated_delta_net/input_conv/q8/launch.h"
#include "ops/linear_attention/gated_delta_net/input_conv/kernels/output.cuh"
#include "ops/linear/kernels/q8/q8_gemv_launch.cuh"
#include "ops/linear/kernels/q8/q8_sliced_k_launch.cuh"
#include <array>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
using ConvGeometry = GdnConvGeometry<6144>;

namespace gdn_q8_window {

template <class Geometry>
using Output = LinearBf16SegmentedOutput<Geometry::channels, Geometry::z_rows>;

// Each contraction tile owns complete temporal windows. Several short sequences may share
// its weight tile, but their histories and valid prefixes are consumed independently.
template <class ConvGeometry, int Width, class Publish>
struct WindowEpilogue {
    GdnConvEpilogue<Publish> conv;
    __nv_bfloat16* z;

    template <int Rows, int Columns, class Destination>
        requires(Columns > Width)
    __device__ __forceinline__ void apply_tile(const Destination&, int row0, int first_column,
                                               const float* projected, int stride, int live,
                                               int rows) const {
        const int sequences = live / Width;
        for (int task = threadIdx.x; task < Rows * sequences; task += blockDim.x) {
            const int local_row = task % Rows;
            if (local_row >= rows) continue;
            const int begin = task / Rows * Width;
            const int row   = row0 + local_row;
            if (row >= ConvGeometry::channels) {
#pragma unroll
                for (int t = 0; t < Width; ++t)
                    z[std::int64_t(first_column + begin + t) * ConvGeometry::z_rows + row -
                      ConvGeometry::channels] =
                        __float2bfloat16_rn(projected[local_row * stride + begin + t]);
            } else {
                float window[Width];
#pragma unroll
                for (int t = 0; t < Width; ++t)
                    window[t] = projected[local_row * stride + begin + t];
                auto sequence      = conv;
                sequence.batch_row = (first_column + begin) / Width;
                sequence.store(row, window);
            }
        }
    }

    template <class Destination, int Columns>
    __device__ __forceinline__ void apply_row(const Destination&, int row, int first_column,
                                              const float (&projected)[Columns], int live) const {
        if (row >= ConvGeometry::channels) {
#pragma unroll
            for (int column = 0; column < Columns; ++column)
                if (column < live)
                    z[static_cast<std::int64_t>(first_column + column) * ConvGeometry::z_rows +
                      row - ConvGeometry::channels] = __float2bfloat16_rn(projected[column]);
            return;
        }
        for (int begin = 0; begin + Width <= live; begin += Width) {
            float window[Width];
#pragma unroll
            for (int token = 0; token < Width; ++token) window[token] = projected[begin + token];
            auto sequence      = conv;
            sequence.batch_row = (first_column + begin) / Width;
            sequence.store(row, window);
        }
    }
};

template <class ConvGeometry, int Width, int TileColumns, int Columns, int KWarps, class Publish>
void launch_sliced(const Q8LinearOperands& operands, Output<ConvGeometry> output,
                   WindowEpilogue<ConvGeometry, Width, Publish> epilogue, cudaStream_t stream) {
    using Schedule =
        Q8A16SlicedKMmaSchedule<TileColumns, KWarps, 1, 1, Q8ScaleAccess::Shared, Cache::ca,
                                Cache::cg, Q8ActivationStage::PaddedZero, 2560, Columns>;
    launch_q8_a16_sliced_k_mma<Schedule>(operands, output, epilogue, stream);
}

template <class ConvGeometry, class Gemv, int Width, int KWarps, int AggregateColumns,
          class Publish>
void launch_window(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                   const ConstStateFieldView& states, const Tensor& valid, Tensor& query,
                   Tensor& key, Tensor& value, Tensor& z, Publish publish, bool aggregate,
                   cudaStream_t stream) {
    const int columns   = Width * x.ne[2];
    const auto flat     = x.view({2560, columns});
    const auto operands = q8_linear_operands(flat, weight);
    const Output<ConvGeometry> output{static_cast<__nv_bfloat16*>(query.data),
                                      static_cast<__nv_bfloat16*>(z.data)};
    const auto adapter = make_gdn_conv_output<ConvGeometry, Width>(conv_weight, states, valid,
                                                                   query, key, value, z, publish);
    const WindowEpilogue<ConvGeometry, Width, Publish> epilogue{adapter.conv, adapter.z};
    if constexpr (Width == 1) {
        if (columns == 1) {
            launch_q8_a16_gemv<Gemv>(operands, output, epilogue, stream);
            return;
        }
    }
    if constexpr (Width <= 8) {
        if (aggregate) {
            if constexpr (Width <= 4) {
                if (columns <= 8) {
                    launch_sliced<ConvGeometry, Width, 8, 8, KWarps>(operands, output, epilogue,
                                                                     stream);
                    return;
                }
            }
            if constexpr (Width >= 2) {
                if (columns <= 16)
                    launch_sliced<ConvGeometry, Width, 16, 16, KWarps>(operands, output, epilogue,
                                                                       stream);
                else if constexpr (AggregateColumns > 16)
                    launch_sliced<ConvGeometry, Width, AggregateColumns, AggregateColumns, KWarps>(
                        operands, output, epilogue, stream);
            }
            return;
        }
    }
    if constexpr (Width >= 2)
        launch_sliced<ConvGeometry, Width, Width <= 8 ? 8 : 16, Width, KWarps>(operands, output,
                                                                               epilogue, stream);
}

template <class ConvGeometry, class Gemv, int KWarps, int AggregateColumns, class Publish,
          std::size_t... I>
constexpr auto window_launchers(std::index_sequence<I...>) {
    return std::array{&launch_window<ConvGeometry, Gemv, static_cast<int>(I) + 1, KWarps,
                                     AggregateColumns, Publish>...};
}

template <class ConvGeometry, class Gemv, int KWarps = 4, int AggregateColumns = 16, class Publish>
void dispatch(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
              const ConstStateFieldView& states, const Tensor& valid, Tensor& query, Tensor& key,
              Tensor& value, Tensor& z, Publish publish, bool aggregate, cudaStream_t stream) {
    if (x.ne[1] < 1 || x.ne[1] > 16 || x.ne[2] < 1 || x.ne[2] > 8 ||
        (aggregate && (x.ne[1] > 8 || x.ne[1] * x.ne[2] > AggregateColumns)) ||
        (!aggregate && x.ne[2] != 1))
        throw std::invalid_argument("Q8 fused convolution: invalid temporal tile");
    static constexpr auto launches =
        window_launchers<ConvGeometry, Gemv, KWarps, AggregateColumns, Publish>(
            std::make_index_sequence<16>{});
    launches[x.ne[1] - 1](x, weight, conv_weight, states, valid, query, key, value, z, publish,
                          aggregate, stream);
}

} // namespace gdn_q8_window

} // namespace ninfer::ops::detail
