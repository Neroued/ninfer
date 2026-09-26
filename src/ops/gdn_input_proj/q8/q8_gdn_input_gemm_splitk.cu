#include "ops/linear/q8/q8_geometry.h"
#include "ops/linear/q8/q8_grouped_sliced_k_launch.cuh"
#include "ops/linear/q8/q8_instances.cuh"
#include "core/weight.h"
#include "ops/gdn_input_proj/q8/q8_gdn_input_kernels.h"

#include "core/device.h"
#include "ops/common/mma.cuh"
#include "ops/common/memory.cuh"
#include "ops/gdn_input_proj/gdn_conv.cuh"
#include "ops/linear/q8/q8_sliced_k_launch.cuh"
#include "ops/linear/common/output.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

constexpr int kRows   = 12288;
constexpr int kHidden = 2048;

constexpr int kRowsPerCta              = 16;
constexpr int kFirstExactCols          = 2;
constexpr int kLastProjectionExactCols = 32;
constexpr int kLastSnapshotExactCols   = 16;
using Output                           = LinearBf16SegmentedOutput<8192, 4096>;

template <class Publish>
struct Q8GdnSplitKConvEpilogue {
    GdnConvEpilogue<Publish> conv;
    __nv_bfloat16* z;

    template <class Output, int ActiveCols>
    __device__ __forceinline__ void apply_row(const Output&, int row, int,
                                              const float (&projected)[ActiveCols], int) const {
        if (row < 8192) {
            conv.store(row, projected);
        } else {
#pragma unroll
            for (int token = 0; token < ActiveCols; ++token) {
                z[static_cast<std::int64_t>(token) * 4096 + row - 8192] =
                    __float2bfloat16_rn(projected[token]);
            }
        }
    }
};

template <int ActiveCols>
void launch_active_cols(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                        cudaStream_t stream) {
    constexpr int TileCols =
        ActiveCols <= 8 ? 8 : (ActiveCols <= 16 ? 16 : (ActiveCols <= 24 ? 24 : 32));
    using Geometry = Q8LinearGeometry<kRows, kHidden>;
    using Schedule = Q8SlicedKDefault<TileCols, ActiveCols>;
    static_assert((8192 % kRowsPerCta) == 0 && (4096 % kRowsPerCta) == 0);
    const Output output{static_cast<__nv_bfloat16*>(qkv.data), static_cast<__nv_bfloat16*>(z.data)};
    launch_q8_a16_sliced_k_mma<
        typename Schedule::template with_problem<Geometry::kInputRows, ActiveCols, true>>(
        q8_linear_operands(x, weight), output, LinearIdentityEpilogue{}, stream);
}

template <int ActiveCols, class Publish>
void launch_active_cols_conv(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                             const Tensor& conv_states, const Tensor& valid_columns,
                             const Tensor& initial_slot, Tensor& query, Tensor& key, Tensor& value,
                             Tensor& z, Publish publish, cudaStream_t stream) {
    static_assert(ActiveCols >= 2 && ActiveCols <= 16);
    constexpr int TileCols = ActiveCols <= 8 ? 8 : 16;
    using Geometry         = Q8LinearGeometry<kRows, kHidden>;
    using Schedule         = Q8SlicedKDefault<TileCols, ActiveCols>;
    const Output ignored_output{static_cast<__nv_bfloat16*>(query.data),
                                static_cast<__nv_bfloat16*>(z.data)};
    const Q8GdnSplitKConvEpilogue<Publish> epilogue{
        {
            static_cast<const __nv_bfloat16*>(conv_weight.data),
            static_cast<const __nv_bfloat16*>(conv_states.data),
            static_cast<const std::int32_t*>(initial_slot.data),
            valid_columns.data == nullptr ? nullptr
                                          : static_cast<const std::int32_t*>(valid_columns.data),
            static_cast<__nv_bfloat16*>(query.data),
            static_cast<__nv_bfloat16*>(key.data),
            static_cast<__nv_bfloat16*>(value.data),
            8192,
            2048,
            2048,
            4096,
            0,
            ActiveCols,
            0,
            publish,
        },
        static_cast<__nv_bfloat16*>(z.data),
    };
    launch_q8_a16_sliced_k_mma<
        typename Schedule::template with_problem<Geometry::kInputRows, ActiveCols, true>>(
        q8_linear_operands(x, weight), ignored_output, epilogue, stream);
}

template <int ActiveCols>
void launch_active_cols_conv_snapshot(const Tensor& x, const Weight& weight,
                                      const Tensor& conv_weight, Tensor& conv_states,
                                      const Tensor& valid_columns, const Tensor& initial_slot,
                                      const Tensor& snapshot_base_slot, Tensor& query, Tensor& key,
                                      Tensor& value, Tensor& z, cudaStream_t stream) {
    launch_active_cols_conv<ActiveCols>(
        x, weight, conv_weight, conv_states, valid_columns, initial_slot, query, key, value, z,
        SnapshotHistoryPublish{static_cast<__nv_bfloat16*>(conv_states.data),
                               static_cast<const std::int32_t*>(snapshot_base_slot.data), 8192},
        stream);
}

template <int ActiveCols>
void launch_active_cols_conv_record(const Tensor& x, const Weight& weight,
                                    const Tensor& conv_weight, const Tensor& conv_states,
                                    const Tensor& valid_columns, const Tensor& initial_slot,
                                    Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value,
                                    Tensor& z, cudaStream_t stream) {
    launch_active_cols_conv<ActiveCols>(
        x, weight, conv_weight, conv_states, valid_columns, initial_slot, query, key, value, z,
        RecordColumnPublish{static_cast<__nv_bfloat16*>(conv_record.data), 8192, ActiveCols},
        stream);
}

template <int TileCols, int KSplits, int NGroups, int MinBlocks>
void launch_medium_cols(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                        cudaStream_t stream) {
    static_assert((8192 % kRowsPerCta) == 0 && (4096 % kRowsPerCta) == 0);
    const Output output{static_cast<__nv_bfloat16*>(qkv.data), static_cast<__nv_bfloat16*>(z.data)};
    using Schedule =
        Q8A16GroupedSlicedKMmaSchedule<TileCols, KSplits, NGroups, 1, MinBlocks, kHidden>;
    launch_q8_a16_grouped_sliced_k_mma<Schedule>(q8_linear_operands(x, weight), output,
                                                 LinearIdentityEpilogue{}, stream);
}

using ProjectionLauncher = void (*)(const Tensor&, const Weight&, Tensor&, Tensor&, cudaStream_t);
using SnapshotLauncher   = void (*)(const Tensor&, const Weight&, const Tensor&, Tensor&,
                                  const Tensor&, const Tensor&, const Tensor&, Tensor&, Tensor&,
                                  Tensor&, Tensor&, cudaStream_t);
using RecordLauncher     = void (*)(const Tensor&, const Weight&, const Tensor&, const Tensor&,
                                const Tensor&, const Tensor&, Tensor&, Tensor&, Tensor&, Tensor&,
                                Tensor&, cudaStream_t);

template <std::size_t... Offsets>
constexpr auto make_projection_launchers(std::index_sequence<Offsets...>) {
    return std::array<ProjectionLauncher, sizeof...(Offsets)>{
        &launch_active_cols<kFirstExactCols + static_cast<int>(Offsets)>...};
}

template <std::size_t... Offsets>
constexpr auto make_snapshot_launchers(std::index_sequence<Offsets...>) {
    return std::array<SnapshotLauncher, sizeof...(Offsets)>{
        &launch_active_cols_conv_snapshot<kFirstExactCols + static_cast<int>(Offsets)>...};
}

template <std::size_t... Offsets>
constexpr auto make_record_launchers(std::index_sequence<Offsets...>) {
    return std::array<RecordLauncher, sizeof...(Offsets)>{
        &launch_active_cols_conv_record<kFirstExactCols + static_cast<int>(Offsets)>...};
}

constexpr auto kProjectionLaunchers = make_projection_launchers(
    std::make_index_sequence<kLastProjectionExactCols - kFirstExactCols + 1>{});
constexpr auto kSnapshotLaunchers = make_snapshot_launchers(
    std::make_index_sequence<kLastSnapshotExactCols - kFirstExactCols + 1>{});
constexpr auto kRecordLaunchers =
    make_record_launchers(std::make_index_sequence<kLastSnapshotExactCols - kFirstExactCols + 1>{});

} // namespace

void q8_gdn_input_splitk_mma_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                    cudaStream_t stream) {
    const std::int32_t cols = x.ne[1];
    if (cols < kFirstExactCols || cols > 96) {
        throw std::invalid_argument("Q8 GDN split-K MMA requires T=2..96");
    }
    if (cols <= kLastProjectionExactCols) {
        kProjectionLaunchers[cols - kFirstExactCols](x, weight, qkv, z, stream);
    } else if (cols <= 48) {
        launch_medium_cols<48, 4, 2, 3>(x, weight, qkv, z, stream);
    } else if (cols <= 64) {
        launch_medium_cols<64, 4, 2, 2>(x, weight, qkv, z, stream);
    } else {
        launch_medium_cols<96, 2, 4, 3>(x, weight, qkv, z, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

void q8_gdn_input_splitk_conv_snapshot_launch(
    const Tensor& x, const Weight& weight, const Tensor& conv_weight, Tensor& conv_states,
    const Tensor& valid_columns, const Tensor& initial_slot, const Tensor& snapshot_base_slot,
    Tensor& query, Tensor& key, Tensor& value, Tensor& z, cudaStream_t stream) {
    const std::int32_t cols = x.ne[1];
    if (cols < kFirstExactCols || cols > kLastSnapshotExactCols) {
        throw std::invalid_argument("Q8 fused GDN input snapshot requires T=2..16");
    }
    kSnapshotLaunchers[cols - kFirstExactCols](x, weight, conv_weight, conv_states, valid_columns,
                                               initial_slot, snapshot_base_slot, query, key, value,
                                               z, stream);
    CUDA_CHECK(cudaGetLastError());
}

void q8_gdn_input_splitk_conv_record_launch(const Tensor& x, const Weight& weight,
                                            const Tensor& conv_weight, const Tensor& conv_states,
                                            const Tensor& valid_columns, const Tensor& initial_slot,
                                            Tensor& conv_record, Tensor& query, Tensor& key,
                                            Tensor& value, Tensor& z, cudaStream_t stream) {
    const std::int32_t cols = x.ne[1];
    if (cols < kFirstExactCols || cols > kLastSnapshotExactCols) {
        throw std::invalid_argument("Q8 fused GDN input record requires T=2..16");
    }
    kRecordLaunchers[cols - kFirstExactCols](x, weight, conv_weight, conv_states, valid_columns,
                                             initial_slot, conv_record, query, key, value, z,
                                             stream);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
