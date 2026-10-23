#include "ops/linear/kernels/q8/q8_geometry.h"
#include "ops/linear/kernels/q8/q8_grouped_sliced_k_launch.cuh"
#include "ops/linear/kernels/q8/q8_instances.cuh"
#include "core/weight.h"
#include "ops/linear_attention/gated_delta_net/input_conv/q8/launch.h"
#include "core/device.h"
#include "ops/linear_attention/gated_delta_net/input_conv/kernels/output.cuh"
#include "ops/linear/kernels/q8/q8_sliced_k_launch.cuh"
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
                             const ConstStateFieldView& conv_states, const Tensor& valid_columns,
                             Tensor& query, Tensor& key, Tensor& value, Tensor& z, Publish publish,
                             cudaStream_t stream) {
    static_assert(ActiveCols >= 2 && ActiveCols <= 16);
    constexpr int TileCols = ActiveCols <= 8 ? 8 : 16;
    using Geometry         = Q8LinearGeometry<kRows, kHidden>;
    using Schedule         = Q8SlicedKDefault<TileCols, ActiveCols>;
    const auto output      = make_gdn_conv_output<GdnConvGeometry<4096>, ActiveCols>(
        conv_weight, conv_states, valid_columns, query, key, value, z, publish);
    launch_q8_a16_sliced_k_mma<
        typename Schedule::template with_problem<Geometry::kInputRows, ActiveCols, true>>(
        q8_linear_operands(x, weight), output, GdnConvRowEpilogue<ActiveCols>{}, stream);
}

template <int ActiveCols>
void launch_active_cols_conv_snapshot(const Tensor& x, const Weight& weight,
                                      const Tensor& conv_weight,
                                      const ConstStateFieldView& conv_states,
                                      const Tensor& valid_columns,
                                      const gdn_input::HistoryDestination& destination,
                                      Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                                      cudaStream_t stream) {
    launch_active_cols_conv<ActiveCols>(x, weight, conv_weight, conv_states, valid_columns, query,
                                        key, value, z,
                                        make_snapshot_history_publish(destination, 8192), stream);
}

template <int ActiveCols>
void launch_active_cols_conv_record(const Tensor& x, const Weight& weight,
                                    const Tensor& conv_weight,
                                    const ConstStateFieldView& conv_states,
                                    const Tensor& valid_columns, Tensor& conv_record, Tensor& query,
                                    Tensor& key, Tensor& value, Tensor& z, cudaStream_t stream) {
    launch_active_cols_conv<ActiveCols>(
        x, weight, conv_weight, conv_states, valid_columns, query, key, value, z,
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
using SnapshotLauncher   = void (*)(const Tensor&, const Weight&, const Tensor&,
                                  const ConstStateFieldView&, const Tensor&,
                                  const gdn_input::HistoryDestination&, Tensor&, Tensor&, Tensor&,
                                  Tensor&, cudaStream_t);
using RecordLauncher     = void (*)(const Tensor&, const Weight&, const Tensor&,
                                const ConstStateFieldView&, const Tensor&, Tensor&, Tensor&,
                                Tensor&, Tensor&, Tensor&, cudaStream_t);

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

ProjectionLauncher select_projection(int columns) {
    if (columns < kFirstExactCols || columns > 96)
        throw std::logic_error("Q8 GDN split-K MMA requires T=2..96");
    if (columns <= kLastProjectionExactCols) return kProjectionLaunchers[columns - kFirstExactCols];
    if (columns <= 48) return launch_medium_cols<48, 4, 2, 3>;
    if (columns <= 64) return launch_medium_cols<64, 4, 2, 2>;
    return launch_medium_cols<96, 2, 4, 3>;
}
} // namespace

void launch_q8_gdn_input_splitk_mma(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                    cudaStream_t stream) {
    select_projection(x.ne[1])(x, weight, qkv, z, stream);
    CUDA_CHECK(cudaGetLastError());
}

void launch_q8_gdn_input_splitk_conv_snapshot(const Tensor& x, const Weight& weight,
                                              const Tensor& conv_weight,
                                              const ConstStateFieldView& conv_states,
                                              const Tensor& valid_columns,
                                              const gdn_input::HistoryDestination& destination,
                                              Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                                              cudaStream_t stream) {
    const std::int32_t cols = x.ne[1];
    if (cols < kFirstExactCols || cols > kLastSnapshotExactCols) {
        throw std::invalid_argument("Q8 fused GDN input snapshot requires T=2..16");
    }
    kSnapshotLaunchers[cols - kFirstExactCols](x, weight, conv_weight, conv_states, valid_columns,
                                               destination, query, key, value, z, stream);
    CUDA_CHECK(cudaGetLastError());
}

void launch_q8_gdn_input_splitk_conv_record(const Tensor& x, const Weight& weight,
                                            const Tensor& conv_weight,
                                            const ConstStateFieldView& conv_states,
                                            const Tensor& valid_columns, Tensor& conv_record,
                                            Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                                            cudaStream_t stream) {
    const std::int32_t cols = x.ne[1];
    if (cols < kFirstExactCols || cols > kLastSnapshotExactCols) {
        throw std::invalid_argument("Q8 fused GDN input record requires T=2..16");
    }
    kRecordLaunchers[cols - kFirstExactCols](x, weight, conv_weight, conv_states, valid_columns,
                                             conv_record, query, key, value, z, stream);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
