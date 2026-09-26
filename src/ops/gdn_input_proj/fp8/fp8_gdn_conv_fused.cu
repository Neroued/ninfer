#include "ops/linear/fp8/fp8_template_launch.cuh"
#include "core/weight.h"
#include "ops/gdn_input_proj/fp8/fp8_gdn_conv_plan.h"

#include "core/device.h"
#include "ops/gdn_input_proj/gdn_conv_output.cuh"
#include "ops/linear/fp8/fp8_schedule.cuh"
#include "ops/linear/fp8/fp8_a16_gemv.cuh"
#include "ops/linear/fp8/fp8_a16_simt.cuh"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

template <int Tokens, class Publish>
struct Fp8GdnConvEpilogue {
    [[maybe_unused]] static constexpr int kRowTokens = Tokens;

    template <class Output>
    __device__ __forceinline__ void apply_row(const Output& output, int row, int,
                                              const float (&values)[Tokens], int) const {
        output.store_row(row, values);
    }
};

using Geometry = Fp8N16384K5120;

using SnapshotLaunch = void (*)(const Tensor&, const Weight&, const Tensor&, Tensor&, const Tensor&,
                                const Tensor&, const Tensor&, Tensor&, Tensor&, Tensor&, Tensor&,
                                cudaStream_t);
using RecordLaunch   = void (*)(const Tensor&, const Weight&, const Tensor&, const Tensor&,
                              const Tensor&, const Tensor&, Tensor&, Tensor&, Tensor&, Tensor&,
                              Tensor&, cudaStream_t);

template <int ActiveTokens, class Publish>
void launch_small_t(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                    const Tensor& conv_states, const Tensor& valid_columns,
                    const Tensor& initial_slot, Tensor& query, Tensor& key, Tensor& value,
                    Tensor& z, Publish publish, cudaStream_t stream) {
    using Schedule =
        Fp8A16SimtSchedule<8, 2, (ActiveTokens >= 5 && ActiveTokens <= 6) ? 8 : 16, ActiveTokens, 1,
                           ActiveTokens <= 4 ? Fp8SimtActivationAccess::SharedPhase
                                             : Fp8SimtActivationAccess::TokenPacked,
                           Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;
    static_assert(Schedule::kBlockTokens == ActiveTokens);
    using Output = GdnConvOutput<ActiveTokens, Publish>;
    launch_fp8_a16_simt<Fp8ScheduleInstance<Schedule, Geometry::kInputRows, ActiveTokens, true>>(
        fp8_a16_operands(x, weight),
        make_gdn_conv_output<ActiveTokens>(conv_weight, conv_states, valid_columns, initial_slot,
                                           query, key, value, z, publish),
        Fp8GdnConvEpilogue<ActiveTokens, Publish>{}, stream);
}

template <int ActiveTokens>
void launch_snapshot_small_t(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                             Tensor& conv_states, const Tensor& valid_columns,
                             const Tensor& initial_slot, const Tensor& snapshot_base_slot,
                             Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                             cudaStream_t stream) {
    launch_small_t<ActiveTokens>(
        x, weight, conv_weight, conv_states, valid_columns, initial_slot, query, key, value, z,
        SnapshotHistoryPublish{static_cast<__nv_bfloat16*>(conv_states.data),
                               static_cast<const std::int32_t*>(snapshot_base_slot.data),
                               kGdnChannels},
        stream);
}

template <int ActiveTokens>
void launch_record_small_t(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                           const Tensor& conv_states, const Tensor& valid_columns,
                           const Tensor& initial_slot, Tensor& conv_record, Tensor& query,
                           Tensor& key, Tensor& value, Tensor& z, cudaStream_t stream) {
    launch_small_t<ActiveTokens>(x, weight, conv_weight, conv_states, valid_columns, initial_slot,
                                 query, key, value, z,
                                 RecordColumnPublish{static_cast<__nv_bfloat16*>(conv_record.data),
                                                     kGdnChannels, ActiveTokens},
                                 stream);
}

void launch_snapshot_decode(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                            Tensor& conv_states, const Tensor& valid_columns,
                            const Tensor& initial_slot, const Tensor& snapshot_base_slot,
                            Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                            cudaStream_t stream) {
    using Schedule = Fp8A16GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 2>;
    launch_fp8_a16_gemv<Fp8ScheduleInstance<Schedule, Geometry::kInputRows>>(
        fp8_a16_operands(x, weight),
        make_gdn_conv_output<1>(
            conv_weight, conv_states, valid_columns, initial_slot, query, key, value, z,
            SnapshotHistoryPublish{static_cast<__nv_bfloat16*>(conv_states.data),
                                   static_cast<const std::int32_t*>(snapshot_base_slot.data),
                                   kGdnChannels}),
        Fp8GdnConvEpilogue<1, SnapshotHistoryPublish>{}, stream);
}

template <std::size_t... Offsets>
constexpr auto make_snapshot_launchers(std::index_sequence<Offsets...>) {
    return std::array<SnapshotLaunch, sizeof...(Offsets)>{
        &launch_snapshot_small_t<2 + static_cast<int>(Offsets)>...};
}

template <std::size_t... Offsets>
constexpr auto make_record_launchers(std::index_sequence<Offsets...>) {
    return std::array<RecordLaunch, sizeof...(Offsets)>{
        &launch_record_small_t<2 + static_cast<int>(Offsets)>...};
}

constexpr auto kSnapshotLaunchers = make_snapshot_launchers(std::make_index_sequence<10 - 2 + 1>{});
constexpr auto kRecordLaunchers   = make_record_launchers(std::make_index_sequence<10 - 2 + 1>{});

} // namespace

void fp8_gdn_snapshot_fused_launch(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                                   Tensor& conv_states, const Tensor& valid_columns,
                                   const Tensor& initial_slot, const Tensor& snapshot_base_slot,
                                   Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                                   cudaStream_t stream) {
    if (x.ne[2] != 1 || x.ne[1] <= 0 || x.ne[1] > 10) {
        throw std::invalid_argument("fp8 GDN snapshot fused: unsupported B/W");
    }
    if (x.ne[1] == 1) {
        launch_snapshot_decode(x, weight, conv_weight, conv_states, valid_columns, initial_slot,
                               snapshot_base_slot, query, key, value, z, stream);
        return;
    }
    kSnapshotLaunchers[static_cast<std::size_t>(x.ne[1] - 2)](
        x, weight, conv_weight, conv_states, valid_columns, initial_slot, snapshot_base_slot, query,
        key, value, z, stream);
}

void fp8_gdn_record_fused_launch(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                                 const Tensor& conv_states, const Tensor& valid_columns,
                                 const Tensor& initial_slot, Tensor& conv_record, Tensor& query,
                                 Tensor& key, Tensor& value, Tensor& z, cudaStream_t stream) {
    if (x.ne[2] != 1 || x.ne[1] < 2 || x.ne[1] > 10) {
        throw std::invalid_argument("fp8 GDN record fused: unsupported B/W");
    }
    kRecordLaunchers[static_cast<std::size_t>(x.ne[1] - 2)](
        x, weight, conv_weight, conv_states, valid_columns, initial_slot, conv_record, query, key,
        value, z, stream);
}

} // namespace ninfer::ops::detail
