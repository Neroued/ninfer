#include "ops/linear_attention/gated_delta_net/replay/launch.h"
#include "ops/common/cuda/reduce.cuh"
#include "ops/linear_attention/gated_delta_net/replay/fold_io.cuh"
#include <type_traits>
#include "core/device.h"
#include "ops/common/host/operands.h"
#include <cuda_bf16.h>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail::gated_delta_net {
namespace {
static_assert(sizeof(GdnReplayFoldKernelRow) == 32);
static_assert(alignof(GdnReplayFoldKernelRow) == 16);
static_assert(sizeof(GdnReplayFoldKernelRows) == 256);
static_assert(alignof(GdnReplayFoldKernelRows) == 16);
static_assert(std::is_trivially_copyable_v<GdnReplayFoldKernelRows>);

template <class Geometry, class Rows>
void launch_replay_fold_fixed(const GdnReplayRecords& records,
                              LinearAttentionStateAllLayersView states, const Rows& rows,
                              std::int32_t active_rows, cudaStream_t stream) {
    const FoldAccess<Geometry, Rows> access{
        static_cast<const __nv_bfloat16*>(records.key.data),
        static_cast<const __nv_bfloat16*>(records.value.data),
        reinterpret_cast<const uint2*>(records.gate.data),
        static_cast<const __nv_bfloat16*>(records.conv.data),
        common::operand::state_field_base<float>(states.recurrent),
        common::operand::state_field_base<__nv_bfloat16>(states.conv),
        static_cast<std::int64_t>(states.recurrent.field.layer_stride / sizeof(float)),
        static_cast<std::int64_t>(states.conv.field.layer_stride / sizeof(__nv_bfloat16)),
        records.spec.record_capacity,
        records.spec.width,
        rows,
    };
    const dim3 grid(static_cast<unsigned>(Geometry::kValueHeads),
                    static_cast<unsigned>(active_rows),
                    static_cast<unsigned>(Geometry::kLayers * (kStateDim / kBlockDv)));
    const dim3 block(common::kWarpSize, kNumWarps, 1);
    recurrent_fold_kernel<Geometry, Rows><<<grid, block, 0, stream>>>(access);
    CUDA_CHECK(cudaGetLastError());
}

template <class Rows>
void dispatch_replay_fold(const GdnReplayRecords& records, LinearAttentionStateAllLayersView states,
                          const Rows& rows, std::int32_t active_rows, cudaStream_t stream) {
    switch (select_fold_geometry(records.spec)) {
    case FoldGeometryId::Layers48Heads48:
        return launch_replay_fold_fixed<FoldGeometry48x48>(records, states, rows, active_rows,
                                                           stream);
    case FoldGeometryId::Layers30Heads32:
        return launch_replay_fold_fixed<FoldGeometry30x32>(records, states, rows, active_rows,
                                                           stream);
    case FoldGeometryId::Layers36Heads48:
        return launch_replay_fold_fixed<FoldGeometry36x48>(records, states, rows, active_rows,
                                                           stream);
    case FoldGeometryId::Layers36Heads24:
        return launch_replay_fold_fixed<FoldGeometry36x24>(records, states, rows, active_rows,
                                                           stream);
    }
}

} // namespace

void launch_replay_fold(const GdnReplayRecords& records, LinearAttentionStateAllLayersView states,
                        const GdnReplayFoldKernelRows& rows, std::int32_t active_rows,
                        cudaStream_t stream) {
    dispatch_replay_fold(records, states, rows, active_rows, stream);
}

void launch_replay_fold(const GdnReplayRecords& records, LinearAttentionStateAllLayersView states,
                        const GdnReplayFoldDeviceRows& rows, std::int32_t active_rows,
                        cudaStream_t stream) {
    dispatch_replay_fold(records, states, rows, active_rows, stream);
}

} // namespace ninfer::ops::detail::gated_delta_net
