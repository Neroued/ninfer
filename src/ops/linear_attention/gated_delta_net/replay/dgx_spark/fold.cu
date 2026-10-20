#include "ops/linear_attention/gated_delta_net/replay/launch.h"
#include "core/device.h"
#include "ops/common/cuda/pipeline.cuh"
#include "ops/linear_attention/gated_delta_net/replay/fold_io.cuh"

#include <cuda_bf16.h>
#include <cstdint>
#include <type_traits>

namespace ninfer::ops::detail::gated_delta_net {
namespace {
constexpr int kFoldWarps = 8, kFoldDv = 32, kFoldTiles = kStateDim / kFoldDv;
constexpr int kFoldWindow = 16, kFoldLanes = 16, kFoldRows = 2, kFoldCols = 8;

struct alignas(16) FoldWindow {
    float key[kFoldWindow][kStateDim];
    __nv_bfloat16 value[kFoldWindow][kFoldDv];
    float alpha[kFoldWindow], beta[kFoldWindow];
};

template <class Geometry, class Rows>
__global__ __launch_bounds__(common::kWarpSize* kFoldWarps, 2) void fold_kernel(
    const __grid_constant__ FoldAccess<Geometry, Rows> access, unsigned row_map) {
    const int lane = threadIdx.x % kFoldLanes, group = threadIdx.x / kFoldLanes;
    const int tid  = threadIdx.y * common::kWarpSize + threadIdx.x;
    const int tile = blockIdx.x % kFoldTiles, head = blockIdx.x / kFoldTiles;
    const int row = (row_map >> (blockIdx.y * 4)) & 7;
    const RecurrentCoordinates coord{
        static_cast<int>(threadIdx.x),
        static_cast<int>(threadIdx.y),
        row,
        static_cast<int>(blockIdx.z),
        tile,
        static_cast<unsigned>(head),
        static_cast<unsigned>(head / (Geometry::kValueHeads / Geometry::kQkHeads)),
        static_cast<unsigned>(tile * kFoldDv + threadIdx.y * 4 + group * kFoldRows),
        static_cast<unsigned>(lane * 4)};
    const int valid = access.active_columns(coord);
    if (!valid) return;
    __shared__ FoldWindow sm;

    // Stage only the committed prefix, overlapping record copies with the state loads.
    for (int i = tid; i < valid * (kFoldDv / 8); i += kFoldWarps * common::kWarpSize) {
        const int token = i / (kFoldDv / 8), col = i % (kFoldDv / 8) * 8;
        common::cp_async<16, Cache::cg>(&sm.value[token][col],
                                        access.value_ptr(coord, token) + tile * kFoldDv + col);
    }
    common::cp_commit();
    __align__(16) float state[kFoldRows][kFoldCols];
    const float* source = access.state_read_base(coord);
#pragma unroll
    for (int r = 0; r < kFoldRows; ++r) {
#pragma unroll
        for (int p = 0; p < kFoldCols / 4; ++p) {
            const int col = lane * 4 + p * kFoldLanes * 4;
            common::store_vec(
                state[r] + p * 4,
                common::load_vec<float4>(source + (coord.dv_base + r) * kStateDim + col));
        }
    }

    // One subgroup normalizes each raw key once for all sixteen state-row subgroups.
    // Keep the same key segments, accumulation order and XOR tree as recurrent_subgroup.
    const int token = threadIdx.y * 2 + group;
    if (token < valid) {
        __align__(16) float key[kFoldCols];
#pragma unroll
        for (int p = 0; p < kFoldCols / 4; ++p) {
            const auto raw =
                load_raw_qk_lane(access.key_ptr(coord, token), lane * 4 + p * kFoldLanes * 4);
#pragma unroll
            for (int j = 0; j < 4; ++j) key[p * 4 + j] = raw.value[j];
        }
        float sum = 0;
#pragma unroll
        for (int j = 0; j < kFoldCols; ++j) sum += key[j] * key[j];
        sum             = common::warp_sum<kFoldLanes>(sum, 0xffffU << (group * kFoldLanes));
        const float inv = rsqrtf(sum + kQkL2NormEps);
#pragma unroll
        for (int j = 0; j < kFoldCols; ++j) key[j] *= inv;
#pragma unroll
        for (int p = 0; p < kFoldCols / 4; ++p)
            common::store_vec(&sm.key[token][lane * 4 + p * kFoldLanes * 4],
                              common::load_vec<float4>(key + p * 4));
        if (lane == 0) {
            const auto gate = access.load_gate(coord, token);
            sm.alpha[token] = expf(gate.g);
            sm.beta[token]  = gate.beta;
        }
    }
    common::cp_wait<0>();
    __syncthreads();

    for (int t = 0; t < valid; ++t) {
        __align__(16) float key[kFoldCols];
#pragma unroll
        for (int p = 0; p < kFoldCols / 4; ++p)
            common::store_vec(key + p * 4,
                              common::load_vec<float4>(&sm.key[t][lane * 4 + p * kFoldLanes * 4]));
        const float alpha = sm.alpha[t], beta = sm.beta[t];
#pragma unroll
        for (int r = 0; r < kFoldRows; ++r) {
            const float value = __bfloat162float(sm.value[t][coord.dv_base % kFoldDv + r]);
            float dot         = 0;
#pragma unroll
            for (int j = 0; j < kFoldCols; ++j) dot += state[r][j] * key[j];
            dot               = common::warp_sum<kFoldLanes>(dot);
            const float delta = beta * (value - alpha * dot);
#pragma unroll
            for (int j = 0; j < kFoldCols; ++j) state[r][j] = alpha * state[r][j] + delta * key[j];
        }
    }
    float* destination = access.state_write_base(coord);
#pragma unroll
    for (int r = 0; r < kFoldRows; ++r) {
#pragma unroll
        for (int p = 0; p < kFoldCols / 4; ++p) {
            const int col = lane * 4 + p * kFoldLanes * 4;
            common::store_vec(destination + (coord.dv_base + r) * kStateDim + col,
                              common::load_vec<float4>(state[r] + p * 4));
        }
    }

    // Each four-warp group owns one disjoint 128-channel convolution-history block.
    auto history = coord;
    history.warp %= kNumWarps;
    const int conv_block = static_cast<int>(blockIdx.x) * (kFoldWarps / kNumWarps) +
                           static_cast<int>(threadIdx.y) / kNumWarps;
    history.value_head = conv_block / 8;
    history.state_tile = conv_block % 8;
    access.publish_final_conv_history(history, valid);
}

static_assert(sizeof(GdnReplayFoldKernelRow) == 16);
static_assert(alignof(GdnReplayFoldKernelRow) == 16);
static_assert(sizeof(GdnReplayFoldKernelRows) == 128);
static_assert(alignof(GdnReplayFoldKernelRows) == 16);
static_assert(std::is_trivially_copyable_v<GdnReplayFoldKernelRows>);

template <class Geometry, class Rows>
void launch_replay_fold_fixed(const GdnReplayRecords& records,
                              LinearAttentionStateAllLayersView states, const Rows& rows,
                              unsigned row_map, int launched_rows, cudaStream_t stream) {
    const FoldAccess<Geometry, Rows> access{
        static_cast<const __nv_bfloat16*>(records.key.data),
        static_cast<const __nv_bfloat16*>(records.value.data),
        reinterpret_cast<const uint2*>(records.gate.data),
        static_cast<const __nv_bfloat16*>(records.conv.data),
        static_cast<float*>(states.recurrent_layer0.data),
        static_cast<__nv_bfloat16*>(states.conv_layer0.data),
        states.recurrent_layer_stride_bytes / static_cast<std::int64_t>(sizeof(float)),
        states.conv_layer_stride_bytes / static_cast<std::int64_t>(sizeof(__nv_bfloat16)),
        states.recurrent_layer0.nb[3] / static_cast<std::int64_t>(sizeof(float)),
        states.conv_layer0.nb[2] / static_cast<std::int64_t>(sizeof(__nv_bfloat16)),
        records.spec.record_capacity,
        records.spec.width,
        rows,
    };
    const dim3 grid(Geometry::kValueHeads * kFoldTiles, launched_rows, Geometry::kLayers);
    fold_kernel<Geometry, Rows>
        <<<grid, dim3(common::kWarpSize, kFoldWarps), 0, stream>>>(access, row_map);
    CUDA_CHECK(cudaGetLastError());
}

template <class Rows>
void dispatch_replay_fold(const GdnReplayRecords& records, LinearAttentionStateAllLayersView states,
                          const Rows& rows, std::int32_t active_rows, cudaStream_t stream) {
    // Host metadata can omit empty rows. Device metadata keeps a fixed grid and is read on replay.
    unsigned row_map  = 0;
    int launched_rows = 0;
    for (int row = 0; row < active_rows; ++row) {
        if constexpr (std::is_same_v<Rows, GdnReplayFoldKernelRows>) {
            if (!rows.row[row].commit_columns) continue;
        }
        row_map |= static_cast<unsigned>(row) << (launched_rows++ * 4);
    }
    if (!launched_rows) return;
    switch (select_fold_geometry(records.spec)) {
    case FoldGeometryId::Layers48Heads48:
        return launch_replay_fold_fixed<FoldGeometry48x48>(records, states, rows, row_map,
                                                           launched_rows, stream);
    case FoldGeometryId::Layers30Heads32:
        return launch_replay_fold_fixed<FoldGeometry30x32>(records, states, rows, row_map,
                                                           launched_rows, stream);
    case FoldGeometryId::Layers36Heads48:
        return launch_replay_fold_fixed<FoldGeometry36x48>(records, states, rows, row_map,
                                                           launched_rows, stream);
    case FoldGeometryId::Layers36Heads24:
        return launch_replay_fold_fixed<FoldGeometry36x24>(records, states, rows, row_map,
                                                           launched_rows, stream);
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
