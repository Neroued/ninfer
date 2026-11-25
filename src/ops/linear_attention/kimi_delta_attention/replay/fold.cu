#include "ops/linear_attention/kimi_delta_attention/replay/launch.h"
#include "ops/linear_attention/kimi_delta_attention/kernels/transition.cuh"
#include "ops/linear_attention/kimi_delta_attention/plan.h"
#include "ops/common/cuda/pipeline.cuh"
#include "ops/common/host/operands.h"
#include "core/platform.h"

#include <algorithm>

namespace ninfer::ops::detail::kimi_delta_attention {
namespace {
constexpr int kTileWidth   = 16;
constexpr int kRowsPerWarp = 4;
constexpr int kGridExtent  = 65535;

__device__ __forceinline__ KdaReplayFoldRow select_row(const KdaReplayFoldKernelRows& rows, int b) {
    return rows.row[b];
}

__device__ __forceinline__ KdaReplayFoldRow select_row(const KdaReplayFoldDeviceRows& rows, int b) {
    const int count = rows.columns[b];
    if (!count) return {-1, -1, 0};
    return {rows.sources[b], rows.destinations[b], count};
}

template <class StateT>
struct FoldArguments {
    const __nv_bfloat16 *key, *conv;
    const float *delta, *alpha;
    StateT* recurrent;
    __nv_bfloat16* history;
    std::int64_t recurrent_stride, history_stride;
    int qk_heads, value_heads, group, conv_channels, width, capacity, row_begin, layer_begin;

    __device__ __forceinline__ std::int64_t key_index(std::int64_t column, int qhead) const {
        return (column * qk_heads + qhead) * kStateDim;
    }

    __device__ __forceinline__ std::int64_t value_index(std::int64_t column, int head) const {
        return (column * value_heads + head) * kStateDim;
    }
};

template <int Warps, class StateT>
__device__ __forceinline__ float4 load_state_row(const StateT* state, int row, int col) {
    if constexpr (NINFER_TARGET_SPARK && Warps < 16)
        return common::load_state_float4(state + row * kStateDim + col);
    return common::load_state_float4_l2_retained(state + row * kStateDim + col);
}

template <class StateT>
__device__ __forceinline__ void store_state_row(StateT* state, int row, int col, float4 value) {
    common::store_state_float4_streaming(state + row * kStateDim + col, value);
}

__device__ __forceinline__ float4 add_outer(float4 acc, float delta, float4 key) {
    return {fmaf(delta, key.x, acc.x), fmaf(delta, key.y, acc.y), fmaf(delta, key.z, acc.z),
            fmaf(delta, key.w, acc.w)};
}

template <int TileRows>
struct alignas(16) FoldWindow {
    float delta[kTileWidth][TileRows];
    float scaled_key[kTileWidth][kStateDim];
    float total_decay[kStateDim];
};

template <class StateT>
__device__ __forceinline__ void publish_conv(const FoldArguments<StateT>& a, KdaReplayFoldRow row,
                                             std::int64_t column, int layer, int stripe,
                                             int stripes, int lane) {
    constexpr int kVectorChannels = sizeof(uint4) / sizeof(__nv_bfloat16);
    constexpr int kChannelsPerCta = common::kWarpSize * kVectorChannels;
    const auto channels           = std::int64_t(a.conv_channels);
    const auto* history           = a.history + layer * a.history_stride;
    const auto* source            = history + row.source_offset / sizeof(__nv_bfloat16);
    auto* destination =
        a.history + layer * a.history_stride + row.destination_offset / sizeof(__nv_bfloat16);
    const auto* record = a.conv + column * channels;
    for (auto c = std::int64_t(stripe) * kChannelsPerCta + lane * kVectorChannels; c < channels;
         c += std::int64_t(stripes) * kChannelsPerCta) {
        // Shifting left in column order preserves each old word until its final read.
#pragma unroll
        for (int i = 0; i < 3; ++i) {
            const auto index = std::int64_t(row.commit_columns) + i;
            const auto* from =
                index < 3 ? source + index * channels : record + (index - 3) * channels;
            common::store_vec(destination + i * channels + c, common::load_vec<uint4>(from + c));
        }
    }
}

__device__ __forceinline__ float4 scaled(float4 a, float4 b) {
    return {a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w};
}

template <int Warps, class StateT>
__device__ __forceinline__ void fold_one(const FoldArguments<StateT>& a, const StateT* source,
                                         StateT* destination, std::int64_t column, int head,
                                         int qhead, int row, int lane, int col) {
    float4 initial[kRowsPerWarp];
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) initial[r] = load_state_row<Warps>(source, row + r, col);

    // The one-column path is independent in each warp and needs no CTA rendezvous.
    const auto vo = a.value_index(column, head);
    float key[kQkPerLane];
    common::load_bf16x4(key, a.key + a.key_index(column, qhead) + col);
    const auto decay  = common::load_vec<float4>(a.alpha + vo + col);
    const float delta = lane < kRowsPerWarp ? a.delta[vo + row + lane] : 0.0F;
    normalize_qk(key, lane);
    const auto k = common::load_vec<float4>(key);
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
        const auto acc =
            add_outer(scaled(initial[r], decay), __shfl_sync(common::kFullWarpMask, delta, r), k);
        store_state_row(destination, row + r, col, acc);
    }
}

template <int Warps, class StateT>
__device__ __forceinline__ void
fold_chunk(const FoldArguments<StateT>& a, FoldWindow<Warps * kRowsPerWarp>& sm,
           float4 (&state)[kRowsPerWarp], std::int64_t column, int width, int head, int qhead,
           int tile, int warp, int lane, int col) {
    constexpr int TileRows = Warps * kRowsPerWarp;
    constexpr int DeltaVec = TileRows / 4;
    const int row          = warp * kRowsPerWarp;

    // Leave the normalized keys in registers while the CTA makes the alpha suffix products.
    float4 key[kTileWidth / Warps];
#pragma unroll
    for (int i = 0; i < kTileWidth / Warps; ++i) {
        const int t = warp + i * Warps;
        if (t < width) {
            const auto vo = a.value_index(column + t, head);
            if (lane < DeltaVec) {
                common::cp_async<16, Cache::cg>(sm.delta[t] + col,
                                                a.delta + vo + tile * TileRows + col);
            }
            common::cp_async<16, Cache::cg>(sm.scaled_key[t] + col, a.alpha + vo + col);
        }
    }
    common::cp_commit();

#pragma unroll
    for (int i = 0; i < kTileWidth / Warps; ++i) {
        const int t = warp + i * Warps;
        if (t < width) {
            float kv[kQkPerLane];
            common::load_bf16x4(kv, a.key + a.key_index(column + t, qhead) + col);
            normalize_qk(kv, lane);
            key[i] = common::load_vec<float4>(kv);
        }
    }
    common::cp_wait<0>();
    __syncthreads();

    const int channel = warp * common::kWarpSize + lane;
    if (channel < kStateDim) {
        float decay = 1.0F;
        for (int t = width - 1; t >= 0; --t) {
            const float alpha         = sm.scaled_key[t][channel];
            sm.scaled_key[t][channel] = decay;
            decay *= alpha;
        }
        sm.total_decay[channel] = decay;
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < kTileWidth / Warps; ++i) {
        const int t = warp + i * Warps;
        if (t < width) {
            const auto suffix = common::load_vec<float4>(sm.scaled_key[t] + col);
            common::store_vec(sm.scaled_key[t] + col, scaled(key[i], suffix));
        }
    }
    __syncthreads();

    const auto decay = common::load_vec<float4>(sm.total_decay + col);
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) state[r] = scaled(state[r], decay);
    for (int t = 0; t < width; ++t) {
        const auto k = common::load_vec<float4>(sm.scaled_key[t] + col);
#pragma unroll
        for (int r = 0; r < kRowsPerWarp; ++r)
            state[r] = add_outer(state[r], sm.delta[t][row + r], k);
    }
}

template <int Warps, class StateT>
__device__ __forceinline__ void
fold_short(const FoldArguments<StateT>& a, FoldWindow<Warps * kRowsPerWarp>& sm,
           float4 (&state)[kRowsPerWarp], std::int64_t column, int width, int head, int qhead,
           int tile, int warp, int lane, int col) {
    constexpr int TileRows = Warps * kRowsPerWarp;
    const int row          = warp * kRowsPerWarp;
    for (int t = warp; t < width; t += Warps) {
        const auto vo = a.value_index(column + t, head);
        if (lane < TileRows / 4)
            common::cp_async<16, Cache::cg>(sm.delta[t] + col,
                                            a.delta + vo + tile * TileRows + col);
        common::cp_commit();
        float key[kQkPerLane];
        common::load_bf16x4(key, a.key + a.key_index(column + t, qhead) + col);
        normalize_qk(key, lane);
        // With at most four columns, reading each key's suffix directly avoids a second
        // rendezvous. Delta is shared across all the row warps in the tile.
        float4 decay{1, 1, 1, 1};
#pragma unroll
        for (int i = 3; i >= 1; --i)
            if (i < width && i > t)
                decay = scaled(decay, common::load_vec<float4>(
                                          a.alpha + a.value_index(column + i, head) + col));
        common::store_vec(sm.scaled_key[t] + col, scaled(common::load_vec<float4>(key), decay));
        if (t == 0)
            common::store_vec(sm.total_decay + col,
                              scaled(decay, common::load_vec<float4>(a.alpha + vo + col)));
        common::cp_wait<0>();
    }
    __syncthreads();
    const auto decay = common::load_vec<float4>(sm.total_decay + col);
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) state[r] = scaled(state[r], decay);
#pragma unroll
    for (int t = 0; t < 4; ++t)
        if (t < width) {
            const auto k = common::load_vec<float4>(sm.scaled_key[t] + col);
#pragma unroll
            for (int r = 0; r < kRowsPerWarp; ++r)
                state[r] = add_outer(state[r], sm.delta[t][row + r], k);
        }
}

template <class Rows, int Warps, class StateT>
__global__ __launch_bounds__(32 * Warps, 2) void linear_fold_kernel(
    const __grid_constant__ FoldArguments<StateT> a, const __grid_constant__ Rows rows) {
    constexpr int TileRows = Warps * kRowsPerWarp;
    constexpr int Tiles    = kStateDim / TileRows;
    static_assert(kStateDim % TileRows == 0 && Warps <= 16);
    const int head = blockIdx.x / Tiles, tile = blockIdx.x % Tiles;
    const int layer = a.layer_begin + blockIdx.y, batch = blockIdx.z;
    const auto row   = select_row(rows, batch);
    const int commit = row.commit_columns;
    if (!commit) return;

    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int col = lane * kQkPerLane, qhead = head / a.group;
    const auto column = (std::int64_t(layer) * a.capacity + a.row_begin + batch) * a.width;
    auto* head_state =
        a.recurrent + layer * a.recurrent_stride + std::int64_t(head) * kStateDim * kStateDim;
    auto* destination = head_state + row.destination_offset / static_cast<std::int64_t>(sizeof(StateT));
    const auto* source = head_state + row.source_offset / static_cast<std::int64_t>(sizeof(StateT));
    const int first_row = tile * TileRows;
    __shared__ FoldWindow<TileRows> sm;
    if (commit == 1) {
        fold_one<Warps>(a, source, destination, column, head, qhead,
                        first_row + warp * kRowsPerWarp, lane, col);
    } else {
        const int state_row = first_row + warp * kRowsPerWarp;
        float4 state[kRowsPerWarp];
#pragma unroll
        for (int r = 0; r < kRowsPerWarp; ++r)
            state[r] = load_state_row<Warps>(source, state_row + r, col);
        if (commit <= 4) {
            fold_short<Warps>(a, sm, state, column, commit, head, qhead, tile, warp, lane, col);
        } else {
            for (int t = 0; t < commit;) {
                const int width = min(kTileWidth, commit - t);
                fold_chunk<Warps>(a, sm, state, column + t, width, head, qhead, tile, warp, lane,
                                  col);
                t += width;
                if (t < commit) __syncthreads();
            }
        }
#pragma unroll
        for (int r = 0; r < kRowsPerWarp; ++r)
            store_state_row(destination, state_row + r, col, state[r]);
    }
    // Tile stripes partition the convolution history across the same CTA grid.
    if (warp == 0)
        publish_conv(a, row, column, layer, tile * a.value_heads + head, Tiles * a.value_heads,
                     lane);
}

template <class Rows, class StateT>
void launch_typed(const KdaReplayRecords& r, LinearAttentionStateAllLayersView s, const Rows& rows,
                  int row_begin, int batch, CudaExecutionView execution) {
    const auto [key, delta, alpha] = ninfer::detail::KdaReplayRecordAccess::planes(r);
    FoldArguments<StateT> a{
        static_cast<const __nv_bfloat16*>(key.data),
        static_cast<const __nv_bfloat16*>(r.conv.data),
        static_cast<const float*>(delta.data),
        static_cast<const float*>(alpha.data),
        common::operand::state_field_base<StateT>(s.recurrent),
        common::operand::state_field_base<__nv_bfloat16>(s.conv),
        static_cast<std::int64_t>(s.recurrent.field.layer_stride / sizeof(StateT)),
        static_cast<std::int64_t>(s.conv.field.layer_stride / sizeof(__nv_bfloat16)),
        r.spec.qk_heads,
        r.spec.value_heads,
        r.spec.value_heads / r.spec.qk_heads,
        r.spec.conv_channels,
        r.spec.width,
        r.spec.record_capacity,
        row_begin,
        0};
    for (std::int64_t begin = 0; begin < r.spec.layers; begin += kGridExtent) {
        a.layer_begin     = static_cast<int>(begin);
        const auto layers = std::min<std::int64_t>(kGridExtent, r.spec.layers - begin);
        const auto submit = [&]<int Warps>() {
            constexpr int tiles = kStateDim / (Warps * kRowsPerWarp);
            const dim3 grid(unsigned(r.spec.value_heads) * tiles, layers, batch);
            linear_fold_kernel<Rows, Warps, StateT>
                <<<grid, common::kWarpSize * Warps, 0, execution.stream()>>>(a, rows);
        };
        const int warps =
            select_fold_warps(layers, r.spec.value_heads, batch, execution.multiprocessor_count());
        if (warps == 16)
            submit.template operator()<16>();
        else if (warps == 8)
            submit.template operator()<8>();
        else
            submit.template operator()<4>();
        CUDA_CHECK(cudaGetLastError());
    }
}

template <class Rows>
void launch(const KdaReplayRecords& r, LinearAttentionStateAllLayersView s, const Rows& rows,
            int row_begin, int batch, CudaExecutionView execution) {
    switch (s.recurrent.field.tensor.dtype) {
    case DType::FP32:
        launch_typed<Rows, float>(r, s, rows, row_begin, batch, execution);
        return;
    case DType::BF16:
        launch_typed<Rows, __nv_bfloat16>(r, s, rows, row_begin, batch, execution);
        return;
    default:
        throw std::logic_error("kda_replay_fold: unsupported recurrent state dtype");
    }
}
} // namespace

void launch_replay_fold(const KdaReplayRecords& r, LinearAttentionStateAllLayersView s,
                        const KdaReplayFoldKernelRows& rows, int row_begin, int batch,
                        CudaExecutionView execution) {
    launch(r, s, rows, row_begin, batch, execution);
}

void launch_replay_fold(const KdaReplayRecords& r, LinearAttentionStateAllLayersView s,
                        const KdaReplayFoldDeviceRows& rows, int batch,
                        CudaExecutionView execution) {
    for (std::int64_t first = 0; first < batch; first += kGridExtent) {
        const KdaReplayFoldDeviceRows group{rows.sources + first, rows.destinations + first,
                                            rows.columns + first};
        launch(r, s, group, static_cast<int>(first),
               std::min<std::int64_t>(kGridExtent, batch - first), execution);
    }
}
} // namespace ninfer::ops::detail::kimi_delta_attention
