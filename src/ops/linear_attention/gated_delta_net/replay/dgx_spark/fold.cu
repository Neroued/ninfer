#include "ops/linear_attention/gated_delta_net/replay/fold_io.cuh"

namespace ninfer::ops::detail::gated_delta_net {
namespace {
constexpr int kFoldWarps = 8;
constexpr int kFoldTiles = 4;
constexpr int kFoldLanes = 8;

template <bool Sequential, int TileRows>
__device__ __forceinline__ void fold_subgroup(float4 (&state)[32 / kFoldLanes],
                                              const FoldWindow<TileRows>& sm, int subgroup,
                                              int lane, int commit) {
    constexpr int Vectors = 32 / kFoldLanes;
    if constexpr (!Sequential) {
#pragma unroll
        for (int p = 0; p < Vectors; ++p) state[p] = scale_state(state[p], sm.total_decay);
    }
    for (int t = 0; t < commit; ++t) {
        const float delta = sm.delta[t][subgroup];
#pragma unroll
        for (int p = 0; p < Vectors; ++p) {
            const auto key =
                common::load_vec<float4>(sm.scaled_key[t] + (p * kFoldLanes + lane) * 4);
            if constexpr (Sequential)
                state[p] = apply_transition(state[p], sm.alpha[t], delta, key);
            else
                state[p] = add_outer(state[p], delta, key);
        }
    }
}

template <class Geometry, class Rows, class StateT>
__global__ __launch_bounds__(common::kWarpSize* kFoldWarps, 2) void subgroup_fold_kernel(
    const __grid_constant__ FoldAccess<Geometry, Rows, StateT> access) {
    constexpr int Threads  = common::kWarpSize * kFoldWarps;
    constexpr int TileRows = kStateDim / kFoldTiles;
    constexpr int Vectors  = 32 / kFoldLanes;
    static_assert(TileRows == Threads / kFoldLanes);
    const int tid = threadIdx.x, lane = tid % kFoldLanes, subgroup = tid / kFoldLanes;
    const int head = blockIdx.x / kFoldTiles, tile = blockIdx.x % kFoldTiles;
    const int batch = blockIdx.y, layer = blockIdx.z;
    const int commit = fold_columns(access.rows, batch);
    if (commit == 0) return;
    const auto row = fold_row(access.rows, batch, commit);
    const auto column =
        (std::int64_t(layer) * access.record_capacity + access.record_begin + batch) * access.width;
    auto* layer_state    = access.recurrent_layer0 + layer * access.recurrent_layer_stride;
    const int offset     = (head * kStateDim + tile * TileRows + subgroup) * kStateDim;
    const StateT* source = layer_state +
                           row.source_offset / static_cast<std::int64_t>(sizeof(StateT)) + offset;
    StateT* destination = layer_state +
                          row.destination_offset / static_cast<std::int64_t>(sizeof(StateT)) +
                          offset;
    __shared__ FoldWindow<TileRows> sm;
    // With at most 16 columns, one 16-byte copy per participating thread stages the tile.
    if (tid < commit * (TileRows / 4)) {
        const int t = tid / (TileRows / 4), col = tid % (TileRows / 4) * 4;
        common::cp_async<16, Cache::cg>(sm.delta[t] + col,
                                        access.delta_ptr(column + t, head) + tile * TileRows + col);
    }
    common::cp_commit();
    float4 state[Vectors];
#pragma unroll
    for (int p = 0; p < Vectors; ++p)
        state[p] = common::load_state_float4(source + (p * kFoldLanes + lane) * 4);

    if (tid < 32) stage_decays(sm, access, column, head, tid, commit);
    if (subgroup < commit) {
        float4 keys[Vectors];
        float sum = 0.f;
#pragma unroll
        for (int p = 0; p < Vectors; ++p) {
            const auto raw = load_raw_qk_lane(access.key_ptr(column + subgroup, head),
                                              (p * kFoldLanes + lane) * 4);
            keys[p]        = make_float4(raw.value[0], raw.value[1], raw.value[2], raw.value[3]);
#pragma unroll
            for (float v : raw.value) sum += v * v;
        }
        const unsigned mask      = 0xffU << (subgroup % Vectors * kFoldLanes);
        sum                      = common::warp_sum<kFoldLanes>(sum, mask);
        const float inverse_norm = rsqrtf(sum + kQkL2NormEps);
#pragma unroll
        for (int p = 0; p < Vectors; ++p)
            common::store_vec(sm.scaled_key[subgroup] + (p * kFoldLanes + lane) * 4,
                              scale_state(keys[p], inverse_norm));
    }
    common::cp_wait<0>();
    __syncthreads();

    const bool fallback = sm.fallback;
    if (!fallback && commit > 1) {
        if (subgroup < commit) {
            const float suffix = sm.suffix[subgroup];
#pragma unroll
            for (int p = 0; p < Vectors; ++p) {
                auto* key = sm.scaled_key[subgroup] + (p * kFoldLanes + lane) * 4;
                common::store_vec(key, scale_state(common::load_vec<float4>(key), suffix));
            }
        }
        __syncthreads();
    }

    if (fallback)
        fold_subgroup<true>(state, sm, subgroup, lane, commit);
    else
        fold_subgroup<false>(state, sm, subgroup, lane, commit);
#pragma unroll
    for (int p = 0; p < Vectors; ++p)
        common::store_state_float4(destination + (p * kFoldLanes + lane) * 4, state[p]);
    if (tid < 32 && tile == 0) access.publish_conv(layer, head, tid, commit, column, row);
}

struct FoldLauncher {
    template <class Geometry, class Rows, class StateT>
    static void launch(const FoldAccess<Geometry, Rows, StateT>& access, int active_rows,
                       cudaStream_t stream) {
        // Use four independently streamed tiles per head with ordinary vector state I/O.
        const dim3 grid(Geometry::kValueHeads * kFoldTiles, active_rows, Geometry::kLayers);
        subgroup_fold_kernel<<<grid, common::kWarpSize * kFoldWarps, 0, stream>>>(access);
    }
};
} // namespace

void launch_replay_fold(const GdnReplayRecords& records, LinearAttentionStateAllLayersView states,
                        const GdnReplayFoldKernelRows& rows, std::int32_t active_rows,
                        std::int32_t record_begin, cudaStream_t stream) {
    dispatch_replay_fold<FoldLauncher>(records, states, rows, active_rows, record_begin, stream);
}

void launch_replay_fold(const GdnReplayRecords& records, LinearAttentionStateAllLayersView states,
                        const GdnReplayFoldDeviceRows& rows, std::int32_t active_rows,
                        std::int32_t record_begin, cudaStream_t stream) {
    dispatch_replay_fold<FoldLauncher>(records, states, rows, active_rows, record_begin, stream);
}

} // namespace ninfer::ops::detail::gated_delta_net
