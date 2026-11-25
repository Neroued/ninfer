#include "ops/linear_attention/gated_delta_net/replay/fold_io.cuh"

namespace ninfer::ops::detail::gated_delta_net {
namespace {
constexpr int kFoldWarps = 8;

template <class StateT>
__device__ __forceinline__ float4 load_state_row(const StateT* state, int row, int col) {
    // Keep prefetched rows resident through the compute pass and possible in-place store.
    return common::load_state_float4_l2_retained(state + row * kStateDim + col);
}

template <class StateT>
__device__ __forceinline__ void store_state_row(StateT* state, int row, int col, float4 value) {
    // A fold streams the state once. Prefer early eviction of completed state rows.
    common::store_state_float4_streaming(state + row * kStateDim + col, value);
}

template <int RowsPerPass, bool Sequential, class StateT>
__device__ __forceinline__ void fold_window(float4 (&initial)[RowsPerPass],
                                            const FoldWindow<kStateDim>& sm, const StateT* source,
                                            StateT* destination, int warp, int col, int commit) {
#pragma unroll 1
    for (int base = warp; base < kStateDim; base += kFoldWarps * RowsPerPass) {
        float4 acc[RowsPerPass];
        // Issue the next pass's loads before the correction loop.
        float4 next[RowsPerPass];
        if (base + kFoldWarps * RowsPerPass < kStateDim) {
#pragma unroll
            for (int r = 0; r < RowsPerPass; ++r)
                next[r] = load_state_row(source, base + (RowsPerPass + r) * kFoldWarps, col);
        }
#pragma unroll
        for (int r = 0; r < RowsPerPass; ++r)
            acc[r] = Sequential ? initial[r] : scale_state(initial[r], sm.total_decay);
        for (int t = 0; t < commit; ++t) {
            const auto key = common::load_vec<float4>(sm.scaled_key[t] + col);
#pragma unroll
            for (int r = 0; r < RowsPerPass; ++r) {
                const float delta = sm.delta[t][base + r * kFoldWarps];
                if constexpr (Sequential)
                    acc[r] = apply_transition(acc[r], sm.alpha[t], delta, key);
                else
                    acc[r] = add_outer(acc[r], delta, key);
            }
        }
#pragma unroll
        for (int r = 0; r < RowsPerPass; ++r) {
            store_state_row(destination, base + r * kFoldWarps, col, acc[r]);
            if (base + kFoldWarps * RowsPerPass < kStateDim) initial[r] = next[r];
        }
    }
}

template <int RowsPerPass, class Geometry, class Rows, class StateT>
__global__ __launch_bounds__(common::kWarpSize* kFoldWarps, 4) void linear_fold_kernel(
    const __grid_constant__ FoldAccess<Geometry, Rows, StateT> access) {
    static_assert(kStateDim % (kFoldWarps * RowsPerPass) == 0);
    const int tid = threadIdx.x, lane = tid % common::kWarpSize, warp = tid / common::kWarpSize;
    const int head = blockIdx.x, batch = blockIdx.z, layer = blockIdx.y;
    const int commit = fold_columns(access.rows, batch);
    if (commit == 0) return;
    const auto row = fold_row(access.rows, batch, commit);
    const auto column =
        (std::int64_t(layer) * access.record_capacity + access.record_begin + batch) * access.width;
    const auto* layer_state = access.recurrent_layer0 + layer * access.recurrent_layer_stride;
    const StateT* source    = layer_state +
                           row.source_offset / static_cast<std::int64_t>(sizeof(StateT)) +
                           head * kStateDim * kStateDim;
    StateT* destination = access.recurrent_layer0 + layer * access.recurrent_layer_stride +
                          row.destination_offset / static_cast<std::int64_t>(sizeof(StateT)) +
                          head * kStateDim * kStateDim;
    const int col = lane * 4;
    __shared__ FoldWindow<kStateDim> sm;

    if (commit == 1) {
        if (warp == 0) {
            auto key = load_raw_qk_lane(access.key_ptr(column, head), col);
            normalize_qk_lane<true>(key.value, lane);
            common::store_vec(sm.scaled_key[0] + col,
                              make_float4(key.value[0], key.value[1], key.value[2], key.value[3]));
            common::cp_async<16, Cache::cg>(sm.delta[0] + col,
                                            access.delta_ptr(column, head) + col);
            common::cp_commit();
            if (lane == 0) sm.total_decay = access.alpha(column, head);
            common::cp_wait<0>();
        }
        float4 initial[RowsPerPass];
#pragma unroll
        for (int r = 0; r < RowsPerPass; ++r)
            initial[r] = load_state_row(source, warp + r * kFoldWarps, col);
        __syncthreads();
        const auto key    = common::load_vec<float4>(sm.scaled_key[0] + col);
        const float decay = sm.total_decay;
#pragma unroll 1
        for (int base = warp; base < kStateDim; base += kFoldWarps * RowsPerPass) {
#pragma unroll
            for (int r = 0; r < RowsPerPass; ++r) {
                const int dv      = base + r * kFoldWarps;
                const auto result = add_outer(scale_state(initial[r], decay), sm.delta[0][dv], key);
                if (dv + kFoldWarps * RowsPerPass < kStateDim)
                    initial[r] = load_state_row(source, dv + kFoldWarps * RowsPerPass, col);
                store_state_row(destination, dv, col, result);
            }
        }
    } else {
        for (int t = warp; t < commit; t += kFoldWarps)
            common::cp_async<16, Cache::cg>(sm.delta[t] + col,
                                            access.delta_ptr(column + t, head) + col);
        common::cp_commit();
        if (warp == 0) stage_decays(sm, access, column, head, lane, commit);
        float4 initial[RowsPerPass];
#pragma unroll
        for (int r = 0; r < RowsPerPass; ++r)
            initial[r] = load_state_row(source, warp + r * kFoldWarps, col);
        __syncthreads();
        const bool fallback = sm.fallback;
        for (int t = warp; t < commit; t += kFoldWarps) {
            auto key = load_raw_qk_lane(access.key_ptr(column + t, head), col);
            normalize_qk_lane<true>(key.value, lane);
            const float suffix = fallback ? 1.f : sm.suffix[t];
            common::store_vec(sm.scaled_key[t] + col,
                              make_float4(key.value[0] * suffix, key.value[1] * suffix,
                                          key.value[2] * suffix, key.value[3] * suffix));
        }
        common::cp_wait<0>();
        __syncthreads();
        if (fallback)
            fold_window<RowsPerPass, true>(initial, sm, source, destination, warp, col, commit);
        else
            fold_window<RowsPerPass, false>(initial, sm, source, destination, warp, col, commit);
    }
    if (warp == 0) access.publish_conv(layer, head, lane, commit, column, row);
}

struct FoldLauncher {
    template <class Geometry, class Rows, class StateT>
    static void launch(const FoldAccess<Geometry, Rows, StateT>& access, int active_rows,
                       cudaStream_t stream) {
        // Two rows per pass improve the 36-layer, 24-head working set.
        constexpr int RowsPerPass = Geometry::kValueHeads == 24 ? 2 : 4;
        const dim3 grid(Geometry::kValueHeads, Geometry::kLayers, active_rows);
        linear_fold_kernel<RowsPerPass>
            <<<grid, common::kWarpSize * kFoldWarps, 0, stream>>>(access);
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
