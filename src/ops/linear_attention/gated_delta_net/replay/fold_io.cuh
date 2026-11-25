#pragma once

#include <cfloat>

#include "ops/common/cuda/pipeline.cuh"
#include "ops/common/host/operands.h"
#include "ops/linear_attention/gated_delta_net/kernels/transition.cuh"
#include "ops/linear_attention/gated_delta_net/replay/launch.h"

namespace ninfer::ops::detail::gated_delta_net {
inline constexpr int kFoldWidth = 16;

template <int Layers, int QkHeads, int ValueHeads, int ConvChannels>
struct FoldGeometry {
    static constexpr int kLayers       = Layers;
    static constexpr int kQkHeads      = QkHeads;
    static constexpr int kValueHeads   = ValueHeads;
    static constexpr int kConvChannels = ConvChannels;
};

__device__ __forceinline__ GdnReplayFoldKernelRow fold_row(const GdnReplayFoldKernelRows& rows,
                                                           int b, int) {
    return rows.row[b];
}

__device__ __forceinline__ GdnReplayFoldKernelRow fold_row(const GdnReplayFoldDeviceRows& rows,
                                                           int b, int commit) {
    return {rows.sources[b], rows.destinations[b], commit, 0};
}

__device__ __forceinline__ int fold_columns(const GdnReplayFoldKernelRows& rows, int b) {
    return rows.row[b].commit_columns;
}

__device__ __forceinline__ int fold_columns(const GdnReplayFoldDeviceRows& rows, int b) {
    return rows.columns[b];
}

template <class Geometry, class Rows, class StateT>
struct FoldAccess {
    const __nv_bfloat16* key_record;
    const float* delta_record;
    const float* alpha_record;
    const __nv_bfloat16* conv_record;
    StateT* recurrent_layer0;
    __nv_bfloat16* conv_layer0;
    std::int64_t recurrent_layer_stride;
    std::int64_t conv_layer_stride;
    int record_capacity;
    int width;
    Rows rows;
    int record_begin;

    __device__ __forceinline__ const __nv_bfloat16* key_ptr(std::int64_t column, int head) const {
        constexpr int group = Geometry::kValueHeads / Geometry::kQkHeads;
        return key_record + (column * Geometry::kQkHeads + head / group) * kStateDim;
    }

    __device__ __forceinline__ const float* delta_ptr(std::int64_t column, int head) const {
        return delta_record + (column * Geometry::kValueHeads + head) * kStateDim;
    }

    __device__ __forceinline__ float alpha(std::int64_t column, int head) const {
        return alpha_record[column * Geometry::kValueHeads + head];
    }

    __device__ __forceinline__ void publish_conv(int layer, int head, int lane, int commit,
                                                 std::int64_t column,
                                                 GdnReplayFoldKernelRow row) const {
        constexpr int channels         = Geometry::kConvChannels;
        constexpr int vector_channels  = sizeof(uint4) / sizeof(__nv_bfloat16);
        constexpr int channels_per_cta = common::kWarpSize * vector_channels;
        static_assert(channels % vector_channels == 0);
        const auto* conv = conv_layer0 + layer * conv_layer_stride;
        for (int c = head * channels_per_cta + lane * vector_channels; c < channels;
             c += Geometry::kValueHeads * channels_per_cta) {
            auto* destination = conv_layer0 + layer * conv_layer_stride +
                                row.destination_offset / sizeof(__nv_bfloat16) + c;
            const auto* source = conv + row.source_offset / sizeof(__nv_bfloat16) + c;
            const auto* record = conv_record + column * channels + c;
            // Copy the tail of the concatenated history. Each channel has one writer and
            // the left shift reads a source word before that word can be overwritten in place.
#pragma unroll
            for (int i = 0; i < 3; ++i) {
                const int index = commit + i;
                const auto* from =
                    index < 3 ? source + index * channels : record + (index - 3) * channels;
                common::store_vec(destination + i * channels, common::load_vec<uint4>(from));
            }
        }
    }
};

template <int TileRows>
struct alignas(16) FoldWindow {
    float delta[kFoldWidth][TileRows];
    float scaled_key[kFoldWidth][kStateDim];
    float suffix[kFoldWidth];
    float alpha[kFoldWidth];
    float total_decay;
    bool fallback;
};

__device__ __forceinline__ float4 add_outer(float4 acc, float delta, float4 key) {
    return {fmaf(delta, key.x, acc.x), fmaf(delta, key.y, acc.y), fmaf(delta, key.z, acc.z),
            fmaf(delta, key.w, acc.w)};
}

__device__ __forceinline__ float4 scale_state(float4 state, float decay) {
    return {state.x * decay, state.y * decay, state.z * decay, state.w * decay};
}

__device__ __forceinline__ float4 apply_transition(float4 state, float alpha, float delta,
                                                   float4 key) {
    return {fmaf(alpha, state.x, delta * key.x), fmaf(alpha, state.y, delta * key.y),
            fmaf(alpha, state.z, delta * key.z), fmaf(alpha, state.w, delta * key.w)};
}

// Form suffix products of the represented decays. A represented zero still resets
// state directly. If a product leaves the normal FP32 range, retain the unscaled
// keys and apply the individual transitions instead.
template <int TileRows, class Access>
__device__ __forceinline__ void stage_decays(FoldWindow<TileRows>& sm, const Access& access,
                                             std::int64_t column, int head, int lane, int commit) {
    const float alpha = lane < commit ? access.alpha(column + lane, head) : 1.f;
    if (lane < commit) sm.alpha[lane] = alpha;
    float decay   = 1.f;
    bool fallback = false;
    for (int t = commit - 1; t >= 0; --t) {
        if (lane == t) sm.suffix[t] = decay;
        const float a       = __shfl_sync(common::kFullWarpMask, alpha, t);
        const float product = decay * a;
        fallback |= !isfinite(product) || (product < FLT_MIN && decay > 0.f && a > 0.f);
        decay = product;
    }
    if (lane == 0) {
        sm.total_decay = decay;
        sm.fallback    = fallback;
    }
}

template <class Launcher, class Geometry, class Rows, class StateT>
void launch_replay_fold_fixed(const GdnReplayRecords& records,
                              LinearAttentionStateAllLayersView states, const Rows& rows,
                              int active_rows, int record_begin, cudaStream_t stream) {
    const auto [key, delta, alpha] = ninfer::detail::GdnReplayRecordAccess::planes(records);
    const FoldAccess<Geometry, Rows, StateT> access{
        static_cast<const __nv_bfloat16*>(key.data),
        static_cast<const float*>(delta.data),
        static_cast<const float*>(alpha.data),
        static_cast<const __nv_bfloat16*>(records.conv.data),
        common::operand::state_field_base<StateT>(states.recurrent),
        common::operand::state_field_base<__nv_bfloat16>(states.conv),
        static_cast<std::int64_t>(states.recurrent.field.layer_stride / sizeof(StateT)),
        static_cast<std::int64_t>(states.conv.field.layer_stride / sizeof(__nv_bfloat16)),
        records.spec.record_capacity,
        records.spec.width,
        rows,
        record_begin,
    };
    Launcher::launch(access, active_rows, stream);
    CUDA_CHECK(cudaGetLastError());
}

template <class Launcher, class Rows>
void dispatch_replay_fold(const GdnReplayRecords& records, LinearAttentionStateAllLayersView states,
                          const Rows& rows, int active_rows, int record_begin,
                          cudaStream_t stream) {
    const auto dispatch = [&]<class StateT>() {
        switch (select_fold_geometry(records.spec)) {
        case FoldGeometryId::Layers48Heads48:
            return launch_replay_fold_fixed<Launcher, FoldGeometry<48, 16, 48, 10240>, Rows,
                                            StateT>(records, states, rows, active_rows,
                                                    record_begin, stream);
        case FoldGeometryId::Layers30Heads32:
            return launch_replay_fold_fixed<Launcher, FoldGeometry<30, 16, 32, 8192>, Rows, StateT>(
                records, states, rows, active_rows, record_begin, stream);
        case FoldGeometryId::Layers36Heads48:
            return launch_replay_fold_fixed<Launcher, FoldGeometry<36, 16, 48, 10240>, Rows,
                                            StateT>(records, states, rows, active_rows,
                                                    record_begin, stream);
        case FoldGeometryId::Layers36Heads24:
            return launch_replay_fold_fixed<Launcher, FoldGeometry<36, 8, 24, 5120>, Rows, StateT>(
                records, states, rows, active_rows, record_begin, stream);
        }
    };
    if (states.recurrent.field.tensor.dtype == DType::BF16)
        dispatch.template operator()<__nv_bfloat16>();
    else
        dispatch.template operator()<float>();
}

} // namespace ninfer::ops::detail::gated_delta_net
