#pragma once

#include "ops/common/cuda/reduce.cuh"
#include "ops/common/cuda/memory.cuh"
#include "ops/linear_attention/gated_delta_net/kernels/transition.cuh"

namespace ninfer::ops::detail::gated_delta_net {
struct RecordEffects {
    template <class Access>
    __device__ __forceinline__ static void observe_key(const Access& access,
                                                       const RecurrentCoordinates& coord,
                                                       std::int32_t token, const RawQkLane& key) {
        access.store_key(coord, token, key);
    }

    template <class Access>
    __device__ __forceinline__ static void
    observe_value_gate(const Access& access, const RecurrentCoordinates& coord, std::int32_t token,
                       const RawValueLane& value, const RawGatePair& gate) {
        access.store_value(coord, token, value);
        access.store_gate(coord, token, gate);
    }

    template <bool NormalizeInputs, class Access>
    __device__ __forceinline__ static void
    publish_output(float (&state)[kDvPerWarp][kQkPerLane], const Access& access,
                   const RecurrentCoordinates& coord, std::int32_t token) {
        OutputEffects::publish_output<NormalizeInputs>(state, access, coord, token);
    }
};

template <bool Masked>
struct RecordAccess {
    const __nv_bfloat16* q;
    const __nv_bfloat16* k;
    const __nv_bfloat16* v;
    const float* g;
    const float* beta;
    const float* states;
    const std::int32_t* valid_columns;
    const std::int64_t* initial_offsets;
    __nv_bfloat16* key_record;
    __nv_bfloat16* value_record;
    uint2* gate_record;
    __nv_bfloat16* out;
    HeadMap heads;
    std::int32_t width;
    float scale;

    __device__ __forceinline__ RecurrentCoordinates coordinates() const {
        return make_coordinates(static_cast<std::int32_t>(blockIdx.y), 0,
                                static_cast<std::int32_t>(blockIdx.z), heads);
    }

    __device__ __forceinline__ std::int32_t
    active_columns(const RecurrentCoordinates& coord) const {
        if constexpr (Masked) { return valid_columns[coord.batch]; }
        return width;
    }

    __device__ __forceinline__ std::int64_t column(const RecurrentCoordinates& coord,
                                                   std::int32_t token) const {
        return static_cast<std::int64_t>(coord.batch) * width + token;
    }

    __device__ __forceinline__ const float*
    state_read_base(const RecurrentCoordinates& coord) const {
        return states + initial_offsets[coord.batch] / sizeof(float) +
               static_cast<std::int64_t>(coord.value_head) * kStateDim * kStateDim;
    }

    __device__ __forceinline__ const __nv_bfloat16* key_ptr(const RecurrentCoordinates& coord,
                                                            std::int32_t token) const {
        return k + (column(coord, token) * heads.H_qk + coord.qk_head) * kStateDim;
    }

    __device__ __forceinline__ const __nv_bfloat16* value_ptr(const RecurrentCoordinates& coord,
                                                              std::int32_t token) const {
        return v + (column(coord, token) * heads.H_v + coord.value_head) * kStateDim;
    }

    __device__ __forceinline__ RawGatePair load_gate(const RecurrentCoordinates& coord,
                                                     std::int32_t token) const {
        return load_source_gate(g, beta, column(coord, token) * heads.H_v + coord.value_head);
    }

    __device__ __forceinline__ const __nv_bfloat16* query_ptr(const RecurrentCoordinates& coord,
                                                              std::int32_t token) const {
        return q + (column(coord, token) * heads.H_qk + coord.qk_head) * kStateDim;
    }

    __device__ __forceinline__ __nv_bfloat16* output_ptr(const RecurrentCoordinates& coord,
                                                         std::int32_t token) const {
        return out + (column(coord, token) * heads.H_v + coord.value_head) * kStateDim;
    }

    __device__ __forceinline__ void store_key(const RecurrentCoordinates& coord, std::int32_t token,
                                              const RawQkLane& raw) const {
        store_key_chunk(coord, token, coord.dqk_base, raw);
    }

    __device__ __forceinline__ void store_key_chunk(const RecurrentCoordinates& coord,
                                                    std::int32_t token, int key_column,
                                                    const RawQkLane& raw) const {
        if (coord.state_tile == 0 && coord.warp == 0 &&
            static_cast<int>(coord.value_head) % heads.group_size() == 0) {
            __nv_bfloat16* destination =
                key_record + (column(coord, token) * heads.H_qk + coord.qk_head) * kStateDim;
            common::store_vec(destination + key_column, raw.bits);
        }
    }

    __device__ __forceinline__ void store_value(const RecurrentCoordinates& coord,
                                                std::int32_t token, const RawValueLane& raw) const {
        if (coord.lane < kDvPerWarp) {
            store_value_element(coord, token, coord.dv_base + coord.lane, raw.bits);
        }
    }

    __device__ __forceinline__ void store_value_element(const RecurrentCoordinates& coord,
                                                        std::int32_t token, int row,
                                                        __nv_bfloat16 bits) const {
        value_record[(column(coord, token) * heads.H_v + coord.value_head) * kStateDim + row] =
            bits;
    }

    __device__ __forceinline__ void store_gate(const RecurrentCoordinates& coord,
                                               std::int32_t token, const RawGatePair& raw) const {
        if (coord.state_tile == 0 && coord.warp == 0 && coord.lane == 0) {
            gate_record[column(coord, token) * heads.H_v + coord.value_head] = raw.bits;
        }
    }
};

template <bool Masked>
__global__ void __launch_bounds__(common::kWarpSize* kNumWarps, 2)
    recurrent_record_kernel(RecordAccess<Masked> access) {
    const RecurrentCoordinates coord = access.coordinates();
    const std::int32_t valid         = access.active_columns(coord);
    __align__(16) float state[kDvPerWarp][kQkPerLane];
    load_state_tile(state, access.state_read_base(coord), coord);
    run_recurrent_sequence<true, RecordEffects>(state, access, coord, valid);
    zero_output_suffix(access, coord, valid, access.width);
}

} // namespace ninfer::ops::detail::gated_delta_net
