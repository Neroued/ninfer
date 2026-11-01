#pragma once

#include "ops/linear_attention/gated_delta_net/kernels/transition.cuh"
#include "ops/common/cuda/pipeline.cuh"

namespace ninfer::ops::detail::gated_delta_net {
namespace subgroup {

inline constexpr int kWindowSize = 16;

enum class Mode { Update, Record, Fold };

template <int Capacity>
struct alignas(16) Window {
    __nv_bfloat16 key[Capacity][kStateDim];
    __nv_bfloat16 query[Capacity][kStateDim];
    __nv_bfloat16 value[Capacity][kBlockDv];
    float gate[Capacity], beta[Capacity];
};

template <bool Readout, int Capacity, class Access>
__device__ __forceinline__ void prefetch(Window<Capacity>& sm, const Access& a,
                                         const RecurrentCoordinates& c, int valid) {
    const int tid = c.warp * 32 + c.lane;
    for (int i = tid; i < valid * 16; i += common::kWarpSize * kNumWarps) {
        const int t = i / 16, col = (i % 16) * 8;
        common::cp_async<16, Cache::cg>(&sm.key[t][col], a.key_ptr(c, t) + col);
        if constexpr (Readout)
            common::cp_async<16, Cache::cg>(&sm.query[t][col], a.query_ptr(c, t) + col);
    }
    for (int i = tid; i < valid * (kBlockDv / 8); i += common::kWarpSize * kNumWarps) {
        const int t = i / (kBlockDv / 8), col = i % (kBlockDv / 8) * 8;
        common::cp_async<16, Cache::cg>(&sm.value[t][col],
                                        a.value_ptr(c, t) + c.state_tile * kBlockDv + col);
    }
    if (tid < valid) {
        const auto g = a.load_gate(c, tid);
        sm.gate[tid] = g.g;
        sm.beta[tid] = g.beta;
    }
    common::cp_commit();
}

template <int Lanes, bool Normalize>
__device__ __forceinline__ void normalize(float (&x)[kStateDim / Lanes]) {
    if constexpr (Normalize) {
        float sum = 0;
#pragma unroll
        for (int i = 0; i < kStateDim / Lanes; ++i) sum += x[i] * x[i];
        sum             = common::warp_sum<Lanes>(sum);
        const float inv = rsqrtf(sum + kQkL2NormEps);
#pragma unroll
        for (int i = 0; i < kStateDim / Lanes; ++i) x[i] *= inv;
    }
}

// A subgroup owns consecutive state rows and coalesced four-key segments. Keep its transition
// arithmetic aligned with replay fold's exact snapshot contract.
template <int Lanes, bool Normalize, Mode Behavior, bool Prefetch, int Capacity, class Access>
__device__ __forceinline__ void run(const Access& a, RecurrentCoordinates original, int valid,
                                    Window<Capacity>* sm) {
    static_assert(Lanes == 8 || Lanes == 16);
    constexpr bool Readout    = Behavior != Mode::Fold;
    constexpr bool Record     = Behavior == Mode::Record;
    constexpr bool WriteState = Behavior != Mode::Record;
    constexpr int Rows        = Lanes / 8;
    constexpr int Cols        = kStateDim / Lanes;
    static_assert((common::kWarpSize / Lanes) * Rows == kDvPerWarp);
    const int lane  = original.lane % Lanes;
    const int group = original.lane / Lanes;
    auto c          = original;
    c.dv_base += group * Rows;
    if constexpr (Prefetch) prefetch<Readout>(*sm, a, original, valid);
    __align__(16) float state[Rows][Cols];
    const float* source = a.state_read_base(c);
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
#pragma unroll
        for (int p = 0; p < Cols / 4; ++p) {
            const int col  = lane * 4 + p * Lanes * 4;
            const float4 x = common::load_vec<float4>(source + (c.dv_base + r) * kStateDim + col);
            common::store_vec(state[r] + p * 4, x);
        }
    }
    if constexpr (Prefetch) {
        common::cp_wait<0>();
        __syncthreads();
    }
    for (int t = 0; t < valid; ++t) {
        float key[Cols];
        const __nv_bfloat16* key_ptr;
        if constexpr (Prefetch)
            key_ptr = sm->key[t];
        else
            key_ptr = a.key_ptr(c, t);
#pragma unroll
        for (int p = 0; p < Cols / 4; ++p) {
            const int col  = lane * 4 + p * Lanes * 4;
            const auto raw = load_raw_qk_lane(key_ptr, col);
#pragma unroll
            for (int j = 0; j < 4; ++j) key[p * 4 + j] = raw.value[j];
            if constexpr (Record) {
                if (group == 0) a.store_key_chunk(original, t, col, raw);
            }
        }
        normalize<Lanes, Normalize>(key);
        RawGatePair gate;
        if constexpr (Prefetch) {
            const float g = sm->gate[t], beta = sm->beta[t];
            gate = {make_uint2(__float_as_uint(g), __float_as_uint(beta)), g, beta};
        } else
            gate = a.load_gate(c, t);
        const float alpha = expf(gate.g);
        if constexpr (Record) a.store_gate(original, t, gate);
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
            __nv_bfloat16 vb;
            if constexpr (Prefetch)
                vb = sm->value[t][c.dv_base % kBlockDv + r];
            else
                vb = a.value_ptr(c, t)[c.dv_base + r];
            const float value = __bfloat162float(vb);
            if constexpr (Record) {
                if (lane == r) a.store_value_element(c, t, c.dv_base + r, vb);
            }
            float dot = 0;
#pragma unroll
            for (int j = 0; j < Cols; ++j) dot += state[r][j] * key[j];
            dot               = common::warp_sum<Lanes>(dot);
            const float delta = gate.beta * (value - alpha * dot);
#pragma unroll
            for (int j = 0; j < Cols; ++j) state[r][j] = alpha * state[r][j] + delta * key[j];
        }
        if constexpr (Readout) {
            float query[Cols];
            const __nv_bfloat16* query_ptr;
            if constexpr (Prefetch)
                query_ptr = sm->query[t];
            else
                query_ptr = a.query_ptr(c, t);
#pragma unroll
            for (int p = 0; p < Cols / 4; ++p) {
                const auto raw = load_raw_qk_lane(query_ptr, lane * 4 + p * Lanes * 4);
#pragma unroll
                for (int j = 0; j < 4; ++j) query[p * 4 + j] = raw.value[j];
            }
            normalize<Lanes, Normalize>(query);
#pragma unroll
            for (int r = 0; r < Rows; ++r) {
                float dot = 0;
#pragma unroll
                for (int j = 0; j < Cols; ++j) dot += state[r][j] * query[j];
                dot = common::warp_sum<Lanes>(dot);
                if (lane == r)
                    a.output_ptr(c, t)[c.dv_base + r] = __float2bfloat16_rn(dot * a.scale);
            }
        }
    }
    if constexpr (WriteState) {
        float* destination = a.state_write_base(c);
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
#pragma unroll
            for (int p = 0; p < Cols / 4; ++p) {
                const int col = lane * 4 + p * Lanes * 4;
                common::store_vec(destination + (c.dv_base + r) * kStateDim + col,
                                  common::load_vec<float4>(state[r] + p * 4));
            }
        }
    }
    if constexpr (Record) {
        for (int t = valid; t < a.width; ++t) {
#pragma unroll
            for (int r = 0; r < Rows; ++r)
                if (lane == r) a.output_ptr(c, t)[c.dv_base + r] = __float2bfloat16(0);
        }
    }
}

} // namespace subgroup
} // namespace ninfer::ops::detail::gated_delta_net
