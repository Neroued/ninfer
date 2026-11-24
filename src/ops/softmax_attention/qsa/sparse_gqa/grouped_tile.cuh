#pragma once

#include "ops/softmax_attention/qsa/sparse_gqa/output_tile.cuh"

namespace ninfer::ops::detail::qsa {

template <KvCacheStorage Storage, int N, int Warps, bool Vector, SparseGqaLoad Load>
using SparseGqaGroupedBase =
    SparseGqaOutputTile<Storage, N, Warps, Vector, Load,
                        Vector && Load == SparseGqaLoad::Staged &&
                            SparseGqaCodec<Storage, true>::format ==
                                KvCacheVectorFormat::Fp8E4m3fnGroup128Ue8m0>;

template <KvCacheStorage Storage, int N, int Warps, bool Vector, SparseGqaLoad Load>
struct SparseGqaGroupedTile : SparseGqaGroupedBase<Storage, N, Warps, Vector, Load> {
    using Parent                       = SparseGqaGroupedBase<Storage, N, Warps, Vector, Load>;
    using Base                         = typename Parent::Base;
    static constexpr bool joint_scales = false;

    struct State {
        unsigned key = 127, value = 127;
        bool single_factor = true;
    };

    __device__ __forceinline__ State prefetch(const SparseGqaArguments& a, int start, int end) {
        State state;
        state.key = Parent::template prefetch<true>(a, start, end);
        return state;
    }

    __device__ __forceinline__ void wait_key(State& state) {
        state.single_factor = Parent::template wait<true>(state.key);
    }

    __device__ __forceinline__ void stage_value(State& state, const SparseGqaArguments& a,
                                                int start, int end) {
        state.value = Parent::template prefetch<false>(a, start, end);
    }

    __device__ __forceinline__ void wait_value(State& state) {
        state.single_factor = Parent::template wait<false>(state.value);
    }

    template <bool Key>
    __device__ __forceinline__ void expand(const State& state) {
        Parent::template expand<Key>(state.single_factor, Key ? state.key : state.value);
    }
};

// Keep both FP8 planes' factors live across the QK/PV stages of one grouped tile.
template <int N, int Warps>
struct SparseGqaGroupedTile<KvCacheStorage::Fp8E4m3fnGroup128Ue8m0, N, Warps, true,
                            SparseGqaLoad::Staged>
    : SparseGqaStorage<KvCacheStorage::Fp8E4m3fnGroup128Ue8m0, N, Warps, SparseGqaLoad::Staged,
                       SparseGqaWarpSplit::Output> {
    static constexpr auto Storage = KvCacheStorage::Fp8E4m3fnGroup128Ue8m0;
    using Base =
        SparseGqaStorage<Storage, N, Warps, SparseGqaLoad::Staged, SparseGqaWarpSplit::Output>;
    static constexpr bool joint_scales = true;
    alignas(16) __half factors[2][N * 2];
    alignas(16) std::uint8_t value_scales[N * 2];

    struct State {
        unsigned key = 127, value = 127;
        bool single_factor = true;
    };

    __device__ __forceinline__ State prefetch(const SparseGqaArguments& a, int start, int end) {
        State state;
        const int row = threadIdx.x / 2;
        if (row < N && start + row < end) {
            const auto ko = this->addresses.key_scale[row / 4] + row % 4 * a.key_scale.row;
            const auto vo = this->addresses.value_scale[row / 4] + row % 4 * a.value_scale.row;
            state.key     = a.key_scale.data[ko + threadIdx.x % 2];
            state.value   = a.value_scale.data[vo + threadIdx.x % 2];
        }
        sparse_gqa_stage<Storage, N, Warps, true, true, false>(a, static_cast<Base&>(*this), start,
                                                               end);
        return state;
    }

    __device__ __forceinline__ void wait_key(State& state) {
        bool normal = true;
        if (threadIdx.x < N * 2) {
            const unsigned k = state.key, v = state.value;
            normal                  = k >= 103 && k <= 142 && v >= 103 && v <= 142;
            factors[0][threadIdx.x] = __float2half_rn(__uint_as_float(k << 23));
            factors[1][threadIdx.x] = __float2half_rn(__uint_as_float(v << 23));
        }
        common::cp_wait<0>();
        // V needs only the later code-copy fence after this joint publication.
        state.single_factor = __syncthreads_and(normal);
        if (!state.single_factor) {
            if (threadIdx.x < N * 2) {
                this->scales[threadIdx.x] = static_cast<std::uint8_t>(state.key);
                value_scales[threadIdx.x] = static_cast<std::uint8_t>(state.value);
            }
            __syncthreads();
        }
    }

    __device__ __forceinline__ void stage_value(const SparseGqaArguments& a, int start, int end) {
        sparse_gqa_stage<Storage, N, Warps, true, false, false>(a, static_cast<Base&>(*this), start,
                                                                end);
    }

    __device__ __forceinline__ void wait_value() {
        common::cp_wait<0>();
        __syncthreads();
    }

    template <bool Key, bool SingleFactor>
    __device__ __forceinline__ void expand_joint() {
        auto* dst = this->kv + (Key ? 0 : N * 256);
        for (int i = threadIdx.x; i < N * 32; i += Warps * 32) {
            const int row = i / 32, d = i % 32 * 8;
            int4 value;
            if constexpr (SingleFactor)
                value = common::fp8_dequant_f16x8_scaled(this->codes + row * 256 + d,
                                                         factors[Key ? 0 : 1][row * 2 + d / 128]);
            else
                value = common::fp8_dequant_f16x8(
                    this->codes + row * 256 + d,
                    (Key ? this->scales : value_scales)[row * 2 + d / 128]);
            common::store_vec(dst + row * 256 + common::swizzle_xor_64(row, d), value);
        }
        __syncthreads();
    }

    template <bool Key>
    __device__ __forceinline__ void expand(const State& state) {
        if (state.single_factor)
            expand_joint<Key, true>();
        else
            expand_joint<Key, false>();
    }
};

} // namespace ninfer::ops::detail::qsa
