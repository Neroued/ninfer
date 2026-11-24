#pragma once

#include "ops/softmax_attention/qsa/sparse_gqa/tile.cuh"

namespace ninfer::ops::detail::qsa {

template <int N, bool Enabled>
struct SparseGqaOutputScales {};

template <int N>
struct SparseGqaOutputScales<N, true> {
    alignas(16) __half factors[N * 2];
};

template <KvCacheStorage Storage, int N, int Warps, bool Vector, SparseGqaLoad Load,
          bool SharedScales>
struct SparseGqaOutputTile : SparseGqaStorage<Storage, N, Warps, Load, SparseGqaWarpSplit::Output>,
                             SparseGqaOutputScales<N, SharedScales> {
    using Base = SparseGqaStorage<Storage, N, Warps, Load, SparseGqaWarpSplit::Output>;
    static_assert(!SharedScales ||
                  (SparseGqaCodec<Storage, true>::format ==
                       KvCacheVectorFormat::Fp8E4m3fnGroup128Ue8m0 &&
                   Vector && Load == SparseGqaLoad::Staged && N * 2 <= Warps * 32));

    template <bool Key>
    static constexpr bool prepared_scale =
        SharedScales &&
        SparseGqaCodec<Storage, Key>::format == KvCacheVectorFormat::Fp8E4m3fnGroup128Ue8m0;

    template <bool Key>
    __device__ __forceinline__ unsigned prefetch_scale(const SparseGqaArguments& a, int start,
                                                       int end) {
        const auto plane      = Key ? a.key_scale : a.value_scale;
        const auto* addresses = Key ? this->addresses.key_scale : this->addresses.value_scale;
        const int row         = threadIdx.x / 2;
        unsigned scale        = 127;
        if (row < N && start + row < end)
            scale = plane.data[addresses[row / 4] + row % 4 * plane.row + threadIdx.x % 2];
        return scale;
    }

    template <bool Key, bool FetchScale = true>
    __device__ __forceinline__ unsigned prefetch(const SparseGqaArguments& a, int start, int end) {
        if constexpr (prepared_scale<Key>) {
            const unsigned scale = FetchScale ? prefetch_scale<Key>(a, start, end) : 127;
            sparse_gqa_stage<Storage, N, Warps, Vector, Key, false>(a, static_cast<Base&>(*this),
                                                                    start, end);
            return scale;
        } else {
            return sparse_gqa_stage<Storage, N, Warps, Vector, Key>(a, static_cast<Base&>(*this),
                                                                    start, end);
        }
    }

    template <bool Key>
    __device__ __forceinline__ bool wait(unsigned& scale) {
        if constexpr (prepared_scale<Key>) {
            bool single_factor = true;
            if (threadIdx.x < N * 2) {
                single_factor = scale >= 103 && scale <= 142;
                // A multiplier outside the FP16 range is consumed by the full decoder.
                this->factors[threadIdx.x] = __float2half_rn(__uint_as_float(scale << 23));
            }
            common::cp_wait<0>();
            return __syncthreads_and(single_factor);
        } else {
            return sparse_gqa_wait_stage<Storage, Key>(static_cast<Base&>(*this), scale);
        }
    }

    template <bool Key, bool SingleFactor>
    __device__ __forceinline__ void expand_shared() {
        auto* dst = this->kv + (Key ? 0 : N * 256);
        for (int i = threadIdx.x; i < N * 32; i += Warps * 32) {
            const int row = i / 32, d = i % 32 * 8;
            int4 values;
            if constexpr (SingleFactor)
                values = common::fp8_dequant_f16x8_scaled(this->codes + row * 256 + d,
                                                          this->factors[row * 2 + d / 128]);
            else
                values = common::fp8_dequant_f16x8(this->codes + row * 256 + d,
                                                   this->scales[row * 2 + d / 128]);
            common::store_vec(dst + row * 256 + common::swizzle_xor_64(row, d), values);
        }
        __syncthreads();
    }

    template <bool Key>
    __device__ __forceinline__ void expand(bool single_factor, unsigned scale) {
        if constexpr (prepared_scale<Key>) {
            // Select once for the tile so the ordinary loop keeps only its operands live.
            if (single_factor)
                expand_shared<Key, true>();
            else {
                if (threadIdx.x < N * 2)
                    this->scales[threadIdx.x] = static_cast<std::uint8_t>(scale);
                __syncthreads();
                expand_shared<Key, false>();
            }
        } else
            sparse_gqa_expand_stage<Storage, Key>(static_cast<Base&>(*this), single_factor, scale);
    }
};

} // namespace ninfer::ops::detail::qsa
