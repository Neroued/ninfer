#pragma once

#include "ops/softmax_attention/qsa/sparse_gqa/cache.cuh"

#include <algorithm>

namespace ninfer::ops::detail::qsa {

template <KvCacheStorage Storage, int N, bool Separate>
struct SparseGqaRaw {};

template <KvCacheStorage Storage, int N>
struct SparseGqaRaw<Storage, N, true> {
    static_assert(kSparseQuantized<Storage>);
    using K = SparseGqaCodec<Storage, true>;
    using V = SparseGqaCodec<Storage, false>;
    alignas(16) std::uint8_t codes[N * std::max(K::code_bytes, V::code_bytes)];
    alignas(16) std::uint8_t scales[N * std::max(K::scale_bytes, V::scale_bytes)];
};

template <KvCacheStorage Storage, int N, int Warps, SparseGqaLoad Load = SparseGqaLoad::Direct,
          SparseGqaWarpSplit Split = SparseGqaWarpSplit::Sequence>
struct SparseGqaStorage : SparseGqaRaw<Storage, N, Load == SparseGqaLoad::Staged> {
    union {
        std::uint16_t kv[2 * N * 256];
        float partial[(Split == SparseGqaWarpSplit::Sequence ? Warps : 1) * 16 * 256];
    };

    // Address staging ends before warp reduction. The four address planes fit the same
    // space as reduction statistics, preserving the three-CTA allocation for 32-key tiles.
    union {
        struct {
            std::int64_t key[N / 4], value[N / 4], key_scale[N / 4], value_scale[N / 4];
        } addresses;

        struct {
            float maximum[Warps * 16], sum[Warps * 16];
        } statistics;
    };
};

// A warp owns rows warp + i*Warps during expansion. Its lanes prefetch the two
// G128 exponents for those rows, retaining them across the preceding QK/PV work.
// Pack the primary FP16 multiplier and original exponent into one register.
template <KvCacheStorage Storage, bool Key, int N, int Warps, SparseGqaLoad Load,
          SparseGqaWarpSplit Split>
__device__ __forceinline__ bool
sparse_gqa_wait_stage(SparseGqaStorage<Storage, N, Warps, Load, Split>&, unsigned& scale) {
    using Codec = SparseGqaCodec<Storage, Key>;
    if constexpr (Load != SparseGqaLoad::Direct &&
                  Codec::format == KvCacheVectorFormat::Fp8E4m3fnGroup128Ue8m0) {
        const bool half_scale  = scale >= 103 && scale <= 142;
        const unsigned primary = half_scale ? scale : min(142u, max(113u, scale));
        const auto multiplier  = __float2half_rn(__uint_as_float(primary << 23));
        scale                  = unsigned(__half_as_ushort(multiplier)) | (scale << 16);
        common::cp_wait<0>();
        return __syncthreads_and(half_scale);
    } else {
        common::cp_wait<0>();
        __syncthreads();
        return true;
    }
}

template <KvCacheStorage Storage, int N, int Warps, SparseGqaLoad Load, SparseGqaWarpSplit Split>
__device__ void sparse_gqa_addresses(const SparseGqaArguments& a,
                                     SparseGqaStorage<Storage, N, Warps, Load, Split>& s,
                                     SparseGqaWork work, std::int64_t q, int head, int start,
                                     int end) {
    const int i = threadIdx.x;
    if (i < N / 4 && start + i * 4 < end) {
        const int ordinal = start / 4 + i;
        // Distinct ids covering every complete block denote the full prefix in any order.
        const int g          = ordinal < work.count
                                   ? (work.count == work.complete ? ordinal : a.ids[q * 512 + ordinal])
                                   : work.complete;
        const int table_row  = a.table_rows[q / a.tokens];
        const auto page      = a.tables[std::int64_t(table_row) * a.table_stride + g / 16];
        const int row        = 4 * (g % 16);
        s.addresses.key[i]   = page + head * a.key.head + row * a.key.row;
        s.addresses.value[i] = page + head * a.value.head + row * a.value.row;
        if constexpr (kSparseQuantized<Storage>) {
            s.addresses.key_scale[i]   = page + head * a.key_scale.head + row * a.key_scale.row;
            s.addresses.value_scale[i] = page + head * a.value_scale.head + row * a.value_scale.row;
        }
    }
    __syncthreads();
}

template <KvCacheStorage Storage, int N, int Warps, bool Vector, bool Key,
          bool PrefetchFp8Scale = true, SparseGqaLoad Load, SparseGqaWarpSplit Split>
__device__ __forceinline__ unsigned
sparse_gqa_stage(const SparseGqaArguments& a, SparseGqaStorage<Storage, N, Warps, Load, Split>& s,
                 int start, int end) {
    using Codec      = SparseGqaCodec<Storage, Key>;
    const auto plane = Key ? a.key : a.value, scale_plane = Key ? a.key_scale : a.value_scale;
    const auto* addresses       = Key ? s.addresses.key : s.addresses.value;
    const auto* scale_addresses = Key ? s.addresses.key_scale : s.addresses.value_scale;
    auto* decoded               = s.kv + (Key ? 0 : N * 256);
    unsigned scale              = 127;
    if constexpr (Load != SparseGqaLoad::Direct) {
        static_assert(kSparseQuantized<Storage> && Vector);
        if constexpr (PrefetchFp8Scale &&
                      Codec::format == KvCacheVectorFormat::Fp8E4m3fnGroup128Ue8m0) {
            static_assert((N + Warps - 1) / Warps * 2 <= 32);
            const int row = (threadIdx.x % 32 / 2) * Warps + threadIdx.x / 32;
            if (row < N && start + row < end)
                scale = scale_plane.data[scale_addresses[row / 4] + row % 4 * scale_plane.row +
                                         threadIdx.x % 2];
        }
        constexpr int Row = Load == SparseGqaLoad::InPlace ? Codec::packed_row : Codec::code_bytes;
        std::uint8_t *codes, *scales;
        if constexpr (Load == SparseGqaLoad::InPlace) {
            codes  = reinterpret_cast<std::uint8_t*>(decoded) + N * (512 - Row);
            scales = codes + Codec::code_bytes;
        } else {
            codes  = s.codes;
            scales = s.scales;
        }
        constexpr int ScaleRow = Load == SparseGqaLoad::InPlace ? Row : Codec::scale_bytes;
        for (int i = threadIdx.x; i < N * (Codec::code_bytes / 16); i += Warps * 32) {
            const int row = i / (Codec::code_bytes / 16), d = i % (Codec::code_bytes / 16) * 16;
            auto* dst = codes + row * Row + d;
            if (start + row < end)
                common::cp_async<16>(dst,
                                     plane.data + addresses[row / 4] + row % 4 * plane.row + d);
            else
                common::store_vec(dst, make_int4(0, 0, 0, 0));
        }
        if constexpr (Codec::format != KvCacheVectorFormat::Fp8E4m3fnGroup128Ue8m0) {
            constexpr int Bytes = Codec::scale_bytes;
            const int row       = threadIdx.x;
            if (row < N) {
                auto* dst = scales + row * ScaleRow;
                if (start + row < end) {
                    const auto* src =
                        scale_plane.data + scale_addresses[row / 4] + row % 4 * scale_plane.row;
                    if (reinterpret_cast<std::uintptr_t>(src) % Bytes == 0)
                        common::cp_async<Bytes>(dst, src);
                    else {
#pragma unroll
                        for (int i = 0; i < Bytes; ++i) dst[i] = src[i];
                    }
                } else {
#pragma unroll
                    for (int i = 0; i < Bytes; ++i) dst[i] = 0;
                }
            }
        }
    } else {
#pragma unroll 1
        for (int i = threadIdx.x; i < N * 32; i += Warps * 32) {
            const int row = i / 32, d = i % 32 * 8;
            auto* dst = decoded + row * 256 + common::swizzle_xor_64(row, d);
            if (start + row >= end)
                common::store_vec(dst, make_int4(0, 0, 0, 0));
            else {
                const auto* src = plane.data + addresses[row / 4] + row % 4 * plane.row;
                if constexpr (!kSparseQuantized<Storage> && Vector)
                    common::cp_async<16>(dst, src + 2 * d);
                else {
                    const std::uint8_t* scales = nullptr;
                    if constexpr (kSparseQuantized<Storage>)
                        scales =
                            scale_plane.data + scale_addresses[row / 4] + row % 4 * scale_plane.row;
                    common::store_vec(dst, sparse_gqa_expand<Storage, Key, Vector>(src, scales, d));
                }
            }
        }
    }
    return scale;
}

// Expand a prefix only while its destination ends below unread packed bytes. The final
// overlapping row loads its operands before the publication barrier. Keeping each row
// with row%Warps preserves the FP8 scale's register owner through every in-place step.
template <KvCacheStorage Storage, bool Key, int N, int Warps, int Begin = 0>
__device__ __forceinline__ void
sparse_gqa_expand_in_place(std::uint16_t* dst, const std::uint8_t* raw, unsigned fp8_scale) {
    using Codec       = SparseGqaCodec<Storage, Key>;
    constexpr int Row = Codec::packed_row, Safe = (N - Begin) * (512 - Row) / 512,
                  Rows = Safe > 0 ? Safe : 1;
    constexpr bool Fp8 = Codec::format == KvCacheVectorFormat::Fp8E4m3fnGroup128Ue8m0;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    if constexpr (Safe == 0) {
        int4 values{};
        if (warp == (Fp8 ? Begin % Warps : 0)) {
            unsigned scale = 0;
            if constexpr (Fp8)
                scale = __shfl_sync(0xffffffffu, fp8_scale, Begin / Warps * 2 + lane / 16);
            values = sparse_gqa_expand_prepared<Storage, Key>(
                raw + Begin * Row, raw + Begin * Row + Codec::code_bytes, lane * 8, scale);
        }
        __syncthreads();
        if (warp == (Fp8 ? Begin % Warps : 0))
            common::store_vec(dst + Begin * 256 + common::swizzle_xor_64(Begin, lane * 8), values);
    } else if constexpr (Fp8) {
        const int first = Begin / Warps * Warps + warp;
        for (int row = first < Begin ? first + Warps : first; row < Begin + Rows; row += Warps) {
            const unsigned scale = __shfl_sync(0xffffffffu, fp8_scale, row / Warps * 2 + lane / 16);
            common::store_vec(dst + row * 256 + common::swizzle_xor_64(row, lane * 8),
                              sparse_gqa_expand_prepared<Storage, Key>(raw + row * Row, nullptr,
                                                                       lane * 8, scale));
        }
    } else {
        for (int i = threadIdx.x; i < Rows * 32; i += Warps * 32) {
            const int row = Begin + i / 32, d = i % 32 * 8;
            common::store_vec(dst + row * 256 + common::swizzle_xor_64(row, d),
                              sparse_gqa_expand<Storage, Key, true>(
                                  raw + row * Row, raw + row * Row + Codec::code_bytes, d));
        }
    }
    __syncthreads();
    if constexpr (Begin + Rows < N)
        sparse_gqa_expand_in_place<Storage, Key, N, Warps, Begin + Rows>(dst, raw, fp8_scale);
}

template <KvCacheStorage Storage, bool Key, int N, int Warps, SparseGqaLoad Load,
          SparseGqaWarpSplit Split>
__device__ __forceinline__ void
sparse_gqa_expand_stage(SparseGqaStorage<Storage, N, Warps, Load, Split>& s, bool half_scale,
                        unsigned fp8_scale) {
    using Codec = SparseGqaCodec<Storage, Key>;
    auto* dst   = s.kv + (Key ? 0 : N * 256);
    if constexpr (Load == SparseGqaLoad::InPlace) {
        constexpr int Row = Codec::packed_row;
        const auto* raw   = reinterpret_cast<const std::uint8_t*>(dst) + N * (512 - Row);
        sparse_gqa_expand_in_place<Storage, Key, N, Warps>(dst, raw, fp8_scale);
    } else if constexpr (Load == SparseGqaLoad::Staged) {
        for (int i = threadIdx.x; i < N * 32; i += Warps * 32) {
            const int row = i / 32, d = i % 32 * 8;
            unsigned scale = 0;
            if constexpr (Codec::format == KvCacheVectorFormat::Fp8E4m3fnGroup128Ue8m0)
                scale =
                    __shfl_sync(0xffffffffu, fp8_scale, row / Warps * 2 + threadIdx.x % 32 / 16);
            common::store_vec(dst + row * 256 + common::swizzle_xor_64(row, d),
                              sparse_gqa_expand_prepared<Storage, Key>(
                                  s.codes + row * Codec::code_bytes,
                                  s.scales + row * Codec::scale_bytes, d, scale));
        }
        __syncthreads();
    }
    if constexpr (Load != SparseGqaLoad::Direct &&
                  Codec::format == KvCacheVectorFormat::Fp8E4m3fnGroup128Ue8m0) {
        // The common expansion uses one exact primary factor. Rare exponents apply
        // their residual in a separate pass, without duplicating the expansion body.
        if (!half_scale) {
            const int row      = (threadIdx.x % 32 / 2) * Warps + threadIdx.x / 32;
            const int exponent = fp8_scale >> 16;
            if (row < N && (exponent < 103 || exponent > 142)) {
                const int primary  = min(142, max(113, exponent));
                const int residual = min(15, max(-24, exponent - primary));
                const auto scale = __float2half_rn(__uint_as_float(unsigned(residual + 127) << 23));
                const auto scale2 = __halves2half2(scale, scale);
                const int begin   = threadIdx.x % 2 * 128;
                for (int d = begin; d < begin + 128; d += 8) {
                    auto* target = dst + row * 256 + common::swizzle_xor_64(row, d);
                    auto values  = common::load_vec<int4>(target);
                    auto* pairs  = reinterpret_cast<__half2*>(&values);
#pragma unroll
                    for (int j = 0; j < 4; ++j) pairs[j] = __hmul2(pairs[j], scale2);
                    common::store_vec(target, values);
                }
            }
            __syncthreads();
        }
    }
}
} // namespace ninfer::ops::detail::qsa
