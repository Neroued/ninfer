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
          SparseGqaWarpSplit Split = SparseGqaWarpSplit::Sequence, int PaddedHeads = 16>
struct SparseGqaStorage : SparseGqaRaw<Storage, N, Load == SparseGqaLoad::Staged> {
    union {
        // Aliased decode needs one FP16 tile and one native packed tile.
        alignas(16) std::uint16_t
            kv[Load == SparseGqaLoad::Aliased
                   ? N * 256 + N *
                                   (std::max(SparseGqaCodec<Storage, true>::code_bytes,
                                             SparseGqaCodec<Storage, false>::code_bytes) +
                                    std::max(SparseGqaCodec<Storage, true>::scale_bytes,
                                             SparseGqaCodec<Storage, false>::scale_bytes)) /
                                   2
                   : 2 * N * 256];
        float partial[(Split == SparseGqaWarpSplit::Sequence ? Warps : 1) * PaddedHeads * 256];
    };

    // Address staging ends before warp reduction. The four address planes fit the same
    // space as reduction statistics, preserving the three-CTA allocation for 32-key tiles.
    union {
        struct {
            std::int64_t key[N / 4], value[N / 4], key_scale[N / 4], value_scale[N / 4];
        } addresses;

        struct {
            float maximum[Warps * PaddedHeads], sum[Warps * PaddedHeads];
        } statistics;
    };
};

template <KvCacheStorage Storage, int N, int Warps, SparseGqaLoad Load, SparseGqaWarpSplit Split,
          int PaddedHeads>
__device__ void
sparse_gqa_addresses(const SparseGqaArguments& a,
                     SparseGqaStorage<Storage, N, Warps, Load, Split, PaddedHeads>& s,
                     SparseGqaWork work, std::int64_t q, int head, int start, int end) {
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

template <KvCacheStorage Storage, int N, int Warps, bool Vector, bool Key, SparseGqaLoad Load,
          SparseGqaWarpSplit Split, int PaddedHeads>
__device__ void sparse_gqa_stage(const SparseGqaArguments& a,
                                 SparseGqaStorage<Storage, N, Warps, Load, Split, PaddedHeads>& s,
                                 int start, int end) {
    using Codec      = SparseGqaCodec<Storage, Key>;
    const auto plane = Key ? a.key : a.value, scale_plane = Key ? a.key_scale : a.value_scale;
    const auto* addresses       = Key ? s.addresses.key : s.addresses.value;
    const auto* scale_addresses = Key ? s.addresses.key_scale : s.addresses.value_scale;
    auto* decoded               = s.kv + (Key || Load == SparseGqaLoad::Aliased ? 0 : N * 256);
    if constexpr (Load != SparseGqaLoad::Direct) {
        static_assert(kSparseQuantized<Storage> && Vector);
        constexpr int Row = Load == SparseGqaLoad::InPlace ? Codec::packed_row : Codec::code_bytes;
        std::uint8_t *codes, *scales;
        if constexpr (Load == SparseGqaLoad::InPlace) {
            codes  = reinterpret_cast<std::uint8_t*>(decoded) + N * (512 - Row);
            scales = codes + Codec::code_bytes;
        } else if constexpr (Load == SparseGqaLoad::Aliased) {
            // QK and PV consume decoded tiles consecutively. The other half holds a
            // compact async slot, disjoint from the current decoded operand.
            codes  = reinterpret_cast<std::uint8_t*>(s.kv + N * 256);
            scales = codes + N * Codec::code_bytes;
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
        if constexpr (Codec::scale_bytes == 2) {
            // Async copies require at least four bytes. Pair rows within the same logical
            // four-token block; a partial final pair reads only its live half.
            const int row = threadIdx.x * 2;
            if (row < N) {
                auto* dst = scales + row * ScaleRow;
                if (start + row < end) {
                    const auto* src =
                        scale_plane.data + scale_addresses[row / 4] + row % 4 * scale_plane.row;
                    if (scale_plane.row == 2 && reinterpret_cast<std::uintptr_t>(src) % 4 == 0 &&
                        start + row + 1 < end)
                        common::cp_async<4>(dst, src);
                    else {
                        const __half x = *reinterpret_cast<const __half*>(src);
                        const __half y =
                            start + row + 1 < end
                                ? *reinterpret_cast<const __half*>(src + scale_plane.row)
                                : __float2half_rn(0);
                        *reinterpret_cast<__half2*>(dst) = __halves2half2(x, y);
                    }
                } else
                    *reinterpret_cast<unsigned*>(dst) = 0;
            }
        } else {
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
}

// Expand a prefix only while its destination ends below unread packed bytes. The final
// overlapping rows first load all operands, synchronize, and then publish expanded words.
template <KvCacheStorage Storage, bool Key, int N, int Warps, int Begin = 0>
__device__ __forceinline__ void sparse_gqa_expand_in_place(std::uint16_t* dst,
                                                           const std::uint8_t* raw) {
    using Codec       = SparseGqaCodec<Storage, Key>;
    constexpr int Row = Codec::packed_row, Safe = (N - Begin) * (512 - Row) / 512,
                  Rows = Safe > 0 ? Safe : 1;
    if constexpr (Safe == 0) {
        int4 values{};
        if (threadIdx.x < 32)
            values = sparse_gqa_expand<Storage, Key, true>(
                raw + Begin * Row, raw + Begin * Row + Codec::code_bytes, threadIdx.x * 8);
        __syncthreads();
        if (threadIdx.x < 32)
            common::store_vec(dst + Begin * 256 + common::swizzle_xor_64(Begin, threadIdx.x * 8),
                              values);
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
        sparse_gqa_expand_in_place<Storage, Key, N, Warps, Begin + Rows>(dst, raw);
}

template <KvCacheStorage Storage, bool Key, int N, int Warps, SparseGqaLoad Load,
          SparseGqaWarpSplit Split, int PaddedHeads>
__device__ void
sparse_gqa_expand_stage(SparseGqaStorage<Storage, N, Warps, Load, Split, PaddedHeads>& s) {
    using Codec = SparseGqaCodec<Storage, Key>;
    auto* dst   = s.kv + (Key || Load == SparseGqaLoad::Aliased ? 0 : N * 256);
    if constexpr (Load == SparseGqaLoad::InPlace) {
        constexpr int Row = Codec::packed_row;
        auto* raw         = reinterpret_cast<std::uint8_t*>(dst) + N * (512 - Row);
        if constexpr (Codec::scale_bytes == 2) {
            const int row = threadIdx.x * 2;
            if (row < N)
                *reinterpret_cast<__half*>(raw + (row + 1) * Row + Codec::code_bytes) =
                    reinterpret_cast<const __half*>(raw + row * Row + Codec::code_bytes)[1];
            __syncthreads();
        }
        sparse_gqa_expand_in_place<Storage, Key, N, Warps>(dst, raw);
    } else if constexpr (Load == SparseGqaLoad::Staged || Load == SparseGqaLoad::Aliased) {
        const std::uint8_t *codes, *scales;
        if constexpr (Load == SparseGqaLoad::Aliased) {
            codes  = reinterpret_cast<const std::uint8_t*>(s.kv + N * 256);
            scales = codes + N * Codec::code_bytes;
        } else {
            codes  = s.codes;
            scales = s.scales;
        }
        for (int i = threadIdx.x; i < N * 32; i += Warps * 32) {
            const int row = i / 32, d = i % 32 * 8;
            common::store_vec(
                dst + row * 256 + common::swizzle_xor_64(row, d),
                sparse_gqa_expand<Storage, Key, true>(codes + row * Codec::code_bytes,
                                                      scales + row * Codec::scale_bytes, d));
        }
        __syncthreads();
    }
}
} // namespace ninfer::ops::detail::qsa
