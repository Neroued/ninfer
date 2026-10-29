#pragma once

#include "ops/softmax_attention/qsa/sparse_gqa/cache.cuh"
#include "ops/softmax_attention/qsa/sparse_gqa/cache_io.cuh"

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
    using K = SparseGqaCodec<Storage, true>;
    using V = SparseGqaCodec<Storage, false>;
    // K and V occupy this slot consecutively, each with its own scale-plane offset.
    static constexpr int packed_bytes =
        std::max(K::code_bytes + K::scale_bytes, V::code_bytes + V::scale_bytes);

    union {
        // Reused serializes K/V loads, expanding quantized rows in their own tile.
        // Aliased retains a decoded tile and a packed slot to overlap K/V loads.
        alignas(
            16) std::uint16_t kv[Load == SparseGqaLoad::Reused    ? N * 256
                                 : Load == SparseGqaLoad::Aliased ? N * 256 + N * packed_bytes / 2
                                                                  : 2 * N * 256];
        float partial[(Split == SparseGqaWarpSplit::Sequence ? Warps : 1) * PaddedHeads * 256];
    };

    // Address staging ends before warp reduction, so their disjoint lifetimes
    // allow the address planes and reduction statistics to share storage.
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
    sparse_gqa_sync<Warps>();
}

template <KvCacheStorage Storage, int N, int Warps, bool Vector, bool Key, int NaturalPipeline = 0,
          SparseGqaLoad Load, SparseGqaWarpSplit Split, int PaddedHeads>
__device__ void sparse_gqa_stage(const SparseGqaArguments& a,
                                 SparseGqaStorage<Storage, N, Warps, Load, Split, PaddedHeads>& s,
                                 int start, int end) {
    using Codec      = SparseGqaCodec<Storage, Key>;
    const auto plane = Key ? a.key : a.value, scale_plane = Key ? a.key_scale : a.value_scale;
    const auto* addresses       = Key ? s.addresses.key : s.addresses.value;
    const auto* scale_addresses = Key ? s.addresses.key_scale : s.addresses.value_scale;
    auto* decoded =
        s.kv +
        (Key || Load == SparseGqaLoad::Aliased || Load == SparseGqaLoad::Reused ? 0 : N * 256);
    if constexpr (Load != SparseGqaLoad::Direct && kSparseQuantized<Storage>) {
        constexpr bool inplace = Load == SparseGqaLoad::InPlace || Load == SparseGqaLoad::Reused;
        constexpr int Row      = inplace ? Codec::packed_row : Codec::code_bytes;
        std::uint8_t *codes, *scales;
        if constexpr (inplace) {
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
        constexpr int ScaleRow = inplace ? Row : Codec::scale_bytes;
        if constexpr (NaturalPipeline && !Vector) {
            sparse_gqa_stage_natural<Codec::code_bytes, Row, N, Warps, false, NaturalPipeline>(
                codes, plane.data, plane.row, addresses, start, end);
        } else
            for (int i = threadIdx.x; i < N * (Codec::code_bytes / 16); i += Warps * 32) {
                const int row = i / (Codec::code_bytes / 16), d = i % (Codec::code_bytes / 16) * 16;
                auto* dst = codes + row * Row + d;
                if (start + row < end)
                    sparse_gqa_copy_16<Vector, Codec::code_bytes>(
                        dst, plane.data + addresses[row / 4] + row % 4 * plane.row, d);
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
                    else if constexpr (NaturalPipeline) {
                        if constexpr (Bytes == 16) {
                            const auto packet = sparse_gqa_load_copy<16>(dst, src, 0);
                            sparse_gqa_store_copy(dst, packet);
                        } else if (reinterpret_cast<std::uintptr_t>(src) % 4 == 0) {
                            common::cp_async<4>(dst, src);
                            common::cp_async<4>(dst + 4, src + 4);
                        } else if (reinterpret_cast<std::uintptr_t>(src) % 2 == 0) {
                            const auto head   = *reinterpret_cast<const std::uint16_t*>(src);
                            const auto middle = *reinterpret_cast<const unsigned*>(src + 2);
                            const auto tail   = *reinterpret_cast<const std::uint16_t*>(src + 6);
                            common::store_vec(dst, make_uint2(__byte_perm(head, middle, 0x5410),
                                                              __byte_perm(middle, tail, 0x5432)));
                        } else {
#pragma unroll
                            for (int i = 0; i < Bytes; ++i) dst[i] = src[i];
                        }
                    } else {
#pragma unroll
                        for (int i = 0; i < Bytes; ++i) dst[i] = src[i];
                    }
                } else {
#pragma unroll
                    for (int i = 0; i < Bytes; ++i) dst[i] = 0;
                }
            }
        }
    } else if constexpr (NaturalPipeline && !Vector && !kSparseQuantized<Storage>) {
        sparse_gqa_stage_natural<512, 512, N, Warps, true, NaturalPipeline>(
            reinterpret_cast<std::uint8_t*>(decoded), plane.data, plane.row, addresses, start, end);
    } else {
#pragma unroll 1
        for (int i = threadIdx.x; i < N * 32; i += Warps * 32) {
            const int row = i / 32, d = i % 32 * 8;
            auto* dst = decoded + row * 256 + common::swizzle_xor_64(row, d);
            if (start + row >= end)
                common::store_vec(dst, make_int4(0, 0, 0, 0));
            else {
                const auto* src = plane.data + addresses[row / 4] + row % 4 * plane.row;
                if constexpr (!kSparseQuantized<Storage>)
                    sparse_gqa_copy_16<Vector, 512>(dst, src, 2 * d);
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
        sparse_gqa_sync<Warps>();
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
    sparse_gqa_sync<Warps>();
    if constexpr (Begin + Rows < N)
        sparse_gqa_expand_in_place<Storage, Key, N, Warps, Begin + Rows>(dst, raw);
}

template <KvCacheStorage Storage, bool Key, int N, int Warps, SparseGqaLoad Load,
          SparseGqaWarpSplit Split, int PaddedHeads>
__device__ void
sparse_gqa_expand_stage(SparseGqaStorage<Storage, N, Warps, Load, Split, PaddedHeads>& s,
                        int live_rows = N) {
    using Codec = SparseGqaCodec<Storage, Key>;
    auto* dst =
        s.kv +
        (Key || Load == SparseGqaLoad::Aliased || Load == SparseGqaLoad::Reused ? 0 : N * 256);
    if constexpr (Load == SparseGqaLoad::InPlace ||
                  (Load == SparseGqaLoad::Reused && kSparseQuantized<Storage>)) {
        constexpr int Row = Codec::packed_row;
        auto* raw         = reinterpret_cast<std::uint8_t*>(dst) + N * (512 - Row);
        if constexpr (Codec::scale_bytes == 2) {
            const int row = threadIdx.x * 2;
            if (row < N)
                *reinterpret_cast<__half*>(raw + (row + 1) * Row + Codec::code_bytes) =
                    reinterpret_cast<const __half*>(raw + row * Row + Codec::code_bytes)[1];
            sparse_gqa_sync<Warps>();
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
        if (live_rows == N) {
            for (int i = threadIdx.x; i < N * 32; i += Warps * 32) {
                const int row = i / 32, d = i % 32 * 8;
                common::store_vec(
                    dst + row * 256 + common::swizzle_xor_64(row, d),
                    sparse_gqa_expand<Storage, Key, true>(codes + row * Codec::code_bytes,
                                                          scales + row * Codec::scale_bytes, d));
            }
        } else {
            for (int i = threadIdx.x; i < live_rows * 32; i += Warps * 32) {
                const int row = i / 32, d = i % 32 * 8;
                common::store_vec(
                    dst + row * 256 + common::swizzle_xor_64(row, d),
                    sparse_gqa_expand<Storage, Key, true>(codes + row * Codec::code_bytes,
                                                          scales + row * Codec::scale_bytes, d));
            }
            for (int i = threadIdx.x + live_rows * 32; i < N * 32; i += Warps * 32) {
                const int row = i / 32, d = i % 32 * 8;
                common::store_vec(dst + row * 256 + common::swizzle_xor_64(row, d),
                                  make_int4(0, 0, 0, 0));
            }
        }
        sparse_gqa_sync<Warps>();
    }
}
} // namespace ninfer::ops::detail::qsa
