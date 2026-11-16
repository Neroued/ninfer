#pragma once

#include "ops/common/cuda/pipeline.cuh"

namespace ninfer::ops::detail::qsa {

// Interior chunks may realign within their own live row. Boundary chunks use
// exact natural-width loads; no selected row borrows bytes from its neighbours.
template <bool Aligned, int RowBytes>
__device__ __forceinline__ void sparse_gqa_copy_16(void* dst, const std::uint8_t* row, int d) {
    const auto* src = row + d;
    if constexpr (Aligned) {
        common::cp_async<16>(dst, src);
    } else if (reinterpret_cast<std::uintptr_t>(src) % 16 == 0) {
        common::cp_async<16>(dst, src);
    } else if (reinterpret_cast<std::uintptr_t>(src) % 4 == 0) {
#pragma unroll
        for (int i = 0; i < 4; ++i)
            common::cp_async<4>(static_cast<std::uint8_t*>(dst) + 4 * i, src + 4 * i);
    } else if (d && d + 16 < RowBytes) {
        const auto address   = reinterpret_cast<std::uintptr_t>(src);
        const auto* words    = reinterpret_cast<const unsigned*>(address & ~std::uintptr_t(3));
        const unsigned shift = (address & 3) * 8;
        const auto lo        = reinterpret_cast<std::uintptr_t>(words) % 16 == 0
                                   ? common::load_vec<uint4>(words)
                                   : make_uint4(words[0], words[1], words[2], words[3]);
        const unsigned tail  = words[4];
        common::store_vec(dst, make_uint4(__funnelshift_r(lo.x, lo.y, shift),
                                          __funnelshift_r(lo.y, lo.z, shift),
                                          __funnelshift_r(lo.z, lo.w, shift),
                                          __funnelshift_r(lo.w, tail, shift)));
    } else {
        uint4 value{};
        if (reinterpret_cast<std::uintptr_t>(src) % 2 == 0) {
#pragma unroll
            for (int i = 0; i < 8; ++i)
                reinterpret_cast<std::uint16_t*>(&value)[i] =
                    reinterpret_cast<const std::uint16_t*>(src)[i];
        } else {
#pragma unroll
            for (int i = 0; i < 16; ++i) reinterpret_cast<std::uint8_t*>(&value)[i] = src[i];
        }
        common::store_vec(dst, value);
    }
}

enum class SparseGqaCopyKind : unsigned { Direct, ShiftedWords, BoundaryWords };

struct SparseGqaCopyPacket {
    unsigned words[5]{};
    SparseGqaCopyKind kind = SparseGqaCopyKind::Direct;
};

// Delay bit assembly until several independent rows have been fetched. No
// prefix or suffix load extends past the bytes owned by its selected row.
template <int Bytes>
__device__ __forceinline__ SparseGqaCopyPacket sparse_gqa_load_copy(void* dst,
                                                                    const std::uint8_t* row,
                                                                    int d) {
    const auto* src    = row + d;
    const auto address = reinterpret_cast<std::uintptr_t>(src);
    SparseGqaCopyPacket packet;
    if (address % 16 == 0) {
        common::cp_async<16>(dst, src);
    } else if (address % 4 == 0) {
#pragma unroll
        for (int j = 0; j < 4; ++j)
            common::cp_async<4>(static_cast<std::uint8_t*>(dst) + 4 * j, src + 4 * j);
    } else if (address % 2 == 0) {
        if (d && d + 16 < Bytes) {
            const auto* words = reinterpret_cast<const unsigned*>(address & ~std::uintptr_t(3));
            const auto lo     = reinterpret_cast<std::uintptr_t>(words) % 16 == 0
                                    ? common::load_vec<uint4>(words)
                                    : make_uint4(words[0], words[1], words[2], words[3]);
            packet = {{lo.x, lo.y, lo.z, lo.w, words[4]}, SparseGqaCopyKind::ShiftedWords};
        } else {
            packet.words[0] = *reinterpret_cast<const std::uint16_t*>(src);
            packet.words[4] = *reinterpret_cast<const std::uint16_t*>(src + 14);
            if (address % 8 == 2) {
                packet.words[1]   = *reinterpret_cast<const unsigned*>(src + 2);
                const auto middle = common::load_vec<uint2>(src + 6);
                packet.words[2]   = middle.x;
                packet.words[3]   = middle.y;
            } else {
                const auto middle = common::load_vec<uint2>(src + 2);
                packet.words[1]   = middle.x;
                packet.words[2]   = middle.y;
                packet.words[3]   = *reinterpret_cast<const unsigned*>(src + 10);
            }
            packet.kind = SparseGqaCopyKind::BoundaryWords;
        }
    } else {
        sparse_gqa_copy_16<false, Bytes>(dst, row, d);
    }
    return packet;
}

__device__ __forceinline__ void sparse_gqa_store_copy(void* dst,
                                                      const SparseGqaCopyPacket& packet) {
    const auto* x = packet.words;
    if (packet.kind == SparseGqaCopyKind::ShiftedWords)
        common::store_vec(
            dst, make_uint4(__funnelshift_r(x[0], x[1], 16), __funnelshift_r(x[1], x[2], 16),
                            __funnelshift_r(x[2], x[3], 16), __funnelshift_r(x[3], x[4], 16)));
    else if (packet.kind == SparseGqaCopyKind::BoundaryWords)
        common::store_vec(
            dst, make_uint4(__byte_perm(x[0], x[1], 0x5410), __funnelshift_r(x[1], x[2], 16),
                            __funnelshift_r(x[2], x[3], 16), __byte_perm(x[3], x[4], 0x5432)));
}

template <int Bytes, int Stride, int N, int Warps, bool Swizzled, int Pipeline>
__device__ __forceinline__ void
sparse_gqa_stage_natural(std::uint8_t* dst, const std::uint8_t* data, std::int64_t row_stride,
                         const std::int64_t* addresses, int start, int end) {
    constexpr int Chunks = Bytes / 16, Threads = Warps * 32;
#pragma unroll 1
    for (int i = threadIdx.x; i < N * Chunks; i += Pipeline * Threads) {
        SparseGqaCopyPacket pending[Pipeline];
#pragma unroll
        for (int j = 0; j < Pipeline; ++j) {
            const int index = i + j * Threads, row = index / Chunks, d = index % Chunks * 16;
            if (index < N * Chunks) {
                auto* target =
                    dst + row * Stride + (Swizzled ? 2 * common::swizzle_xor_64(row, d / 2) : d);
                if (start + row < end)
                    pending[j] = sparse_gqa_load_copy<Bytes>(
                        target, data + addresses[row / 4] + row % 4 * row_stride, d);
                else
                    common::store_vec(target, make_uint4(0, 0, 0, 0));
            }
        }
#pragma unroll
        for (int j = 0; j < Pipeline; ++j) {
            const int index = i + j * Threads, row = index / Chunks, d = index % Chunks * 16;
            if (index < N * Chunks)
                sparse_gqa_store_copy(dst + row * Stride +
                                          (Swizzled ? 2 * common::swizzle_xor_64(row, d / 2) : d),
                                      pending[j]);
        }
    }
}

} // namespace ninfer::ops::detail::qsa
