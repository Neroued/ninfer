#pragma once
#include "ops/moe/kernels/cuda/block_scaled/clamped/nvfp4/compute.cuh"

namespace ninfer::ops::detail::moe_cuda::block_scaled::clamped::nvfp4::csf_scale_tile {
template <int I, bool Gate, int Columns>
struct Shape {
    static constexpr int k = Gate ? kHidden : I, features = Gate ? 32 : 64, groups = k / 16;
    static constexpr int threads = 256, warps = 8, columns = Columns, pages = Gate ? 4 : 1,
                         capacity = 256;
};

template <class S>
struct Tile {
    struct alignas(16) Encoded {
        std::uint8_t nibbles[4][32];
    };

    // Encoded metadata and expanded scales share storage. restore() consumes
    // batches backwards, loading every overlapping source before writing it.
    union alignas(16) Data {
        Encoded encoded[S::k / 64];
        unsigned scales[(S::k / 64) * 64];
    } data;

    unsigned bounds[S::k / 64][4];

    // Exception pages stay compressed. Pages above this bounded cache use
    // the original records, including dense real-weight exception tails.
    alignas(16) std::uint16_t exceptions[S::pages][S::capacity + 8];
    common::Csf32RowBases128 bases;
    unsigned starts[S::pages];
    int lengths[S::pages];
};

__device__ __forceinline__ unsigned scale_word(unsigned row, unsigned group4) {
    // Consecutive rows are conflict-free during decode; the XOR also spreads
    // the two-K16-group word loads of each consumer warp over shared banks.
    return group4 * 64 + (row ^ ((group4 & 15) * 2));
}

template <class S>
struct Destination {
    Tile<S>& tile;
    unsigned row, group;

    __device__ __forceinline__ void store_exception(unsigned lane, unsigned col,
                                                    std::uint8_t value) const {
        const unsigned g = group + col;
        reinterpret_cast<std::uint8_t*>(
            tile.data.scales)[scale_word(row + lane, g / 4) * 4 + g % 4] = value;
    }
};

template <int I, bool Gate, class S>
__device__ __forceinline__ void restore_warp(Tile<S>& tile, common::Nvfp4MultiplierMatrix weight,
                                             int row0) {
    const auto& source = weight.csf;
    const auto rows    = make_uint4(row0, row0 + (Gate ? I : 32), 0, 0);
    const int tid = threadIdx.x % 32, warp = 0, lane = tid;
    if (warp == 0) common::csf32_copy_row_bases_async<2>(source, rows, tile.bases);
    for (int item = tid; item < (S::k / 64) * 8; item += 32) {
        const int group = item / 8, index = item / 2 % 4, half = item & 1;
        const auto b = common::csf32_detail::locate<Gate ? 2 * I : kHidden, S::groups>(
            source, common::csf32_detail::row_origin(rows, index / 2), group * 4 + (index & 1) * 2);
        common::cp_async<16>(tile.data.encoded[group].nibbles[index] + half * 16,
                             source.nibbles + b.nibbles + half * 16);
    }
    common::cp_commit();
    for (int item = tid; item < (S::k / 64) * 4; item += 32) {
        const int group = item / 4, index = item % 4;
        const auto b = common::csf32_detail::locate<Gate ? 2 * I : kHidden, S::groups>(
            source, common::csf32_detail::row_origin(rows, index / 2), group * 4 + (index & 1) * 2);
        tile.bounds[group][index] =
            unsigned(source.offsets[b.offset]) | (unsigned(source.offsets[b.offset + 1]) << 16);
    }
    if (tid < S::pages) {
        const auto b = common::csf32_detail::locate<Gate ? 2 * I : kHidden, S::groups>(
            source, Gate && tid >= 2 ? rows.y : rows.x, Gate ? (tid & 1) * 128 : 0);
        const int count   = source.offsets[(b.page + 1) * 257 - 1];
        tile.starts[tid]  = source.page_bases[b.page] / 2;
        tile.lengths[tid] = count <= S::capacity ? count : -1;
    }
    __syncwarp();
    constexpr int vectors = (S::capacity + 8) / 8;
    for (int item = tid; item < S::pages * vectors; item += 32) {
        const int page = item / vectors, segment = item % vectors, count = tile.lengths[page];
        const unsigned start = tile.starts[page];
        if (count > 0 && segment * 8 < count + (start & 7)) {
            const std::uint64_t off = std::uint64_t(start & ~7U) + segment * 8;
            common::cp_async_zfill<16>(
                tile.exceptions[page] + segment * 8, source.exceptions + off,
                int(min(std::uint64_t(16), (source.exception_count - off) * 2)));
        }
    }
    common::cp_commit();
    common::cp_wait<0>();
    __syncwarp();

    // Each warp captures both source halves before expanding the current
    // group. Descending groups make the compressed/output overlap safe.
    for (int group4 = S::k / 64 - 1; group4 >= 0; --group4) {
        unsigned packed[2], first[2], second[2];
#pragma unroll
        for (int part = 0; part < 2; ++part) {
            packed[part] = unsigned(tile.data.encoded[group4].nibbles[part * 2][lane]) |
                           (unsigned(tile.data.encoded[group4].nibbles[part * 2 + 1][lane]) << 8);
            first[part]  = tile.bounds[group4][part * 2];
            second[part] = tile.bounds[group4][part * 2 + 1];
        }
        __syncwarp();
#pragma unroll
        for (int part = 0; part < 2; ++part) {
            const unsigned repeated = __byte_perm(packed[part], 0, 0x1100);
            const unsigned decoded  = ((repeated & 0x000f000fU) | ((repeated >> 4) & 0x0f000f00U)) +
                                     unsigned(tile.bases.values[part][lane]) * 0x01010101U;
            tile.data.scales[scale_word(part * 32 + lane, group4)] = decoded;
            const unsigned begin0 = first[part] & 65535, begin1 = second[part] & 65535;
            const unsigned count0 = (first[part] >> 16) - begin0,
                           count1 = (second[part] >> 16) - begin1;
            if (count0 || count1) {
                __syncwarp();
                const int page       = Gate ? part * 2 + group4 / 32 : 0;
                const unsigned start = tile.starts[page], bias = start & 7;
                const bool cached = tile.lengths[page] >= 0;
                const Destination<S> d0{tile, unsigned(part * 32), unsigned(group4 * 4)};
                const Destination<S> d1{tile, unsigned(part * 32), unsigned(group4 * 4 + 2)};
                for (unsigned index = lane; index < max(count0, count1); index += 32) {
                    if (index < count0) {
                        const unsigned v =
                            cached ? tile.exceptions[page][bias + begin0 + index]
                                   : source.exceptions[std::uint64_t(start) + begin0 + index];
                        d0.store_exception((v & 255) / 2, v & 1, v >> 8);
                    }
                    if (index < count1) {
                        const unsigned v =
                            cached ? tile.exceptions[page][bias + begin1 + index]
                                   : source.exceptions[std::uint64_t(start) + begin1 + index];
                        d1.store_exception((v & 255) / 2, v & 1, v >> 8);
                    }
                }
            }
        }
        __syncwarp();
    }
}

} // namespace ninfer::ops::detail::moe_cuda::block_scaled::clamped::nvfp4::csf_scale_tile
