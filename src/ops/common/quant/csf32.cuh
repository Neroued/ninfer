#pragma once

#include "core/weight_bank.h"
#include "ops/common/cuda/pipeline.cuh"
#include "ops/common/quant/nvfp4.cuh"

namespace ninfer::ops::common {
namespace csf32_detail {

__device__ __forceinline__ unsigned lane_id() {
    unsigned lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));
    return lane;
}

__device__ __forceinline__ unsigned row_origin(uint4 rows, unsigned index) {
    return index & 2 ? (index & 1 ? rows.w : rows.z) : (index & 1 ? rows.y : rows.x);
}

struct Block {
    std::uint64_t nibbles, offset, page;
    unsigned row, columns;
};

template <int MatrixRows, int ScaleColumns>
__device__ __forceinline__ Block locate(const Csf32ScaleView& source, unsigned row,
                                        unsigned column) {
    static_assert(MatrixRows >= 0 && MatrixRows % 32 == 0 && ScaleColumns >= 0);
    const unsigned rows    = MatrixRows ? MatrixRows : static_cast<unsigned>(source.rows);
    const unsigned columns = ScaleColumns ? ScaleColumns : static_cast<unsigned>(source.columns);
    row += static_cast<unsigned>(source.row_begin);
    const unsigned row_block = row / 32, row_tile = row_block / 4, pair = column / 2;
    const unsigned pairs = (columns + 1) / 2;
    const unsigned tile_rows =
        MatrixRows && MatrixRows % 128 == 0 ? 4 : min(4U, rows / 32 - row_tile * 4);
    const unsigned width = ScaleColumns && ScaleColumns % 2 == 0 ? 2 : min(2U, columns - column);
    const std::uint64_t ordinal =
        std::uint64_t(row_tile) * 4 * pairs + pair * tile_rows + row_block % 4;
    const std::uint64_t page   = (std::uint64_t(row_tile) * ((pairs + 1) / 2) + pair / 2) / 32;
    const std::uint64_t packed = ScaleColumns && ScaleColumns % 2 == 0
                                     ? ordinal * 32
                                     : std::uint64_t(row_tile) * 64 * columns +
                                           std::uint64_t(pair) * 32 * tile_rows +
                                           (row_block % 4) * 16 * width;
    return {packed, ordinal + page, page, row, width};
}

__device__ __forceinline__ std::uint16_t base_pair(const Csf32ScaleView& source,
                                                   const Block& block) {
    const unsigned lane   = lane_id();
    const unsigned packed = source.nibbles[block.nibbles + (block.columns == 2 ? lane : lane / 2)];
    const unsigned base   = source.bases[block.row + lane];
    const unsigned lo     = base + ((packed >> (block.columns == 2 ? 0 : (lane & 1) * 4)) & 15);
    const unsigned hi     = block.columns == 2 ? base + (packed >> 4) : 0;
    return static_cast<std::uint16_t>(lo | (hi << 8));
}

template <class Destination>
__device__ __forceinline__ void patch_exceptions(const std::uint16_t* exceptions, unsigned begin,
                                                 unsigned end, Destination destination) {
    constexpr unsigned mask = 0xffffffffU;
    __syncwarp(mask);
    for (unsigned index = begin + lane_id(); index < end; index += 32) {
        const unsigned value = exceptions[index], position = value & 255;
        destination.store_exception(position / 2, position & 1,
                                    static_cast<std::uint8_t>(value >> 8));
    }
    __syncwarp(mask);
}

template <class Destination>
__device__ __forceinline__ void patch(const Csf32ScaleView& source, std::uint64_t page,
                                      unsigned begin, unsigned end, Destination destination) {
    patch_exceptions(source.exceptions + std::uint64_t(source.page_bases[page]) / 2, begin, end,
                     destination);
}

template <int MatrixRows, int ScaleColumns, class Destination>
__device__ __forceinline__ void decode(const Csf32ScaleView& source, unsigned row, unsigned column,
                                       Destination destination) {
    const auto block = locate<MatrixRows, ScaleColumns>(source, row, column);
    destination.store_pair(lane_id(), base_pair(source, block), block.columns);
    const unsigned begin = source.offsets[block.offset], end = source.offsets[block.offset + 1];
    if (begin != end) patch(source, block.page, begin, end, destination);
}

template <int KStep, bool LinearRows = false>
struct NativeDestination {
    std::uint8_t* data;
    unsigned row, group;

    __device__ __forceinline__ unsigned address(unsigned r, unsigned g) const {
        if constexpr (LinearRows) return (g / 4 * 128 + r) * 4 + g % 4;
        return nvfp4_scale_byte_offset(r, g, KStep);
    }

    __device__ __forceinline__ void store_pair(unsigned lane, std::uint16_t value,
                                               unsigned columns) const {
        auto* destination = data + address(row + lane, group);
        if (columns == 2)
            *reinterpret_cast<std::uint16_t*>(destination) = value;
        else
            *destination = static_cast<std::uint8_t>(value);
    }

    __device__ __forceinline__ void store_exception(unsigned lane, unsigned column,
                                                    std::uint8_t value) const {
        data[address(row + lane, group + column)] = value;
    }
};

template <class T>
__device__ __forceinline__ const T* advance(const T* data, std::uint64_t bytes) {
    return reinterpret_cast<const T*>(reinterpret_cast<const std::uint8_t*>(data) + bytes);
}

} // namespace csf32_detail

// One full warp restores a valid 32-row, two-scale-column block. Source coordinates are
// warp-uniform and relative to the matrix view; destination coordinates address its native
// M128x4 scale tile.
// Template extents, when nonzero, describe the original source matrix. Shared slots belong
// to the caller; it supplies cross-warp synchronization before other warps consume them.
template <int KStep, int MatrixRows = 0, int ScaleColumns = 0>
__device__ __forceinline__ void
csf32_stage_block(const Csf32ScaleView& source, unsigned source_row32, unsigned source_group2,
                  std::uint8_t* shared_native_scales, unsigned destination_row32,
                  unsigned destination_group2) {
    static_assert(KStep > 0 && KStep % 64 == 0);
    csf32_detail::decode<MatrixRows, ScaleColumns>(
        source, source_row32, source_group2,
        csf32_detail::NativeDestination<KStep>{shared_native_scales, destination_row32,
                                               destination_group2});
}

struct alignas(16) Csf32RowBases128 {
    std::uint8_t values[4][32];
};

struct alignas(16) Csf32EncodedTile128x4 {
    std::uint8_t nibbles[8][32];
    std::uint32_t offset_words[8][2];
    std::uint32_t page_bases[8];
    unsigned odd_offsets;
};

namespace csf32_detail {

struct StagedBlock {
    std::uint16_t pair;
    unsigned begin, end;
};

template <int MatrixRows, int ScaleColumns>
__device__ __forceinline__ StagedBlock staged_block(const Csf32ScaleView& source,
                                                    unsigned source_row32, unsigned source_group2,
                                                    const Csf32RowBases128& bases,
                                                    const Csf32EncodedTile128x4& encoded,
                                                    unsigned row_part, unsigned pair) {
    const unsigned index = row_part * 2 + pair, lane = lane_id();
    const unsigned packed = encoded.nibbles[index][lane], base = bases.values[row_part][lane];
    const auto value =
        static_cast<std::uint16_t>((base + (packed & 15)) | ((base + (packed >> 4)) << 8));
    const auto first = encoded.offset_words[index][0];
    const bool odd   = (encoded.odd_offsets >> (index * 4)) & 1;
    return {value, odd ? first >> 16 : first & 65535,
            odd ? encoded.offset_words[index][1] & 65535 : first >> 16};
}

} // namespace csf32_detail

// Full-warps cooperatively copy fixed compressed planes to caller-owned shared storage.
// Four valid 32-row blocks begin at 32-aligned row origins; four complete scale columns
// begin at a 4-aligned group origin. Bases can be retained across K tiles. The caller
// owns cp.async commit/wait and synchronization before decode.
template <int RowBlocks = 4>
__device__ __forceinline__ void csf32_copy_row_bases_async(const Csf32ScaleView& source,
                                                           uint4 source_row32s,
                                                           Csf32RowBases128& destination) {
    const unsigned lane = csf32_detail::lane_id();
    static_assert(RowBlocks == 2 || RowBlocks == 4);
    if (lane < RowBlocks * 2) {
        const unsigned row = lane / 2, half = (lane & 1) * 16;
        cp_async<16>(destination.values[row] + half,
                     source.bases + source.row_begin +
                         csf32_detail::row_origin(source_row32s, row) + half);
    }
}

template <int MatrixRows = 0, int ScaleColumns = 0, int RowBlocks = 4>
__device__ __forceinline__ void
csf32_copy_tile128x4_async(const Csf32ScaleView& source, uint4 source_row32s,
                           unsigned source_group4, Csf32EncodedTile128x4& destination) {
    const unsigned lane = csf32_detail::lane_id(), index = lane / 4, part = lane & 3;
    static_assert(RowBlocks == 2 || RowBlocks == 4);
    if (index >= RowBlocks * 2) return;
    const auto block = csf32_detail::locate<MatrixRows, ScaleColumns>(
        source, csf32_detail::row_origin(source_row32s, index / 2),
        source_group4 + (index & 1) * 2);
    const auto address = reinterpret_cast<std::uintptr_t>(source.offsets + block.offset);
    const auto odd     = __ballot_sync(RowBlocks == 2 ? 0xffffU : 0xffffffffU, (address & 2) != 0);
    if (!lane) destination.odd_offsets = odd;
    if (part < 2) {
        cp_async<16>(destination.nibbles[index] + part * 16,
                     source.nibbles + block.nibbles + part * 16);
    } else {
        const auto aligned = address & ~std::uintptr_t(3);
        if (part == 2) {
            if (block.offset == 0 && (address & 2))
                destination.offset_words[index][0] = unsigned(source.offsets[0]) << 16;
            else
                cp_async<4>(&destination.offset_words[index][0],
                            reinterpret_cast<const void*>(aligned));
        } else if (address & 2) {
            const std::uint64_t row_count = MatrixRows ? MatrixRows : source.rows;
            const std::uint64_t columns   = ScaleColumns ? ScaleColumns : source.columns;
            const auto pairs              = (columns + 1) / 2;
            const auto page_count   = (((row_count + 127) / 128) * ((pairs + 1) / 2) + 31) / 32;
            const auto offset_count = row_count / 32 * pairs + page_count;
            const auto end = reinterpret_cast<std::uintptr_t>(source.offsets + offset_count);
            const int valid =
                static_cast<int>(min(std::uint64_t(4), std::uint64_t(end - aligned - 4)));
            cp_async_zfill<4>(&destination.offset_words[index][1],
                              reinterpret_cast<const void*>(aligned + 4), valid);
        } else {
            destination.offset_words[index][1] = 0;
        }
    }
    if (lane < RowBlocks * 2) {
        const auto page_block = csf32_detail::locate<MatrixRows, ScaleColumns>(
            source, csf32_detail::row_origin(source_row32s, lane / 2),
            source_group4 + (lane & 1) * 2);
        cp_async<4>(&destination.page_bases[lane], source.page_bases + page_block.page);
    }
}

// One full warp restores a staged 32x2 block. row_part selects one of the four source
// row origins and pair selects columns 0/1 or 2/3 of the encoded tile. Fixed planes must
// be ready before entry; the caller synchronizes consumers. LinearRows uses [K/64,128]
// packed four-scale words, avoiding four-way bank conflicts when each lane writes a row.
template <int KStep, int MatrixRows = 0, int ScaleColumns = 0, bool LinearRows = false>
__device__ __forceinline__ void
csf32_decode_block(const Csf32ScaleView& source, uint4 source_row32s, unsigned source_group4,
                   const Csf32RowBases128& bases, const Csf32EncodedTile128x4& encoded,
                   unsigned row_part, unsigned pair, std::uint8_t* shared_native_scales,
                   unsigned destination_row32, unsigned destination_group2) {
    static_assert(KStep > 0 && KStep % 64 == 0);
    const auto block = csf32_detail::staged_block<MatrixRows, ScaleColumns>(
        source, csf32_detail::row_origin(source_row32s, row_part), source_group4 + pair * 2, bases,
        encoded, row_part, pair);
    const csf32_detail::NativeDestination<KStep, LinearRows> destination{
        shared_native_scales, destination_row32, destination_group2};
    destination.store_pair(csf32_detail::lane_id(), block.pair, 2);
    if (block.begin != block.end)
        csf32_detail::patch_exceptions(
            source.exceptions + std::uint64_t(encoded.page_bases[row_part * 2 + pair]) / 2,
            block.begin, block.end, destination);
}

// Restore a whole four-scale word, then patch both two-column exception lists
// with a single pair of warp barriers. Source columns are four-aligned.
template <int KStep, int MatrixRows = 0, int ScaleColumns = 0, bool LinearRows = false>
__device__ __forceinline__ void
csf32_decode_word(const Csf32ScaleView& source, uint4 rows, unsigned group4,
                  const Csf32RowBases128& bases, const Csf32EncodedTile128x4& encoded,
                  unsigned part, std::uint8_t* data, unsigned destination_row,
                  unsigned destination_group) {
    const auto lo = csf32_detail::staged_block<MatrixRows, ScaleColumns>(
        source, csf32_detail::row_origin(rows, part), group4, bases, encoded, part, 0);
    const auto hi = csf32_detail::staged_block<MatrixRows, ScaleColumns>(
        source, csf32_detail::row_origin(rows, part), group4 + 2, bases, encoded, part, 1);
    const csf32_detail::NativeDestination<KStep, LinearRows> d0{data, destination_row,
                                                                destination_group};
    const csf32_detail::NativeDestination<KStep, LinearRows> d1{data, destination_row,
                                                                destination_group + 2};
    const unsigned lane = csf32_detail::lane_id();
    *reinterpret_cast<unsigned*>(data + d0.address(destination_row + lane, destination_group)) =
        unsigned(lo.pair) | (unsigned(hi.pair) << 16);
    const unsigned n0 = lo.end - lo.begin, n1 = hi.end - hi.begin;
    if (n0 || n1) {
        __syncwarp();
        const auto* exceptions =
            source.exceptions + std::uint64_t(encoded.page_bases[part * 2]) / 2;
        for (unsigned index = lane; index < max(n0, n1); index += 32) {
            if (index < n0) {
                const unsigned value = exceptions[lo.begin + index];
                d0.store_exception((value & 255) / 2, value & 1, value >> 8);
            }
            if (index < n1) {
                const unsigned value = exceptions[hi.begin + index];
                d1.store_exception((value & 255) / 2, value & 1, value >> 8);
            }
        }
        __syncwarp();
    }
}

struct Nvfp4MultiplierMatrix {
    const std::uint8_t* codes  = nullptr;
    const std::uint8_t* scales = nullptr;
    Csf32ScaleView csf;
    float multiplier = 1.0F;
    bool valid       = false;
};

// Weight planes/scalars follow the physical slot; activation calibration remains indexed
// separately by logical expert ID. The public consumer validates format and ID bounds.
template <bool KnownCsf = false>
__device__ __forceinline__ Nvfp4MultiplierMatrix
nvfp4_mul_expert_matrix(const ExpertMatrixDeviceView& view, unsigned logical_expert) {
    Nvfp4MultiplierMatrix result;
    unsigned physical    = logical_expert;
    std::uint64_t offset = 0;
    if (view.offsets) {
        offset = view.offsets[logical_expert];
        if (offset == kExpertAbsentOffset) return result;
    } else if (view.expert_indices) {
        physical = view.expert_indices[logical_expert];
        if (physical == UINT32_MAX) return result;
    }
    result.codes = reinterpret_cast<const std::uint8_t*>(view.codes) +
                   (view.offsets ? offset : std::uint64_t(physical) * view.code_stride);
    result.multiplier = view.matrix_scales[std::uint64_t(physical) * view.matrix_scale_stride];
    if (KnownCsf || view.layout == QuantLayout::Csf32) {
        result.csf            = view.csf.first;
        result.csf.bases      = csf32_detail::advance(result.csf.bases,
                                                      std::uint64_t(physical) * view.csf.bases_stride);
        result.csf.nibbles    = csf32_detail::advance(result.csf.nibbles, std::uint64_t(physical) *
                                                                              view.csf.nibbles_stride);
        result.csf.offsets    = csf32_detail::advance(result.csf.offsets, std::uint64_t(physical) *
                                                                              view.csf.offsets_stride);
        result.csf.page_bases = csf32_detail::advance(
            result.csf.page_bases, std::uint64_t(physical) * view.csf.page_bases_stride);
    } else {
        result.scales = reinterpret_cast<const std::uint8_t*>(view.scales) +
                        (view.offsets ? offset : std::uint64_t(physical) * view.scale_stride);
    }
    result.valid = true;
    return result;
}

} // namespace ninfer::ops::common
