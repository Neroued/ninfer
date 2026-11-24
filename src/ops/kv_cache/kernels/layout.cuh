#pragma once

#include "core/paged_kv_cache.h"

#include <cstdint>
#include <type_traits>

namespace ninfer::ops::detail::kv {

inline constexpr int kPageShift = 6;
inline constexpr int kPageMask  = kPagedKVPageSize - 1;
inline constexpr int kHeadDim   = 256;
inline constexpr int kInt8Group =
    kv_cache_format_info(KvCacheVectorFormat::Int8Group64F16).group_size;
inline constexpr int kInt8Groups = kHeadDim / kInt8Group;
inline constexpr int kNvfp4Group =
    kv_cache_format_info(KvCacheVectorFormat::Nvfp4Group16Ue4m3).group_size;
inline constexpr int kNvfp4Groups = kHeadDim / kNvfp4Group;

static_assert(kPagedKVPageSize == (1 << kPageShift));

template <int Heads>
struct D256Geometry {
    static_assert(Heads == 2 || Heads == 4);
    static constexpr int KVHeads = Heads;
};

struct DirectMetadata {
    const std::int64_t* table;

    __device__ __forceinline__ int valid_tokens(int width) const { return width; }

    __device__ __forceinline__ const std::int64_t* block_table() const { return table; }
};

template <bool Masked>
struct BatchMetadata {
    static constexpr bool masked = Masked;
    const std::int64_t* tables;
    const std::int32_t* valid_columns;
    const std::int32_t* table_rows;
    std::int32_t table_stride;
    std::int32_t reserved = 0; // Keep every captured argument byte defined.

    __device__ __forceinline__ int valid_tokens(int width) const {
        if constexpr (Masked) {
            const int valid = valid_columns[0];
            return valid <= 0 ? 0 : (valid < width ? valid : width);
        }
        return width;
    }

    __device__ __forceinline__ const std::int64_t* block_table() const {
        return tables + static_cast<std::int64_t>(table_rows[0]) * table_stride;
    }
};

struct RaggedMetadata {
    const std::int64_t* tables;
    const std::int32_t* token_to_seq;
    const std::int32_t* table_rows;
    const std::int32_t* seq_lengths; // Only read when filtering a mixed batch.
    int table_stride;
    int max_length; // Zero writes every packed token.

    __device__ __forceinline__ int valid_tokens(int width) const { return width; }
};

__device__ __forceinline__ std::int64_t lookup_page(const std::int64_t* table, int position) {
    return table[position >> kPageShift];
}

template <class Metadata>
__device__ __forceinline__ bool write_token_page(Metadata metadata, const int* positions, int token,
                                                 int lane, int& position, std::int64_t& record) {
    if constexpr (std::is_same_v<Metadata, RaggedMetadata>) {
        int segment = lane == 0 ? metadata.token_to_seq[token] : 0;
        if (metadata.max_length) {
            const int length = lane == 0 ? metadata.seq_lengths[segment] : 0;
            if (__shfl_sync(0xffffffffU, length, 0) > metadata.max_length) return false;
        }
        if (lane == 0) {
            position          = positions[token];
            const auto* table = metadata.tables +
                                std::int64_t(metadata.table_rows[segment]) * metadata.table_stride;
            record = lookup_page(table, position);
        }
        position = __shfl_sync(0xffffffffU, position, 0);
    } else {
        position = positions[0] + token;
        record   = lane == 0 ? lookup_page(metadata.block_table(), position) : 0;
    }
    return true;
}

// The table stores record byte offsets; each plane base already includes its record-local offset.
template <int Leading, int ElementBytes>
__device__ __forceinline__ std::int64_t record_offset(std::int64_t record, int head, int row,
                                                      int component) {
    return record / ElementBytes +
           static_cast<std::int64_t>(Leading) * (kPagedKVPageSize * head + row) + component;
}

template <int Leading, int ElementBytes>
__device__ __forceinline__ std::int64_t record_offset(const std::int64_t* table, int head,
                                                      int position, int component) {
    return record_offset<Leading, ElementBytes>(lookup_page(table, position), head,
                                                position & kPageMask, component);
}

// One State field: [feature, ring position, head]. The caller applies its record byte offset.
template <int Dim, int Capacity>
__device__ __forceinline__ std::int64_t cyclic_offset(int position, int head, int padded_capacity,
                                                      int component = 0) {
    static_assert((Capacity & (Capacity - 1)) == 0);
    return component +
           static_cast<std::int64_t>(Dim) *
               ((position & (Capacity - 1)) + static_cast<std::int64_t>(padded_capacity) * head);
}

template <class Geometry>
__device__ __forceinline__ std::int64_t source_offset(int head, int component, int token) {
    return component + static_cast<std::int64_t>(kHeadDim) *
                           (head + static_cast<std::int64_t>(Geometry::KVHeads) * token);
}

template <int CodeExtent, int ScaleExtent, int ValuesPerCode = 1, int ScaleBytes = 2>
struct QuantizedRowLayout {
    __device__ static __forceinline__ std::int64_t code_offset(std::int64_t record, int head,
                                                               int component, int offset) {
        return record_offset<CodeExtent, 1>(record, head, offset, component / ValuesPerCode);
    }

    __device__ static __forceinline__ std::int64_t scale_offset(std::int64_t record, int head,
                                                                int group, int offset) {
        return record_offset<ScaleExtent, ScaleBytes>(record, head, offset, group);
    }

    __device__ static __forceinline__ std::int64_t scale_offset(std::int64_t record, int head,
                                                                int offset) {
        return scale_offset(record, head, 0, offset);
    }
};

using Int8G64Layout  = QuantizedRowLayout<256, 4>;
using Fp8G128Layout  = QuantizedRowLayout<256, 2, 1, 1>;
using Nvfp4G16Layout = QuantizedRowLayout<128, 16, 2, 1>;

} // namespace ninfer::ops::detail::kv
