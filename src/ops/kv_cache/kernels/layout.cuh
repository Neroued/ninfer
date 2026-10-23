#pragma once

#include "core/paged_kv_cache.h"

#include <cstdint>

namespace ninfer::ops::detail::kv {

inline constexpr int kPageShift   = 6;
inline constexpr int kPageMask    = kPagedKVPageSize - 1;
inline constexpr int kHeadDim     = 256;
inline constexpr int kInt8Group   = 64;
inline constexpr int kInt8Groups  = kHeadDim / kInt8Group;
inline constexpr int kNvfp4Group  = 16;
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

__device__ __forceinline__ std::int64_t lookup_page(const std::int64_t* table, int position) {
    return table[position >> kPageShift];
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
        static_assert(ScaleExtent == 1);
        return scale_offset(record, head, 0, offset);
    }
};

using Int8G64Layout  = QuantizedRowLayout<256, 4>;
using Fp8RowLayout   = QuantizedRowLayout<256, 1>;
using Nvfp4G16Layout = QuantizedRowLayout<128, 16, 2, 1>;

} // namespace ninfer::ops::detail::kv
