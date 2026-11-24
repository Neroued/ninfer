#pragma once

#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr int kInlineSeqs       = 64;
inline constexpr int kRaggedShortWidth = 16;
inline constexpr int kRaggedNarrowTile = 64;

template <class Entry>
struct CausalSegmentMap {
    const Entry* device_entries = nullptr;
    int count                   = 0;
    int device_id               = 0;
    Entry inline_entries[kInlineSeqs]{};

#if defined(__CUDACC__)
    __device__ __forceinline__ Entry entry(int index) const {
        return device_entries ? device_entries[index] : inline_entries[index];
    }
#endif
};

struct CausalShortMetadata {
    CausalSegmentMap<int> segments;
    const int* cu_seqlens;
    const int* seq_lengths;
    int visible_capacity;
    int reserved = 0;
};

struct CausalTiledSegment {
    int segment;
    int tile_end;
    int token_end;
};

struct CausalTiledMetadata {
    CausalSegmentMap<CausalTiledSegment> segments;
    const int* cu_seqlens;
    const int* seq_lengths;
    const std::int64_t* tables;
    const int* table_rows;
    int table_stride;
    int tile_count;
    int tile_rows = 0;
    int heads_first = 0;

#if defined(__CUDACC__)
    template <int CausalTiledSegment::*End>
    __device__ __forceinline__ int find(int index) const {
        int first = 0, last = segments.count - 1;
        while (first < last) {
            const int mid = first + (last - first) / 2;
            if (index < segments.entry(mid).*End)
                last = mid;
            else
                first = mid + 1;
        }
        return first;
    }

    __device__ __forceinline__ const std::int64_t* block_table(int segment) const {
        return tables + std::int64_t(table_rows[segment]) * table_stride;
    }
#endif
};

} // namespace ninfer::ops::detail
