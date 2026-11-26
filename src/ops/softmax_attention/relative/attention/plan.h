#pragma once

#include "ninfer/ops/relative_attention.h"

namespace ninfer::ops::relative {

inline constexpr auto kOp = "relative_softmax_attention";
inline constexpr int kDim = 128, kHeads = 24, kKvHeads = 8, kGroup = 3;
inline constexpr int kRelDim = 16, kExtent = 1024, kWindow = 1024;
inline constexpr int kMaxWidth = 16, kRaggedRows = 64, kKeyTile = 32;
inline constexpr int kSwaVerifyKeyTile = 16;
inline constexpr int kScratchAlignment = 16, kMaxSplits = 128;

struct TileInterval {
    int first, end;
};

// Tile offsets relative to the first aligned tile in an attention interval.
#if defined(__CUDACC__)
__host__ __device__
#endif
    constexpr TileInterval
    split_tile_interval(int tiles, int splits, int split) {
    const int each = tiles / splits, remainder = tiles % splits;
    const int first = split * each + (split < remainder ? split : remainder);
    return {first, first + each + (split < remainder)};
}

struct DecodePlan {
    int batch, width, splits;
    bool simt;
    [[nodiscard]] std::size_t rows() const;
    [[nodiscard]] std::size_t bytes() const;
};

struct RaggedTile {
    int first, count, sequence, sequence_first;
};

void validate_profile(AttentionHeadGeometry, KvCacheStorage, int sms);
void validate_window(std::uint32_t window);
DecodePlan select_decode(bool swa, KvCacheStorage, ContextBucket, int batch, int width, int sms);

// The increments of this virtual prefix cover ceil(length / kRaggedRows)
// with at most one vacant tile per segment.
#if defined(__CUDACC__)
__host__ __device__
#endif
    constexpr int
    ragged_tiles(int tokens, int segments) {
    return (tokens - segments) / kRaggedRows + segments;
}

struct Plane {
    const unsigned char* base;
    const std::int64_t* offsets;
};

struct ChunkView {
    const unsigned char *k{}, *v{}, *ks{}, *vs{};
};

struct CacheView {
    Plane k, v, ks, vs;
    const int* table_rows;
    int stride; // Cyclic slots per head or logical pages per table row.
    int table_count;
    ChunkView chunk{};
};

struct Query {
    const std::uint16_t* q;
    const std::uint16_t* r;
    const std::uint16_t* projection;
    const std::uint16_t* gate;
    const std::uint16_t* sink;
    const int* positions;
    std::uint16_t* out;
    float scale;
};

struct Partials {
    float* values;
    float* lse;
};

void launch_decode(Query, CacheView, bool swa, KvCacheStorage, const int* valid, DecodePlan,
                   Partials, cudaStream_t);
void launch_ragged(Query, CacheView, bool swa, KvCacheStorage, const int* prefix, int tokens,
                   int segments, RaggedTile*, int tiles, cudaStream_t);

} // namespace ninfer::ops::relative
