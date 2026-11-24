#pragma once

#include "core/tensor.h"
#include "core/layout.h"
#include "ops/common/numeric/math.h"
#include "ops/linear_attention/kimi_delta_attention/operands.h"

namespace ninfer::ops::detail::kimi_delta_attention {
inline constexpr int kChunkSize  = 16;
inline constexpr int kInlineSeqs = 64;

struct RaggedSegments {
    const int* cu_seqlens;
    const int* lengths;
    int count;
};

struct ChunkWindow {
    int begin, end;
    int first_segment, segments, longest_segment;
};

struct ChunkIndexEntry {
    int segment;
    int first_chunk;
};

struct PackedChunks {
    ChunkWindow window;
    RaggedSegments segments;
    const ChunkIndexEntry* device_index;
    ChunkIndexEntry inline_index[kInlineSeqs];

    __device__ const ChunkIndexEntry* index() const {
        return device_index ? device_index : inline_index;
    }
};

__host__ __device__ inline int chunk_count(int tokens) {
    return common::div_up(tokens, kChunkSize);
}

// One producer/consumer packet. The vector and square tiles use the swizzles below in both
// global and shared memory, so recurrence can asynchronously copy whole packets without repacking.
struct alignas(256) Chunk {
    __nv_bfloat16 kd[kChunkSize * kStateDim];
    __nv_bfloat16 qd[kChunkSize * kStateDim];
    __nv_bfloat16 kr[kChunkSize * kStateDim];
    float gamma[kStateDim];
    float solve[kChunkSize * kChunkSize];
    float mqk[kChunkSize * kChunkSize];
};

static_assert(sizeof(Chunk) == 14848);

__host__ __device__ constexpr int vector_index(int row, int col) {
    return row * kStateDim + (col ^ (row * 8));
}

// Kr time rows are interleaved {0,4,1,5,2,6,3,7} for transposed BF16 MMA loads.
__host__ __device__ constexpr int restored_index(int row, int col) {
    const int physical = (row & ~7) + (row & 3) * 2 + ((row >> 2) & 1);
    return vector_index(physical, col);
}

__host__ __device__ constexpr int square_index(int row, int col) {
    return row * kChunkSize + (col ^ (((row >> 1) & 3) * 4));
}

struct ChunkWorkspace {
    LayoutRegion packets;
    int window_chunks;
    LayoutRegion index;
    std::size_t bytes;
};

inline std::size_t chunk_packet_bytes(int value_heads, int chunks) {
    return Tensor::metadata(DType::U8, {int(sizeof(Chunk)), chunks, value_heads}).bytes();
}

inline std::size_t chunked_workspace_bytes(int value_heads, int tokens) {
    return chunk_packet_bytes(value_heads, chunk_count(tokens));
}

inline ChunkWorkspace chunk_workspace(int value_heads, int chunks, int segments) {
    LayoutBuilder builder;
    ChunkWorkspace layout{};
    layout.window_chunks = chunks;
    layout.packets =
        builder.add(chunk_packet_bytes(value_heads, chunks), alignof(Chunk), "kda.packets");
    if (segments > kInlineSeqs) {
        layout.index = builder.add(std::size_t(segments) * sizeof(ChunkIndexEntry),
                                   alignof(ChunkIndexEntry), "kda.chunk_index");
    }
    layout.bytes = builder.finish(1, "KDA chunk workspace");
    return layout;
}

} // namespace ninfer::ops::detail::kimi_delta_attention
