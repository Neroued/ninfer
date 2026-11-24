#pragma once

#include "core/layout.h"
#include "ops/common/numeric/math.h"
#include "ops/linear_attention/gated_delta_net/operands.h"
#include <cuda_bf16.h>

namespace ninfer::ops::detail::gated_delta_net::chunked {
inline constexpr int kChunkSize = 16;
inline constexpr float kNormEps = 1.0e-6F;

// Q/K are independent of value-head gates. One packet is shared by the entire Q/K group.
struct alignas(256) QkChunk {
    __nv_bfloat16 q[kChunkSize * kStateDim];
    __nv_bfloat16 k[kChunkSize * kStateDim];
};

struct alignas(256) ControlChunk {
    // With G_i = sum_{t<=i} g_t and L_ij = beta_i exp(G_i-G_j) dot(K_i,K_j), j<i:
    // solve = (I+L)^-1 diag(beta), mqk_ij = exp(G_i-G_j) dot(Q_i,K_j), j<=i.
    float solve[kChunkSize * kChunkSize];
    float mqk[kChunkSize * kChunkSize];
    float prefix[kChunkSize]; // exp(G_i)
    float suffix[kChunkSize]; // exp(G_last-G_i)
    std::uint32_t padding[32];
};

static_assert(kChunkSize == 16);
static_assert(sizeof(QkChunk) == 8192);
static_assert(sizeof(ControlChunk) == 2304);

__host__ __device__ inline int chunk_count(int tokens) {
    return common::div_up(tokens, kChunkSize);
}

__host__ __device__ constexpr int vector_index(int row, int col) {
    return row * kStateDim + (col ^ (row * 8));
}

__host__ __device__ constexpr int square_index(int row, int col) {
    return row * kChunkSize + (col ^ (((row >> 1) & 3) * 4));
}

inline constexpr int kInlineSeqs = 64;

struct ChunkIndexEntry {
    int segment;
    int first_chunk;
};

struct WorkspaceLayout {
    TensorRegion qk;
    TensorRegion control;
    LayoutRegion index;
    std::size_t total_bytes;
};

inline WorkspaceLayout packed_workspace_layout(int qk_heads, int value_heads, int chunks,
                                               int chunked_segments) {
    LayoutBuilder builder;
    WorkspaceLayout layout{};
    layout.qk =
        builder.add_tensor(DType::U8, {int(sizeof(QkChunk)), chunks, qk_heads}, 256, "gdn.qk");
    layout.control = builder.add_tensor(DType::U8, {int(sizeof(ControlChunk)), chunks, value_heads},
                                        256, "gdn.control");
    if (chunked_segments > kInlineSeqs) {
        layout.index = builder.add(std::size_t(chunked_segments) * sizeof(ChunkIndexEntry),
                                   alignof(ChunkIndexEntry), "gdn.chunk_index");
    }
    layout.total_bytes = builder.finish(1, "GDN chunk workspace");
    return layout;
}

inline WorkspaceLayout workspace_layout(int qk_heads, int value_heads, int tokens) {
    return packed_workspace_layout(qk_heads, value_heads, chunk_count(tokens), 1);
}

} // namespace ninfer::ops::detail::gated_delta_net::chunked
