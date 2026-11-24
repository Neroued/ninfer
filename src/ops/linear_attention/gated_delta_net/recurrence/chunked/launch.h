#pragma once

#include "core/device.h"
#include "ops/linear_attention/gated_delta_net/recurrence/launch.h"
#include "ops/linear_attention/gated_delta_net/recurrence/chunked/workspace.h"

namespace ninfer::ops::detail::gated_delta_net::chunked {
struct PackedChunks {
    RaggedSegments segments;
    int total_chunks;
    int chunked_segments;
    ChunkIndexEntry inline_index[kInlineSeqs];
    ChunkIndexEntry* device_index;

    __device__ const ChunkIndexEntry* index() const {
        return device_index ? device_index : inline_index;
    }
};

void launch_build_chunk_index(RaggedSegments segments, int first_chunked, ChunkIndexEntry* index,
                              cudaStream_t stream);
void launch_prepare_ragged(const RecurrenceOperands& args, const PackedChunks& packed, QkChunk* qk,
                           ControlChunk* control, bool normalize_qk, cudaStream_t stream);
void launch_recurrence_ragged(const RecurrenceOperands& args, const PackedChunks& packed,
                              const QkChunk* qk, const ControlChunk* control, int value_tile,
                              cudaStream_t stream);

void launch_prepare(const RecurrenceOperands& args, QkChunk* qk, ControlChunk* control,
                    bool normalize_qk, cudaStream_t stream);
void launch_recurrence(const RecurrenceOperands& args, const QkChunk* qk,
                       const ControlChunk* control, int value_tile, cudaStream_t stream);
} // namespace ninfer::ops::detail::gated_delta_net::chunked
