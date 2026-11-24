#pragma once

#include "ops/linear_attention/gated_delta_net/recurrence/chunked/launch.h"
#include <cub/block/block_scan.cuh>

namespace ninfer::ops::detail::gated_delta_net::chunked {

struct ChunkCount {
    int segments;
    int chunks;
};

struct AddChunkCounts {
    __device__ ChunkCount operator()(ChunkCount lhs, ChunkCount rhs) const {
        return {lhs.segments + rhs.segments, lhs.chunks + rhs.chunks};
    }
};

// One CTA emits the compact index in original segment order. It can run alongside
// the short recurrent CTAs because they only read the caller's sequence metadata.
template <int BlockX, int BlockY = 1>
__device__ void build_chunk_index(RaggedSegments segments, int first_chunked,
                                  ChunkIndexEntry* index) {
    using Scan = cub::BlockScan<ChunkCount, BlockX, cub::BLOCK_SCAN_WARP_SCANS, BlockY>;
    __shared__ typename Scan::TempStorage storage;
    const int tid = threadIdx.y * BlockX + threadIdx.x;
    ChunkCount carry{};
    for (std::int64_t base = 0; base < segments.count; base += BlockX * BlockY) {
        const auto s     = base + tid;
        const int length = s < segments.count ? segments.lengths[s] : 0;
        const int chunks = length >= first_chunked ? chunk_count(length) : 0;
        ChunkCount prefix, total;
        Scan(storage).ExclusiveScan({int(chunks != 0), chunks}, prefix, ChunkCount{},
                                    AddChunkCounts{}, total);
        if (chunks != 0) {
            index[carry.segments + prefix.segments] = {int(s), carry.chunks + prefix.chunks};
        }
        carry = AddChunkCounts{}(carry, total);
        __syncthreads();
    }
}

} // namespace ninfer::ops::detail::gated_delta_net::chunked
