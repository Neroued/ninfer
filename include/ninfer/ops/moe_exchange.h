#pragma once

#include "ninfer/ops/moe_compact.h"

namespace ninfer::ops {

// Prepared, device-resident addressing. offsets[e:e+1] selects every intermediate slice
// required by logical expert e. Each hit names a slot in the destination's local bank.
struct ExpertDestination {
    std::int32_t peer, slot;
};

struct ExpertHit {
    std::int32_t row, slot;
    float coefficient;
};

struct ExpertPacket {
    std::int32_t* counts;
    std::uint16_t* hidden;
    ExpertHit* hits;
};

struct ExpertPeerBuffers {
    ExpertPacket send, receive;
    std::int32_t* source_to_row; // Sender-private inverse, -1 for no hit on this peer.
    float* returned;
    float* partial; // Weighted rows produced by this rank, sent back to the source.
};

// A compute window over device-resident receive segments. A grouped row ID is
// segment * columns + packet_row. Each segment has its own valid prefix.
struct ExpertPacketBatch {
    const ExpertPeerBuffers* peers;
    int segments, columns;
    Device device;
};

struct ExpertDispatchView {
    const std::int32_t* offsets;
    const ExpertDestination* destinations;
};

// Borrowed prepared view on the execution device. Each call uses its owner's ordered stream.
// Zero source has no hidden or routes and still initializes every outbound packet.
void expert_block_validity(MoeTokenValidity, int base, Tensor& flags, CudaExecutionView);
void pack_expert_rows(const Tensor* hidden, const ExpertRoutes*, MoeTokenValidity,
                      ExpertDispatchView, ExpertPeerBuffers* device_peers, int peers,
                      int block_columns, CudaExecutionView);
// Group every source in the window by local expert, retaining its return row.
void group_expert_hits(ExpertPacketBatch, const CompactExpertAssignments&, Tensor& cursors,
                       CudaExecutionView);
// Consume the segmented BF16 rows directly and write each weighted FP32 partial
// into its source's return segment. No concatenation or host count read is required.
void compact_routed_swiglu(ExpertPacketBatch, const RoutedSwiGLUProfile&, const CudaSwiGLUExperts&,
                           const CompactExpertAssignments&, MemoryArena&, CudaExecutionView);
// FP32 partials are already weighted. Add shared once, then apply the model epilogue.
void return_expert_rows(const ExpertPeerBuffers*, int peers, const Tensor& shared, Tensor& output,
                        MoeTokenValidity, MoeEpilogue, CudaExecutionView,
                        const Tensor* shared_gate = nullptr);
// Accumulates tx rows/hits and rx rows/hits per peer for an explicitly observed exchange.
void observe_expert_packets(const ExpertPeerBuffers*, int peers, Tensor& totals, CudaExecutionView);

} // namespace ninfer::ops
