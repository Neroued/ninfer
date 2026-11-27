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
// Build the expert-major prefix from the incoming compact hits, exactly once per packet.
void group_expert_hits(ExpertPacket, const CompactExpertAssignments&, Tensor& cursors,
                       CudaExecutionView);
// FP32 partials are already weighted. Add shared once, then apply the model epilogue.
void return_expert_rows(const ExpertPeerBuffers*, int peers, const Tensor& shared, Tensor& output,
                        MoeTokenValidity, MoeEpilogue, CudaExecutionView);
// Accumulates tx rows/hits and rx rows/hits per peer for an explicitly observed exchange.
void observe_expert_packets(const ExpertPeerBuffers*, int peers, Tensor& totals, CudaExecutionView);

} // namespace ninfer::ops
