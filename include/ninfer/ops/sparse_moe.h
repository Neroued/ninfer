#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

struct SparseMoeWeights {
    Weight router_shared_gate;
    Weight routed_gate_up;
    Weight routed_down;
    Weight shared_gate_up;
    Weight shared_down;
};

enum class SparseMoeEpilogue : std::uint8_t {
    AddResidual,
};

struct WeightPrefetchSpan {
    const void* data  = nullptr;
    std::size_t bytes = 0;
};

/**
 * Returns the transient capacity required by SparseMoe for every T in the inclusive
 * [min_tokens,max_tokens] interval. The routed QTypes are the fixed implementation profile.
 * Invalid profiles or intervals throw.
 */
[[nodiscard]] std::size_t sparse_moe_workspace_capacity_bytes(QType routed_gate_up,
                                                              QType routed_down,
                                                              std::int32_t min_tokens,
                                                              std::int32_t max_tokens);

/**
 * Closed sparse-MoE Op for the exact future 35B-A3B geometry.
 *
 * For contiguous BF16 x [2048,T] and destination [2048,T] with T>0, 256 routed experts, top-8
 * selection, and one always-on shared expert, this Op owns router projection and selection,
 * selected routed and shared SwiGLU projections, down projections, their merge, and the
 * AddResidual epilogue independently for every token column. At an exact top-8 boundary tie the
 * lower expert id wins. destination is the only observable mutation: its incoming value is the
 * residual and its outgoing value is the BF16 sparse-MoE result plus that residual.
 *
 * The complete mathematical oracle starts from represented BF16 inputs, exact stored-weight
 * decode, and evaluates the logical formula naively in FP32/FP64. Scores, route weights, expert
 * activations, workspace representation, reduction association, and scale placement are private
 * execution choices rather than semantic rounding boundaries.
 *
 * The five weights have the exact registered shapes: BF16 router/shared gate [257,2048], routed
 * gate/up [256*1024,2048], routed down [256*2048,512], shared gate/up [1024,2048], and shared down
 * [2048,512]. Admitted codec profiles are Q4+Q5, Q4+Q6, and W8+W8 for the two routed banks; both
 * shared banks are W8. Expert e directly selects its stored row spans; no selected-weight gather
 * or repack occurs.
 *
 * Every positive T is supported.
 *
 * x, destination, all weight planes, and live workspace must be pairwise non-overlapping.
 * Execution is enqueued on stream without host synchronization. Workspace is caller-owned,
 * graph-stable transient storage and carries no state beyond the call.
 */
/**
 * `next_prefetch` names the weight payload the next decode-step consumer will stream. The D4
 * epilogue, whose tail runs on an otherwise idle bus, issues fire-and-forget L2 prefetches for it,
 * clamped to the op's own cap. Purely a cache hint: no value is read through it and the emitted
 * tokens are unaffected. An empty span disables the hint, which is what every non-decode route
 * passes.
 */
void sparse_moe(const Tensor& x, const SparseMoeWeights& weights, SparseMoeEpilogue epilogue,
                Tensor& destination, WorkspaceArena& workspace, cudaStream_t stream,
                WeightPrefetchSpan next_prefetch = {});

} // namespace ninfer::ops
