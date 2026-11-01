#pragma once

#include "core/device.h"
#include "core/tensor.h"
#include <cstdint>

namespace ninfer::ops {

struct CausalPairInputs {
    Tensor hidden, previous_hidden;
    Tensor ids, positions;
    Tensor rope, previous_rope;
    Tensor prefix, skip_first;
    Tensor rows;
};

struct CausalPairOutputs {
    Tensor hidden, ids, positions, lengths;
    Tensor rope;
};

// Pair adopted tokens with their preceding hidden/coordinates, selecting input row b=rows[r]
// for output row r; absent rows means identity and C=B. For n=max(prefix[b]-skip_first[b],0),
// j<n, i=j+skip_first[b]: output hidden[:,j,r] is hidden[:,i-1,b] if i>0, previous_hidden[:,b]
// otherwise; IDs are ids[i,b], positions are positions[i,b]-1, and optional coordinates are
// rope[i-1,:,b] if i>0, previous_rope[:,b] otherwise. Writes lengths[r]=n. rows[r]=-1 writes an
// all-zero row without reading input metadata; physical tails j>=n are zero and read no inputs.
//
// BF16 hidden[D,W,B], previous_hidden[D,B], output hidden[D,P,C] have contiguous features;
// bases and outer strides are 16-byte aligned, D>0 is divisible by eight. I32 IDs/positions are
// [W,B] in, [P,C] out. Optional I32 rope [W,A,B], previous_rope [A,B], output rope [P,A,C]
// either all exist, A=1 or 3, or are all absent. These tensors have positive, element-aligned,
// non-overlapping strides. prefix/skip_first[B], optional rows[C], output lengths[C] are
// contiguous I32. W,P>=1, B,C=1..8. Caller supplies rows in {-1} union [0,B), 0<=prefix<=W,
// skip_first in {0,1}, n<=P, and positive consumed positions. Read-only inputs may alias;
// output ranges are mutually disjoint and disjoint from all inputs. Copies bits exactly,
// including coordinates. No workspace or persistent effects; Device controls are read on replay.
void prepare_causal_pairs(const CausalPairInputs& input, const CausalPairOutputs& output,
                          CudaExecutionView execution);

/**
 *   For row b and 0<=j<=K:
 *     verify_ids[j,b] = anchors[b]                         when j=0
 *                       drafts[j-1,b]                     when 0<j<=Pcur[b]
 *                       anchors[b]                        otherwise;
 *     positions[j,b]  = base_positions[b] + min(j,Pcur[b]).
 *
 *   All tensors are I32, contiguous except drafts, which permits positive, naturally aligned,
 *   non-overlapping strides. anchors/base_positions/current_extents are [B], drafts is
 *   [K,B] with K>=1 and B>=1, and verify_ids/positions are [K+1,B]. Each current extent is in
 *   [0,K]. Inputs and outputs do not overlap.
 *
 *   Writes every physical output element, including safe invalid-tail values. Inputs remain
 *   unchanged.
 *
 */
void prepare_verify_inputs(const Tensor& anchors, const Tensor& drafts,
                           const Tensor& base_positions, const Tensor& current_extents,
                           Tensor& verify_ids, Tensor& positions, CudaExecutionView execution);

/**
 * Prepare only the target verification ids when the caller already owns the matching position
 * matrix. Shapes and id semantics are identical to prepare_verify_inputs; the
 * existing positions remain untouched.
 */
void prepare_verify_ids(const Tensor& anchors, const Tensor& drafts, const Tensor& current_extents,
                        Tensor& verify_ids, CudaExecutionView execution);

/**
 *   For T=K+1 and each row b with A=accepted[b], L=licensed_counts[b]:
 *     alignment_ids[j,b] = verify_ids[j+1,b]  for 0<=j<A
 *                          next_anchors[b]     otherwise;
 *     remaining_after = max(remaining_budgets[b]-L,0);
 *     context_after   = max(max_context-updated_frontiers[b]-1,0);
 *     next_extents[b] = min(K,max(remaining_after-1,0),context_after);
 *     For S=max(K-1,1) and 0<=s<S:
 *       ar_positions[b,s]      = updated_frontiers[b]+s;
 *       ar_rope_positions[b,s] = ar_positions[b,s]+rope_deltas[b];
 *       ar_valid_columns[b,s]  = (s+1 < next_extents[b]).
 *
 *   verify_ids/alignment_ids are distinct contiguous I32 [K+1,B]. ar_positions,
 *   ar_rope_positions, and ar_valid_columns are I32 [B,max(K-1,1)] with contiguous rows and one
 *   shared step stride at least B; this permits an exact-B prefix of a fixed-capacity frame. All
 *   other tensors are contiguous I32 [B]. B>=1, 1<=K<=5, 0<=accepted[b]<=K,
 *   licensed_counts[b]=accepted[b]+1, updated_frontiers and remaining_budgets are non-negative,
 *   and max_context is positive. The Op writes every output slot, including safe invalid-tail
 *   values. Inputs remain unchanged. No workspace or other state is used.
 */
void prepare_autoregressive_round(const Tensor& verify_ids, const Tensor& next_anchors,
                                  const Tensor& accepted, const Tensor& updated_frontiers,
                                  const Tensor& remaining_budgets, const Tensor& licensed_counts,
                                  const Tensor& rope_deltas, Tensor& alignment_ids,
                                  Tensor& next_extents, Tensor& ar_positions,
                                  Tensor& ar_rope_positions, Tensor& ar_valid_columns,
                                  std::int32_t max_context, CudaExecutionView execution);

/**
 * Op: prepare a batch of finite anchor-and-mask query blocks.
 *
 * anchors, lengths, and valid_columns are distinct contiguous device I32 [B] vectors. ids and
 * positions are distinct contiguous I32 [W,B] outputs. For each batch row b:
 *
 *   ids[0,b]       = anchors[b]
 *   ids[i,b]       = mask_id,                         1 <= i < W
 *   positions[i,b] = lengths[b] + min(i, V[b] - 1),  0 <= i < W
 *
 * where V[b]=valid_columns[b] and 1 <= V[b] <= W. Repeating the last valid position makes the
 * invalid physical tail safe for downstream fixed-width execution; valid_columns remains the
 * authority that excludes those rows from mathematical work and persistent-state updates.
 *
 * The registered domain is W=1..16, B=1..8, and nonnegative mask_id. The caller guarantees
 * nonnegative lengths whose final valid positions fit I32. Inputs are unchanged, every output
 * element is overwritten, and the Op owns no workspace or persistent state.
 */
void prepare_masked_block(const Tensor& anchors, const Tensor& lengths, const Tensor& valid_columns,
                          std::int32_t mask_id, Tensor& ids, Tensor& positions,
                          CudaExecutionView execution);

/**
 * Prepare lane-owned BF16 prefixes as one compact ragged batch.
 *
 * source is BF16 [D,W,C], dense within each column with explicit column/lane strides.
 * lanes/starts/ends are I32 [B], destination is contiguous BF16 [D,W,B],
 * positions is contiguous I32 [W,B], and counts is I32 [B]. For N=ends[b]-starts[b], columns j<N
 * copy source[:,j,lanes[b]], use absolute position starts[b]+j, and publish counts[b]=N. The
 * physical tail j>=N is zero-filled and repeats the final live position (or starts[b] when N=0).
 * The caller guarantees 0<=N<=W and 0<=lanes[b]<C. D must be divisible by eight and all BF16
 * column starts must be 16-byte aligned. Physical W is independent of the live prefix N.
 * The three outputs are pairwise non-overlapping and do not overlap any input. Read-only inputs
 * may alias one another. Inputs remain unchanged; all
 * destination, position and count elements are overwritten exactly. The Op needs no transient
 * workspace and supports CUDA Graph capture/replay with changing device metadata.
 */
void prepare_ragged_prefix(const Tensor& source, const Tensor& lanes, const Tensor& starts,
                           const Tensor& ends, Tensor& destination, Tensor& positions,
                           Tensor& counts, CudaExecutionView execution);

} // namespace ninfer::ops
