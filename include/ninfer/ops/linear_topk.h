#pragma once

#include "core/device.h"
#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"


#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

// Capacity for every extent in the inclusive interval; invalid profiles or intervals throw.
[[nodiscard]] std::size_t linear_topk_workspace_capacity_bytes(QType qtype, std::int32_t head_rows,
                                                               std::int32_t input_rows,
                                                               std::int32_t min_columns,
                                                               std::int32_t max_columns);

/**
 * scores[:,t], ids[:,t] = top16({(W[v,:] * hidden[:,t], v) : v < valid_rows}).
 *
 * hidden [K,T] is contiguous BF16, T > 0. W [N,K] uses a registered full-head
 * Q8_G32_FP16 or FP8_E4M3FN_ROW_BF16 geometry; valid_rows must be 248077.
 * Physical rows at or beyond valid_rows never participate.
 * candidate_scores is contiguous FP32 [16,T]; candidate_ids is contiguous I32 [16,T].
 * Input and output tensors require 16-byte alignment. Rank is the fastest-moving axis.
 * Candidates are ordered by descending computed score, with ties resolved by lower global id.
 * Projection uses A16; scores are published in FP32.
 *
 * Overwrites both outputs and preserves all inputs. Inputs, outputs, weight planes and caller
 * workspace must not overlap. Workspace is scoped to the call; no persistent state or allocation.
 */
void linear_topk(const Tensor& hidden, const Weight& head, std::int32_t valid_rows,
                 Tensor& candidate_ids, Tensor& candidate_scores, MemoryArena& workspace,
                 CudaExecutionView execution);

/**
 * Mapped-head form: use every row of a registered Q4_G64_FP16 head and map its local row
 * through contiguous I32 row_to_global_ids [N]. The caller supplies distinct ids in [0,248077).
 * Global ids determine both output ids and tie-breaking; all other requirements are as above.
 */
void linear_topk(const Tensor& hidden, const Weight& head, const Tensor& row_to_global_ids,
                 Tensor& candidate_ids, Tensor& candidate_scores, MemoryArena& workspace,
                 CudaExecutionView execution);

// Capacity for the key-producing overload below, for every positive column extent in
// [min_columns,max_columns]. The query includes activation encoding and all candidate merges.
[[nodiscard]] std::size_t linear_topk_workspace_capacity_bytes(QType qtype, std::int32_t head_rows,
                                                               std::int32_t input_rows,
                                                               LinearPolicy policy,
                                                               std::int32_t min_columns,
                                                               std::int32_t max_columns);

/**
 * Local vocabulary top-16 as lossless FP32 score/global-ID keys, following vocabulary.h.
 * hidden is contiguous BF16 [K,D*B]; keys is contiguous I64 [16,D,B], D>0, B=1..8,
 * with candidate rank fastest. Both bases are 16-byte aligned.
 * Row v of native W [N,K] has global ID token_begin+v; only IDs below token_domain
 * participate. token_begin>=0, token_domain>0, token_begin+N<=INT_MAX. Empty support
 * produces zero keys. Lists are descending by computed FP32 score, ties by smaller global ID;
 * signed zero is canonicalized. Consumed inputs and scores are finite.
 *
 * Policy controls activation permission; MXFP8 requires an A8 permission. This overload uses
 * contiguous vocabulary shards, including Q4, and takes no row-to-ID map. Projection scores
 * have no BF16/FP16 boundary. Every output is overwritten and inputs are preserved.
 * All input, output, weight planes and live caller workspace are disjoint on execution.device().
 * Workspace is call-scoped, covers the complete operation and has no retained storage or hidden
 * device allocation. Descriptors/scalars are capture-time values; Graph replay reads current data.
 */
void linear_topk(const Tensor& hidden, const Weight& head, std::int32_t token_begin,
                 std::int32_t token_domain, LinearPolicy policy, Tensor& keys,
                 MemoryArena& workspace, CudaExecutionView execution);

} // namespace ninfer::ops
