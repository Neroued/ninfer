#pragma once

#include "core/device.h"
#include "core/tensor.h"
#include "core/arena.h"
#include "ninfer/ops/sampling.h"

namespace ninfer::ops {

enum class CandidateSelectorInput { Candidates, GatheredKeys };

// Capacity for every K/B pair in the inclusive intervals, K=1..15 and B=1..8.
[[nodiscard]] std::size_t
candidate_selector_path_workspace_capacity_bytes(CandidateSelectorInput input,
                                                 std::int32_t min_steps, std::int32_t max_steps,
                                                 std::int32_t min_batch, std::int32_t max_batch);

/**
 * Op: candidate_selector_path
 *
 * For K in [1,15] and B in [1,8], the inputs are contiguous candidate_ids I32 [16,K,B],
 * unary_scores FP32 [16,K,B], projected_hidden BF16 [256,K,B], anchors I32 [B],
 * predecessor_codebook and successor_codebook BF16 [256,V], base_positions I32 [B], and a
 * device-resident SamplingConfig[B]. Candidate rank is the fastest axis. The 16 candidate ids in
 * each row are distinct, and all candidate and anchor token ids lie in [0,token_domain), with
 * 0<token_domain<=V. Both codebooks have the same physical V. Input scores and computed edges
 * are finite. The caller/producer guarantees the Device ID value contract.
 *
 * Starting with predecessor=anchors[b], each position i in [0,K) computes:
 *
 *   edge[c] = unary_scores[c,i,b]
 *           + sum_r predecessor_codebook[r,predecessor]
 *                   * projected_hidden[r,i,b]
 *                   * successor_codebook[r,candidate_ids[c,i,b]].
 *
 * A row with configs[b].temperature<=0 selects the lowest candidate rank attaining max(edge) and
 * writes its exact one-hot distribution. A positive-temperature row writes the FP32 softmax of
 * edge/temperature, then draws a candidate with counter key
 * (configs[b].seed,base_positions[b]+i,kSamplePurposeDFlash2Proposal). The selected global id is
 * written to drafts[i,b] and becomes the next predecessor. The Op ignores all other
 * SamplingConfig fields and never updates token_counts.
 *
 * drafts is contiguous I32 [K,B] and proposal_q is contiguous FP32 [16,K,B]. Both outputs are
 * completely overwritten. Inputs, outputs, codebooks, and the config array must be pairwise
 * non-overlapping. The Op has no persistent state or internal allocation. Caller workspace is
 * transient and must not overlap any input or output.
 */
void candidate_selector_path(const Tensor& candidate_ids, const Tensor& unary_scores,
                             const Tensor& projected_hidden, const Tensor& anchors,
                             const Tensor& predecessor_codebook, const Tensor& successor_codebook,
                             std::int32_t token_domain, const Tensor& base_positions,
                             const SamplingConfig* configs, Tensor& drafts, Tensor& proposal_q,
                             MemoryArena& workspace, CudaExecutionView execution);

/**
 * Gathered input form. keys is contiguous I64 [16,K,B,2], using vocabulary.h's lossless
 * FP32 score/global-ID keys and unsigned ordering. Each local list is descending and zero-padded;
 * ranks have disjoint IDs in [0,token_domain), and every column has nonempty global support.
 * Merge the global top 16 before applying the same conditional path mathematics and RNG above.
 *
 * candidate_ids is completely overwritten I32 [16,K,B], ordered by score descending and then
 * global ID ascending. Missing entries are INT_MAX and have exact zero proposal_q. The selected
 * draft always has positive q. The two ranks supply identical global inputs, seed and logical
 * positions; the Op has no rank-dependent random key. Config fields other than temperature/seed
 * remain ignored. Anchors are valid IDs. All other operands, ownership, storage and effects follow
 * the separate-input form. Merge is part of path execution, with no standalone merge allocation.
 */
void candidate_selector_path(const Tensor& keys, const Tensor& projected_hidden,
                             const Tensor& anchors, const Tensor& predecessor_codebook,
                             const Tensor& successor_codebook, std::int32_t token_domain,
                             const Tensor& base_positions, const SamplingConfig* configs,
                             Tensor& candidate_ids, Tensor& drafts, Tensor& proposal_q,
                             MemoryArena& workspace, CudaExecutionView execution);

} // namespace ninfer::ops
