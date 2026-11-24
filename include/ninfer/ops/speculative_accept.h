#pragma once

#include "core/device.h"
#include "core/tensor.h"
#include "ninfer/ops/sampling.h"

#include <cstdint>

namespace ninfer::ops {

struct SpeculativeAcceptExecutionEnvelope {
    // Every row has temperature<=0, both penalties disabled and no token mask.
    // False retains support for any registered mixture of greedy and stochastic rows.
    bool all_rows_greedy_without_penalties = false;
};

// Caller-owned transient capacity over the inclusive draft-count and batch-size intervals.
[[nodiscard]] std::size_t speculative_accept_greedy_drafts_workspace_capacity_bytes(
    std::int32_t token_domain, std::int32_t min_drafts, std::int32_t max_drafts,
    std::int32_t min_batch, std::int32_t max_batch);

/**
 * Accept a prefix of one-hot greedy proposals against the target distribution.
 *
 * For row b, consume target columns 0..current_extents[b]. Greedy rows accept matching
 * penalty-adjusted target argmax IDs, then emit the first correction or bonus ID. Positive-
 * temperature rows accept draft i with probability p_i(draft_i); the first rejection samples
 * normalized p_i with that draft removed. Full acceptance samples the bonus column.
 * Filtering, ties and normalization follow sampling.h. The penalty overlay at column i is
 * drafts[0:i,b]. RNG uses the request seed, base_positions[b]+i+1 and the speculative
 * accept/correction/bonus purpose. There is no mutable RNG or committed-state update.
 *
 * target_tokens and licensed_tokens are contiguous I32 [K+1,B], logits is contiguous BF16
 * [physical_rows,K+1,B]. drafts is I32 [K,B] with positive, naturally aligned, non-overlapping
 * strides. current_extents/base_positions/licensed_counts are contiguous I32 [B]; configs is
 * Device SamplingConfig[B]. K,B>=1, token_domain in [1,physical_rows], extents in [0,K], and
 * base_positions+i+1 fits nonnegative I32. Raw target_tokens are exact argmax over token_domain,
 * with lower IDs winning ties; only raw greedy rows without penalties/masks consume them.
 *
 * If A drafts are accepted, output their prefix and one correction/bonus token, zero the
 * remaining physical suffix, and write licensed_counts[b]=A+1. Every consumed target column
 * has nonempty support. Outputs are mutually disjoint and overlap no input/config/count/mask
 * storage. All inputs, counts, positions and proposals remain unchanged. The caller commits
 * its final prefix and updates counts separately.
 *
 * Uses caller-owned workspace reported by the capacity query; no hidden Device allocation.
 */
void speculative_accept_greedy_drafts(const Tensor& target_tokens, const Tensor& logits,
                                      const Tensor& drafts, const Tensor& current_extents,
                                      const Tensor& base_positions, Tensor& licensed_tokens,
                                      Tensor& licensed_counts, std::int32_t token_domain,
                                      const SamplingConfig* configs, MemoryArena& workspace,
                                      CudaExecutionView execution);

// The raw-greedy envelope requires no workspace.
[[nodiscard]] std::size_t speculative_accept_sparse_drafts_workspace_capacity_bytes(
    std::int32_t token_domain, SpeculativeAcceptExecutionEnvelope envelope, std::int32_t min_drafts,
    std::int32_t max_drafts, std::int32_t min_batch, std::int32_t max_batch);

/**
 * Accept a prefix of sparse proposals with their actual FP32 q distributions.
 *
 * For P=current_extents[b], only columns 0..P are live. Greedy rows accept the longest prefix
 * matching the adjusted target argmax, then emit a correction/bonus. Positive-temperature rows
 * use min(1,p(d)/q(d)) acceptance; first rejection samples normalized max(p-q,0), and full
 * acceptance samples the target bonus column. Target filtering, prefix penalties and RNG
 * positions/purposes follow speculative_accept_greedy_drafts.
 *
 * Registered profile: token_domain=248077, K=1..15, B=1..8, BF16 logits [248320,K+1,B],
 * I32 target_tokens/licensed_tokens [K+1,B], I32 drafts [K,B], I32 candidate_ids [16,K,B],
 * FP32 proposal_q [16,K,B], I32 current_extents/base_positions/licensed_counts [B]. Everything
 * is contiguous except drafts, which permits positive, naturally aligned, non-overlapping
 * strides. Extents lie in [0,K]. Each live candidate list has distinct global IDs in the domain;
 * q is normalized and is the represented distribution used to generate that draft, with positive
 * probability at the chosen ID. q is consumed directly, without reconstructing a dense proposal.
 *
 * Writes the accepted prefix, one correction/bonus token and zero physical suffix; counts=A+1.
 * Counts/configs, positions, proposals and all other inputs remain read-only. Inputs and outputs
 * obey the disjointness and logical-position requirements of the greedy-proposal entry.
 * The caller adopts a final output prefix and owns every persistent state/count update.
 *
 * A true all_rows_greedy_without_penalties envelope enables the raw-target warp route; false
 * supports mixed rows. Uses caller-owned capacity-query workspace with no hidden allocation.
 */
void speculative_accept_sparse_drafts(const Tensor& target_tokens, const Tensor& logits,
                                      const Tensor& drafts, const Tensor& candidate_ids,
                                      const Tensor& proposal_q, const Tensor& current_extents,
                                      const Tensor& base_positions, Tensor& licensed_tokens,
                                      Tensor& licensed_counts, std::int32_t token_domain,
                                      const SamplingConfig* configs,
                                      SpeculativeAcceptExecutionEnvelope envelope,
                                      MemoryArena& workspace, CudaExecutionView execution);

} // namespace ninfer::ops
