#pragma once

#include "ninfer/ops/sampling.h"

namespace ninfer::ops {

// Candidate words are stored as I64 but ordered and communicated as unsigned 64-bit integers.
// For finite adjusted score x (signed zero canonicalized), let b be its FP32 bits and
// u = signbit(b) ? ~b : (b ^ 0x80000000). The key is (U64(u)<<32) | (UINT_MAX-global_id).
// Larger keys win; equal scores prefer the smaller ID. Zero denotes no candidate.
// MAX combines greedy keys; concatenating sorted Top-20 lists preserves global Top-20.

[[nodiscard]] std::size_t vocabulary_candidates_workspace_capacity_bytes(std::int32_t physical_rows,
                                                                         std::int64_t max_columns,
                                                                         std::int32_t candidates);

/**
 * Select local vocabulary candidates from contiguous BF16 logits [V,T,B]. Local row v has
 * global ID token_begin+v; only IDs < token_domain participate. token_begin>=0, token_domain>0,
 * V,T>0, B=1..8. Output keys is contiguous I64 [K,T,B], K=1 or 20.
 * K=1 promises greedy configs. K=20 supports mixed rows; a greedy row emits only its best key.
 * Scores use sampling.h penalties, mask and tie order, before temperature or probability filters.
 * top_k limits each row's candidate count. Consumed logits and adjusted scores are finite.
 *
 * configs is Device SamplingConfig[B]; its counts and masks retain GLOBAL vocabulary indexing.
 * They are read-only. valid_columns is Device I32 [B], 0<=n[b]<=T. An optional I32 drafts
 * [T-1,B] supplies the penalty overlay drafts[0:t,b] for column t; an empty Tensor disables it.
 * Masks use position t regardless of whether drafts are present. Only t<n[b] is consumed.
 * Missing candidates, locally empty masks, padding IDs and inactive columns produce zero keys.
 *
 * Tensors are contiguous except drafts, which permits positive, naturally aligned,
 * non-overlapping strides. All belong to execution.device(). keys overlaps no
 * input, config or its referenced storage. Workspace is caller-owned, disjoint from operands,
 * and sized for at least T*B columns. No persistent state is changed. All metadata is read on
 * Device at execution time; descriptors/scalars are capture-time values.
 */
void vocabulary_candidates(const Tensor& logits, std::int32_t token_begin,
                           std::int32_t token_domain, const SamplingConfig* configs,
                           const Tensor& valid_columns, const Tensor& drafts, Tensor& keys,
                           MemoryArena& workspace, CudaExecutionView execution);

/**
 * Sample from gathered disjoint vocabulary candidates, keys I64 [K,1,B,R], K=1 or 20,
 * R=1..16, B=1..8. Each rank list is descending and zero-padded. K=1 promises greedy configs;
 * for stochastic rows each input list contains every locally required Top-K candidate.
 * Apply global Top-K, then sampling.h min-p/top-p, normalization and RNG, using Device
 * configs[B], logical_positions I32 [B] and purpose. Output I32 [B] is the global ID, or -1
 * when all ranks have empty support. Counts, masks and inputs are preserved; config mask/count
 * pointers are not dereferenced. Output overlaps no input or configs. All storage is contiguous and
 * naturally aligned on execution.device(). No workspace or hidden allocation.
 */
void sample_vocabulary_candidates(const Tensor& keys, const SamplingConfig* configs,
                                  const Tensor& logical_positions, std::int32_t purpose,
                                  Tensor& output, CudaExecutionView execution);

/**
 * Greedy-draft speculative acceptance from gathered keys I64 [K,D+1,B,R]. D=1..15; K/B/R,
 * key representation and storage follow sample_vocabulary_candidates. drafts is I32 [D,B],
 * current_extents is I32 [B] in [0,D]. Only columns 0..current_extents[b] are consumed.
 * Candidates already include each column's preceding draft penalty overlay.
 *
 * Greedy rows accept matching target IDs. Stochastic rows use the one-hot proposal law:
 * accept draft j with probability p_j(draft_j), sample normalized p_j with draft_j removed
 * on first rejection, otherwise sample the bonus column. Filters/RNG purposes and positions
 * are those of speculative_accept_greedy_drafts. Inputs and token counts remain read-only.
 * Each consumed column must have nonempty global support.
 *
 * drafts permits positive, naturally aligned, non-overlapping strides. Base logical positions
 * are read-only I32 [B], with the same RNG meaning as speculative_accept_greedy_drafts.
 * Writes I32 licensed_tokens [D+1,B] (accepted prefix, correction/bonus, then zeros) and
 * licensed_counts [B]. Outputs are disjoint and overlap no inputs/configs. No workspace or
 * persistent state effects. The caller later commits its chosen prefix and updates counts.
 */
void accept_vocabulary_candidates(const Tensor& keys, const SamplingConfig* configs,
                                  const Tensor& drafts, const Tensor& current_extents,
                                  const Tensor& base_positions, Tensor& licensed_tokens,
                                  Tensor& licensed_counts, CudaExecutionView execution);

} // namespace ninfer::ops
