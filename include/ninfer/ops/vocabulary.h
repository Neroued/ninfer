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
 * V,T,B>0. Output keys is contiguous I64 [K,T,B], K=1 or 20.
 * K=1 promises greedy configs. K=20 supports mixed rows; a greedy row emits only its best key.
 * Scores use sampling.h penalties, mask and tie order, before temperature or probability filters.
 * top_k limits each row's candidate count. Consumed logits and adjusted scores are finite.
 *
 * configs is Device SamplingConfig[B]; its counts and masks retain GLOBAL vocabulary indexing.
 * They are read-only. valid_columns is Device I32 [B], 0<=n[b]<=T. An optional I32 drafts
 * [D,B], D>=1, supplies the penalty overlay drafts[0:t,b] for column t. With drafts the caller
 * provides n[b]<=min(T,D+1). An empty Tensor disables the overlay.
 * Masks use position t regardless of whether drafts are present. Only t<n[b] is consumed.
 * Missing candidates, locally empty masks, padding IDs and inactive columns produce zero keys.
 * Inactive columns do not read logits, configs, draft values or referenced mask/count storage.
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

[[nodiscard]] std::size_t vocabulary_candidates_with_lse_workspace_capacity_bytes(
    std::int32_t physical_rows, std::int64_t max_columns, std::int32_t candidates);

// Same candidate keys in contiguous I64 records [K+1,T,B]. Word K holds the FP32 bits
// of log(sum exp((adjusted_score - shard_max) / tau)) in its low 32 bits, with the high bits
// zero. The sum includes all allowed local scores. shard_max is the score in word 0, and
// tau is the positive configured temperature or one. Inactive and empty columns write zero
// to all words. Scratch follows the with_lse capacity query (candidates = K).
void vocabulary_candidates_with_lse(const Tensor& logits, std::int32_t token_begin,
                                    std::int32_t token_domain, const SamplingConfig* configs,
                                    const Tensor& valid_columns, const Tensor& drafts,
                                    Tensor& records, MemoryArena& workspace,
                                    CudaExecutionView execution);

/**
 * Sample from gathered disjoint vocabulary candidates, keys I64 [K,1,B,R], K=1 or 20,
 * R=1..16, B>=1. Each rank list is descending and zero-padded. K=1 promises greedy configs;
 * for stochastic rows each input list contains every locally required Top-K candidate.
 * Apply global Top-K, then sampling.h min-p/top-p, normalization and RNG, using Device
 * configs[B], logical_positions I32 [B] and purpose. Output I32 [B] is the global ID, or -1
 * when all ranks have empty support. Counts, masks and inputs are preserved; config mask/count
 * pointers are not dereferenced. Output overlaps no input or configs. All storage is contiguous and
 * naturally aligned on execution.device(). live_rows is empty for dense rows or contiguous
 * Device I32[B] with values 0/1. An inactive row writes output zero and reads no keys, config or
 * logical position. No workspace or hidden allocation.
 */
void sample_vocabulary_candidates(const Tensor& keys, const SamplingConfig* configs,
                                  const Tensor& logical_positions, std::int32_t purpose,
                                  Tensor& output, const Tensor& live_rows,
                                  CudaExecutionView execution);

// Same draw from gathered I64 records [K+1,1,B,R]. The disjoint contiguous Device FP32 [B]
// output is the full-domain top-1 probability, or zero for inactive rows or empty global
// support. Each nonempty rank contributes exp((shard_max - global_max) / tau + local_rel_lse)
// to the denominator; ranks with word 0 equal to zero contribute nothing.
void sample_vocabulary_candidates(const Tensor& records, const SamplingConfig* configs,
                                  const Tensor& logical_positions, std::int32_t purpose,
                                  Tensor& output, Tensor& top1_probabilities,
                                  const Tensor& live_rows, CudaExecutionView execution);

/**
 * Greedy-draft speculative acceptance from gathered keys I64 [K,W,B,R]. W=2..16; K/B/R,
 * key representation and storage follow sample_vocabulary_candidates. drafts is I32 [D,B], D>=1.
 * Live current_extents are I32 [B] in [0,min(W-1,D)]. Only columns 0..current_extents[b] are
 * consumed. Candidates already include each column's preceding draft penalty overlay.
 *
 * Greedy rows accept matching target IDs. Stochastic rows use the one-hot proposal law:
 * accept draft j with probability p_j(draft_j), sample normalized p_j with draft_j removed
 * on first rejection, otherwise sample the bonus column. Filters/RNG purposes and positions
 * are those of speculative_accept_greedy_drafts. Inputs and token counts remain read-only.
 * Each consumed column must have nonempty global support.
 *
 * drafts permits positive, naturally aligned, non-overlapping strides. Base logical positions
 * are read-only I32 [B], with the same RNG meaning as speculative_accept_greedy_drafts.
 * Writes I32 licensed_tokens [W,B] (accepted prefix, correction/bonus, then zeros) and
 * licensed_counts [B]. Outputs are disjoint and overlap no inputs/configs. No workspace or
 * persistent state effects. The caller later commits its chosen prefix and updates counts.
 * live_rows follows sample_vocabulary_candidates. Inactive rows write zero tokens/counts and
 * read no row inputs; a live row with zero draft extent still produces its target/bonus token.
 */
void accept_vocabulary_candidates(const Tensor& keys, const SamplingConfig* configs,
                                  const Tensor& drafts, const Tensor& current_extents,
                                  const Tensor& base_positions, Tensor& licensed_tokens,
                                  Tensor& licensed_counts, const Tensor& live_rows,
                                  CudaExecutionView execution);

/**
 * Sparse-proposal acceptance from gathered target keys I64 [K,W,B,R], following the
 * key/filter/storage contract above. candidate_ids I32 [16,D,B] and proposal_q FP32 [16,D,B]
 * describe the conditional proposal distribution actually used for each draft prefix.
 * D=1..15. The 16 candidates are contiguous; each tensor has independent positive, aligned,
 * non-overlapping column and row strides. q is finite, nonnegative and sums to one (up to FP32
 * rounding). Positive-q IDs are distinct valid global IDs. Zero-q IDs are ignored (INT_MAX is
 * allowed). For every consumed positive-temperature draft, q(draft)>0. Greedy rows ignore IDs/q.
 *
 * Target keys already include penalties and masks. Apply the existing target Top-K/min-p/top-p
 * normalization once, accept with min(1,p(draft)/q(draft)), and on first rejection draw from
 * normalize(max(p-q,0)), aligned by global ID. A proposal ID outside target support adds no
 * correction mass. If FP32 rounding leaves zero residual mass, correction falls back to p.
 * Full acceptance draws the bonus from p. Greedy rows accept matching target
 * IDs. RNG purposes and base_position+column+1 match accept_vocabulary_candidates.
 *
 * D/B/R, nonempty consumed target support, strided drafts, extents, positions, output writes,
 * disjointness and read-only state follow that entry. Only columns <=current_extents[b]
 * participate, and IDs/q are read only for positive-temperature columns <current_extents[b].
 * Outputs also overlap neither IDs nor q. No workspace, allocation or persistent state effects.
 */
void accept_vocabulary_candidates_sparse(const Tensor& keys, const SamplingConfig* configs,
                                         const Tensor& drafts, const Tensor& candidate_ids,
                                         const Tensor& proposal_q, const Tensor& current_extents,
                                         const Tensor& base_positions, Tensor& licensed_tokens,
                                         Tensor& licensed_counts, const Tensor& live_rows,
                                         CudaExecutionView execution);

} // namespace ninfer::ops
