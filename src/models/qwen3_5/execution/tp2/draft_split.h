#pragma once

// Two-device (tp 2) forms of the DFlash2 drafter (execution/draft.cpp). Our file.

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/program/context.h"
#include "ninfer/ops/allreduce.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace ninfer::models::qwen3_5::execution {

// Rank 1's half of the split DFlash2 drafter (load/sharding.h): its ExecutionCore (rank 1's
// device, Parameters with the drafter shards, workspace, round state and prefill hidden) and its
// own drafter state (DFlash rings of its four KV heads in rank 1's StateImages, prefill and pending
// target features captured from rank 1's copy of the replicated residual). Owned by the Program's
// rank 1 runtime; TpExecution::dflash names it. Rank 1's DFlash2 decode frame is the round's
// `peer_frame`.
struct DFlashPeerDrafter {
    ExecutionCore execution;
    DFlashPersistentState& dflash;
};

// Whether a rank's Parameters hold the split drafter: its layers' context key blocks carry half of
// the drafter's KV heads.
[[nodiscard]] bool dflash2_drafter_split(const Parameters& parameters);

// The split drafter's round prelude on both ranks: each rank gathers its pending target features,
// appends its context (replicated feature projection, its own four KV heads) and runs its half of
// the drafter layers; the attention output and MLP down projections are row-parallel, summed by
// one allreduce_sum each, after which both ranks apply the dynamic convolution's finish to the
// identical sum. Both ranks then rank the candidates of their proposal head blocks
// (dflash2_candidates_split, no broadcast), and the rank that holds the candidate selector
// (load/sharding.cpp: rank 1 by default) selects the drafts and their distributions into its
// frame; dflash2_share_drafts_tp2 then gives them to the other rank.
void dflash2_draft_round_tp2(DFlashBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                             DFlashEnvelopes envelopes);

// Whether a split drafter layer's MLP gate/up runs the fused SwiGLU projection: an NVFP4 gate/up
// (A16Only, tools/convert/official_recipes.py _dflash2_nvfp4_gate_up) has a fused A16 route at
// every width, while Q8 keeps the linear projection followed by silu_mul (the drafts of artifacts
// with a Q8 drafter stay what they were).
[[nodiscard]] inline bool dflash2_split_mlp_fused(const LinearParameters& gate_up) {
    return gate_up.weight.qtype == QType::NVFP4;
}

// The rank that holds the split drafter's candidate selector (load/sharding.cpp), read from rank
// 0's Parameters: 0 when they hold it, else 1.
[[nodiscard]] std::size_t dflash2_selector_rank(const Parameters& rank0);

// After a DFlash2 round's proposal at tp 2: the selecting rank's draft tokens reach the other rank
// (rank 0 -> 1 when rank 0 selects, which covers the drafter on rank 0 alone; rank 1 -> 0 with the
// selector on rank 1, together with the proposal distributions rank 0's acceptance reads), and rank
// 1 derives its verification inputs from its drafts. Rank 0 derives its own afterwards.
void dflash2_share_drafts_tp2(DFlashBatchContext& state, std::int32_t batch_size);

// The split drafter's eager context catch-up on rank 1 (ProgramImpl::enqueue_dflash_context_append
// does rank 0's): uploads `host_ingress` to rank 1's decode frame, gathers rank 1's pending
// features and appends rank 1's context.
void dflash_append_context_peer(DFlashPeerDrafter& peer, const qwen3_5::DFlashDecodeIngress& host,
                                std::int32_t batch, std::uint32_t draft_window,
                                ops::KVCacheAppendPrefixExecutionEnvelope envelope);

// Rank 1's DFlash feature sinks of the split drafter: target verification scatters into its pending
// features, and a prefill chunk into its prefill features, whose consumer appends rank 1's context
// with the chunk's bindings (`host_ingress`, already filled by rank 0's consumer).
[[nodiscard]] DFlashFeatureSink dflash_peer_batch_sink(DFlashPeerDrafter& peer, const Tensor& lanes,
                                                       const Tensor& valid_columns,
                                                       std::int32_t width, std::int32_t batch);
[[nodiscard]] DFlashFeatureSink
dflash_peer_prefill_sink(DFlashPeerDrafter& peer, qwen3_5::DFlashPrefillIngress* host_ingress);

// Vocabulary-split DFlash2 candidate ranking. `head[r]` is rank r's row block of the optimized
// proposal head with its block of the row -> token map (load/sharding.h: DFlash2 splits both by
// rows). Rank r ranks `hidden[r]` [H,M] over its block (ops::linear_topk_split) and packs its top
// sixteen (ops::topk_split_pack); one allreduce_sum of the packed candidates gives both ranks both
// lists, and rank 0 writes the complete top sixteen, in linear_topk's order and encoding, to `ids`
// I32 [16,M] and `scores` FP32 [16,M]. Each block row equals the whole head's row bit for bit and
// the keys carry global token ids, so the result equals ops::linear_topk over the whole head.
//
// With `broadcast` only rank 0 holds the hidden (the drafter runs on rank 0 alone): rank 1's
// `hidden[1]` is overwritten with rank 0's through one more allreduce_sum against zeros, whose
// FP32 combine of x and +0 returns x (a -0 becomes +0, which no product or sum can tell apart);
// rank 0's `hidden[0]` is rewritten in place with the same values. Without it both ranks already
// hold the same hidden. Rank r's scratch comes from `workspace[r]`, which must hold
// dflash2_candidates_split_workspace_bytes(M, broadcast). With `peer_ids` and `peer_scores` (rank
// 1's tensors of the same shapes), rank 1 merges the same union into them as well, for a candidate
// selector on rank 1.
void dflash2_candidates_split(const std::array<Tensor, 2>& hidden,
                              const std::array<const ProposalParameters*, 2>& head, bool broadcast,
                              Tensor& ids, Tensor& scores,
                              const std::array<WorkspaceArena*, 2>& workspace,
                              const ExecutionContext& execution, const ops::PeerEvents& events,
                              Tensor* peer_ids = nullptr, Tensor* peer_scores = nullptr);

// One rank's scratch for dflash2_candidates_split over `columns` hidden columns of `hidden` rows.
[[nodiscard]] std::size_t dflash2_candidates_split_workspace_bytes(const ProposalParameters& head,
                                                                   std::int32_t hidden,
                                                                   std::int32_t columns,
                                                                   bool broadcast);

} // namespace ninfer::models::qwen3_5::execution
