#pragma once

// Two-device tensor-parallel (tp == 2) execution operands of the Text backbone.
//
// Rank r of a tp2 Model holds the head-, intermediate- and vocabulary-split shards described in
// load/sharding.h and executes on ExecutionContext::dev[r]. The hidden/residual axis is
// replicated: every layer closes with a row-parallel projection whose all-reduce leaves the same
// BF16 residual on both ranks, so norms, the token embedding and every per-rank state update see
// identical inputs. Rank 0 alone owns request bookkeeping, sampling and the round egress; rank 1
// owns only its shards of the weights, KV planes, GDN state and scratch.

#include "core/arena.h"
#include "core/device.h"
#include "core/gdn_replay_records.h"
#include "core/linear_attention_state.h"
#include "core/tensor.h"
#include "models/qwen3_5/config.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/execution/tp2/prefill_slices.h"
#include "models/qwen3_5/program/round_buffers.h"
#include "models/qwen3_5/state/decoder_state.h"
#include "ninfer/ops/allreduce.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace ninfer::models::qwen3_5::execution {

inline constexpr int kTensorParallelWidth = 2;

struct DFlashPeerDrafter; // execution/tp2/draft_split.h
struct Qwen4Runtime;      // execution/qwen4_runtime.h

// The logical Text config narrowed to one rank's share: attention query/KV heads, GDN key/value
// heads, the Dense intermediate width and the vocabulary are divided by `width`; the hidden size,
// head dimensions, layer schedule and RoPE are unchanged. It names the per-rank extents of the
// shared workspace recipes, KV planes and GDN state pools. Qwen4Exp keeps its MoE geometry (routed
// experts partitioned by ID, shared expert whole on one rank per block), its QSA indexer and PLE,
// and with `hc_split_streams` halves the hyper-connection streams (the rank's residual is
// hc_count/width streams wide). Throws std::invalid_argument for an extent `width` does not
// divide.
[[nodiscard]] TextConfig shard_text_config(const TextConfig& config, int width,
                                           bool hc_split_streams = false);

// Rank 1's copy of the ordinary decode control. The Program publishes it on rank 1's stream before
// the call, by uploading the same OrdinaryDecodeIngress record rank 0 receives into a frame of
// the same layout on rank 1; only the first B entries of each vector are read. The fields name
// the same tokens, positions and state slots as rank 0's, and the same KV execution rows: rank
// 0's KVExecutionTablePool replays every acquire and publication onto rank 1's mirror at the same
// row index. Rank 1 never samples, so the ingress sampling configs are never read there.
struct OrdinaryPeerFrame {
    Tensor tokens;                  // I32 [capacity]
    Tensor cache_positions;         // I32 [capacity]
    Tensor rope_positions;          // I32 [capacity]
    Tensor text_kv_table_rows;      // I32 [capacity]
    Tensor state_source_slots;      // I32 [capacity]
    Tensor state_destination_slots; // I32 [capacity]
};

// Views a rank-1 OrdinaryDecodeState's ingress fields; the frame must be resident on rank 1.
[[nodiscard]] OrdinaryPeerFrame ordinary_peer_frame(const qwen3_5::OrdinaryDecodeState& frame);

// Vocabulary-split output head. Rank r projects `hidden[r]` [H,C] through its head shard
// [V_r,H] into `partial[r]` [V_r,C]; V = V_0 + V_1, rank 0 first.
[[nodiscard]] std::size_t output_head_split_workspace_bytes(const LinearParameters& shard,
                                                            std::int32_t first, std::int32_t last);

// Rank 0 alone receives the complete [V,C] `logits`: it pulls rank 1's contiguous `partial[1]`
// block into its own `staging` [V_1,C] with one cross-device copy, then one kernel (`concat_rows`)
// interleaves both halves column by column; a kernel rather than two pitched copies, because a
// captured 2D memcpy node cannot take a changed column count or buffer in place between the CUDA
// Graph profiles of one class. Rank 1 keeps no copy of the logits. `logits` and
// `staging` are contiguous BF16 on rank 0. On return rank 0's stream is ordered after rank 1's
// projection and rank 1's stream after rank 0's pull, so both may reuse their operands.
void output_logits_split_rank0(const std::array<Tensor, 2>& hidden,
                               const std::array<const LinearParameters*, 2>& head,
                               const std::array<Tensor, 2>& partial, const Tensor& logits,
                               const Tensor& staging,
                               const std::array<WorkspaceArena*, 2>& workspace,
                               const ExecutionContext& execution, const ops::PeerEvents& events);

// Vocabulary-split argmax of the optimized MTP proposal head (load/sharding.h). Rank r projects
// `hidden[r]` [H,C] through its row block `head[r]` [R,H] (rank 0 rows [0,R), rank 1 [R,2R)) into
// `partial[r]` [R,C], takes the argmax of each column into `local[r]` I32 [C] and packs it with its
// value into `candidates[r]` [8,C] (ops::argmax_split_pack). One allreduce_sum of the candidates,
// through `staging[r]` [8,C] and the pair's captured transport (the mailbox, or the copies under a
// StagedScope), gives both ranks both candidates, and rank 0 writes the complete argmax row in
// [0,2R) to `tokens` I32 [C] (ops::argmax_split_select): the row argmax() over the complete
// [2R,C] logits selects, lower row on ties, since each half row equals the whole head's row. Rank
// 1 keeps no result. All operands are contiguous and on their rank; `tokens` on rank 0.
void proposal_argmax_split(const std::array<Tensor, 2>& hidden,
                           const std::array<const LinearParameters*, 2>& head,
                           const std::array<Tensor, 2>& partial, const std::array<Tensor, 2>& local,
                           const std::array<Tensor, 2>& candidates,
                           const std::array<Tensor, 2>& staging, const Tensor& tokens,
                           const std::array<WorkspaceArena*, 2>& workspace,
                           const ExecutionContext& execution, const ops::PeerEvents& events);

struct GdnGatingOverlap; // tp2/gdn_gating_overlap.h

// Everything a TextContext needs to drive rank 1 in lockstep with its own rank-0 operands. The
// TextContext's DeviceContext must be `execution->dev[0]`. All members are borrowed and must
// outlive the context; the pointers into Program storage are stable for the Program's lifetime.
// The Program owns one instance; a prefill call copies it to name the prefilling sequence's rank-1
// MTP KV row (`mtp_kv`), the only per-sequence member.
struct TpExecution {
    const ExecutionContext* execution = nullptr; // tp == 2
    const ops::PeerEvents* events     = nullptr; // one instance per stream pair, Program-owned
    const Parameters* parameters      = nullptr; // Parameters(model, 1)
    WorkspaceArena* work              = nullptr; // rank 1's transient arena
    // Rank 1's GDN state pool (value heads / 2, conv channels / 2), slot-for-slot with rank 0's.
    LinearAttentionStatePool* linear_attention = nullptr;
    // Rank 1's text KV cache (KV heads / 2); its page pool and execution tables are rank 0's
    // mirrors, so a rank-0 execution row names the same pages on rank 1.
    const qwen3_5::PagedKVCache* text_cache = nullptr;
    // I32 [1] on rank 1: the prefill execution row, equal to rank 0's RoundState
    // text_kv_table_row. Read by prefill only.
    Tensor text_kv_table_row;
    // I32 [1] on rank 1: its copy of rank 0's RoundState rope_delta, the prefilling sequence's
    // RoPE delta. The Program publishes both at sequence start and a prefill chunk rewrites both;
    // the prompt MTP proposal steps offset rank 1's positions by it.
    Tensor rope_delta;
    // Rank 1's ordinary decode control. Read by ordinary decode only; may be null otherwise.
    const OrdinaryPeerFrame* ordinary = nullptr;
    // Rank 1's ReplaySSM records of its GDN heads, written by speculative target verification
    // (MTP and DFlash2); null without a speculative backend.
    const GdnReplayRecords* replay_records = nullptr;
    // Rank 1's runtime inputs of the Text forward (with routed experts its backend over the odd
    // expert IDs, its PLE convolution states and decode PLE rows).
    const Qwen4Runtime* qwen4 = nullptr;
    // Side streams of the tp 2 GDN gating overlap (tp2/gdn_gating_overlap.h), Program-owned.
    GdnGatingOverlap* gdn_gating = nullptr;

    // --- DFlash2 with the split drafter (load/sharding.h; null otherwise)
    // ------------------------- Rank 1's drafter operands and state; every DFlash2 call that
    // appends context, proposes or captures target features drives it beside rank 0's.
    DFlashPeerDrafter* dflash = nullptr;

    // --- MTP (speculative backend Mtp only; null or empty otherwise) ----------------------------
    // Rank 1's MTP KV cache (KV heads / 2); like the text cache, its page pool and execution
    // tables are rank 0's mirrors.
    const qwen3_5::PagedKVCache* mtp_cache = nullptr;
    // Rank 1's view of the prefilling sequence's MTP KV row: rank 0's row lease mirrored onto
    // rank 1's tables. Set per prefill call; decode reads its rows from the round frame instead.
    qwen3_5::PagedKVCacheView mtp_kv;
    // I32 [1] on rank 1: the prefilling sequence's MTP execution row, equal to rank 0's RoundState
    // backend_kv_table_row. Read by the prompt MTP proposal steps only.
    Tensor backend_kv_table_row;
    // Rank 1's MTP prefill frame: its copies of the proposal hidden and position.
    const qwen3_5::MtpPrefillState* mtp = nullptr;
    // BF16 [hidden, prefill_chunk] on rank 1: the final-normed prefill chunk, which rank 1's
    // half of the MTP input projection contracts.
    Tensor prefill_hidden;
    // The captured MTP or masked-draft (DFlash2) draft phase keeps its all-reduces on the staged
    // copies even when the pinned-host mailbox carries the target forward's
    // (ops::PeerEvents::StagedScope). Set by the Program when the mailbox hung in a full
    // speculative round (WSL2) or by NINFER_TP_MAILBOX_DRAFT=copies.
    bool staged_draft_collectives = false;

    [[nodiscard]] bool complete() const noexcept {
        return execution != nullptr && events != nullptr && parameters != nullptr &&
               work != nullptr && linear_attention != nullptr && text_cache != nullptr &&
               rope_delta.data != nullptr;
    }

    // The MTP members every tensor-parallel MTP call reads (mtp_kv is per call).
    [[nodiscard]] bool mtp_complete() const noexcept {
        return mtp_cache != nullptr && mtp != nullptr && prefill_hidden.data != nullptr &&
               backend_kv_table_row.data != nullptr && replay_records != nullptr;
    }
};

} // namespace ninfer::models::qwen3_5::execution
