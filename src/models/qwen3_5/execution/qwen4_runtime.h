#pragma once

// The Program-owned inputs of the Text forward (Qwen4Runtime) and its observation interfaces: the
// teacher-forcing taps, the routed-expert probe and a drafter's feature tap. Small on purpose: the
// Program and the validation dumps bind these without the forward (qwen4_forward.h).

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <span>

namespace ninfer::models::qwen3_5::execution {

class Qwen4RoutedExperts; // qwen4_experts.h
struct Qwen4ExpertBank;   // parameters.h
class PleGather;          // ple_gather.h
struct Qwen4DraftHead;    // qwen4_draft_head.h

// Host taps of the wide residual after selected layers (teacher-forcing dumps, eager only).
class Qwen4Taps {
public:
    virtual ~Qwen4Taps()                                       = default;
    [[nodiscard]] virtual bool wants(std::int32_t layer) const = 0;
    // `wide` BF16 [S*H, T] committed residual after `layer` (or the head mixer output when
    // layer == -1, BF16 [H,T]); column t is sequence position first_position + t.
    virtual void capture(std::int32_t layer, const Tensor& wide, std::int32_t first_position,
                         cudaStream_t stream) = 0;

    // tp 2: the same tap's two rank copies (each on its rank and stream), which the replicated
    // schedule keeps bit-identical (identical routing on both ranks depends on it).
    virtual void compare_ranks(std::int32_t layer, const Tensor& rank0, const Tensor& rank1,
                               cudaStream_t stream0, cudaStream_t stream1) {
        (void)layer, (void)rank0, (void)rank1, (void)stream0, (void)stream1;
    }

    // T128: the two rank partials of a TP2 prefill exchange, before the sum (eager only).
    // site 0 = mixer output projection (BF16 [H,T]), 1 = FFN: MoE (FP32 [H,T], NINFER_TP_AR=0
    // path) or the dense FFN's row-parallel down projection (as the mixer's).
    [[nodiscard]] virtual bool wants_exchange() const { return false; }

    virtual void capture_exchange(std::int32_t layer, int site, const Tensor& rank0,
                                  const Tensor& rank1, std::int32_t first_position,
                                  cudaStream_t stream0, cudaStream_t stream1) {
        (void)layer, (void)site, (void)rank0, (void)rank1, (void)first_position, (void)stream0,
            (void)stream1;
    }

    // MTP teacher forcing (T24): every prompt column runs the whole MTP layer and its draft
    // logits (BF16 [V,1] of the column at `position`, predicting the token at position + 2) are
    // captured, with the prompt tokens.
    [[nodiscard]] virtual bool wants_mtp_logits() const { return false; }

    virtual void capture_mtp_logits(const Tensor&, std::int32_t, cudaStream_t) {}

    virtual void capture_tokens(std::span<const std::int32_t>) {}
};

// Host probe of a routed-expert call's inputs (eager validation only; T24 calibrates the MTP
// layer's activation ranges with it): x BF16 [H,T], ids I32 [10,T], weights FP32 [10,T].
class Qwen4MoeProbe {
public:
    virtual ~Qwen4MoeProbe()                                         = default;
    [[nodiscard]] virtual bool wants(std::int32_t layer) const       = 0;
    virtual void capture(std::int32_t layer, const Tensor& x, const Tensor& ids,
                         const Tensor& weights, cudaStream_t stream) = 0;
};

// Layer-output taps of a speculative drafter over the plain residual (the DFlash2 target
// features: TextContext adapts its DFlashFeatureSink): every rank's residual after the embedding
// and after each layer's closing all-reduce has completed, enqueued on the ranks' streams.
// Execution only; the hyper-connection residual publishes none.
class Qwen4FeatureTap {
public:
    virtual ~Qwen4FeatureTap()                                            = default;
    virtual void begin(const std::array<Tensor, 2>& x)                    = 0;
    virtual void capture_layer(int layer, const std::array<Tensor, 2>& x) = 0;
    // Whether the tap takes a layer's output by column windows (a prompt chunk's features; not a
    // batched-feature tap): the tp 2 half-chunk overlap needs it.
    [[nodiscard]] virtual bool captures_columns() const = 0;
    // A layer's output for the columns [first, first + x[r].ne[1]) of the call only: the same
    // bytes, at the same place, as capture_layer publishes for those columns.
    virtual void capture_layer_columns(int layer, const std::array<Tensor, 2>& x,
                                       std::int32_t first) = 0;
};

// Program-owned inputs a TextContext reads (bound per execution core). At tp 2 each rank
// has its own: rank 0's is the ExecutionCore's, rank 1's TpExecution::qwen4. The PLE gather, its
// staging and the decode host rows are rank 0's (one host gather, uploaded to both ranks); the
// taps are rank 0's alone (the replicated residual).
struct Qwen4Runtime {
    // This rank's routed-expert backend: at tp 2 it computes the experts of the rank's ID parity.
    Qwen4RoutedExperts* experts = nullptr;
    // MTP layer (T24): its routed-expert backend and its bank with per-expert divisor planes (the
    // Program expands a bank-wide divisor; Qwen4MtpParameters). The backend may be `experts`
    // itself (the offload backend serves the MTP layer as layer num_hidden_layers). Null without
    // MTP.
    Qwen4RoutedExperts* mtp_experts        = nullptr;
    const Qwen4ExpertBank* mtp_expert_bank = nullptr;
    // PLE convolution states of the PLE layer, BF16 [rows, slots] (StateImage).
    Tensor ple_states;
    // Speculative verification: the PLE convolution inputs of the round, BF16 [rows, K+1,
    // max_concurrency]; the commit advances the history by each row's accepted prefix (T10's
    // record/fold contract). Empty without a speculative backend.
    Tensor ple_record;
    // Prefill: host n-gram gather and pinned BF16 staging [embed_dim, prefill_chunk].
    PleGather* ple_gather   = nullptr;
    std::uint16_t* ple_host = nullptr;
    // Decode rounds: BF16 [embed_dim, columns] rows of the round's tokens (ordinary: one column per
    // lane; MTP: K+1 columns per lane, lane-major), uploaded by the round from `decode_ple_host`
    // (pinned, written by the Program before the launch).
    Tensor decode_ple_rows;
    const std::uint16_t* decode_ple_host = nullptr;
    // Deferred decode rows (mapped host word of this rank, null = uploaded at the round's start):
    // the round does not upload them; the PLE layer copies them from `decode_ple_host` itself once
    // the Program has raised the word (ops::host::consume_host_flag), so the host gathers them
    // after the launch while the device runs the layers before the PLE layer. Every execution of
    // such a round (graph or eager) consumes one raise.
    std::uint32_t* decode_ple_staged = nullptr;
    Qwen4Taps* taps                  = nullptr;
    Qwen4MoeProbe* moe_probe         = nullptr;
    // T43: the MTP proposal head over a draft token shortlist (rank 0's at tp 2), null = the full
    // Text output head. Program-owned (qwen4_draft_head.h).
    const Qwen4DraftHead* draft_head = nullptr;
};

} // namespace ninfer::models::qwen3_5::execution
