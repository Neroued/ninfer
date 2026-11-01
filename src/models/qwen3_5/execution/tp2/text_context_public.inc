// Class-body fragment of TextContext (execution/text.h), included in its public section: the
// two-device (tp == 2) entry points. Not a header; our file, see docs/maintainer/upstream-merge.md.
//
// One Text call over the rank-0 operands, and with a non-null `tp` over both ranks of a
// two-device Model (tp.h). With `tp`, the text and multimodal prefill chunk (with its MTP prompt
// alignment or DFlash feature capture) and ordinary decode run the Qwen4 forward on both ranks
// (qwen4_text.cpp), and speculative target verification (with or without DFlash feature capture),
// the MTP decode-round forwards and proposals, the MTP bridge and their logits run split through
// the rank-array overloads below; the single-rank MTP and verification entries are rejected with
// std::invalid_argument. The constructor rejects incomplete Text forward and MTP bindings.

public:
// Rank 0's operand then rank 1's, each resident on that rank's device.
using RankTensors = std::array<Tensor, kTensorParallelWidth>;

// The split DFlash2 drafter (execution/tp2/draft_split.h) keeps its own target features on
// rank 1: every call that runs a DFlash feature tap on rank 0's residual then also feeds this
// sink rank 1's copy of the replicated residual, on rank 1's stream, and a prefill chunk's
// consumer appends rank 1's context after rank 0's. Null (the default): rank 1 captures
// nothing. The sink must outlive the calls it serves.
void set_peer_feature_sink(DFlashFeatureSink* sink) noexcept { peer_feature_sink_ = sink; }

// --- tp == 2 speculative forms --------------------------------------------------------------
// Rank 1's control tensors are its own copies of rank 0's values (the Program uploads the same
// host ingress record to both ranks and derives the rest with the same Ops). Token ids a Qwen3.5
// MTP call embeds are rank 0's alone: rank 0's half of the MTP input projection contracts the
// token embedding and rank 1's the target hidden. Sampling, acceptance and the winning draft
// ids are rank 0's, and so are the logits: rank 0 alone gathers them, into its own frames.

// Verification with ReplaySSM records (GdnStateAction::RecordForReplay) on each rank. Both
// ranks keep their normalized hidden [H,W,B]; rank 0's `logits` receive the complete [V,W,B]
// logits and rank 0 writes the per-column argmax into `target_tokens`.
void target_verify_batch(const RankTensors& ids, const RankTensors& cache_positions,
                         const RankTensors& rope_positions, const RankTensors& valid_columns,
                         const RankTensors& kv_table_rows,
                         const RankTensors& linear_state_source_slots,
                         ops::CausalAttentionExecutionEnvelope envelope, const RankTensors& hidden,
                         Tensor& logits, Tensor& target_tokens);
// The same verification with the DFlash2 drafter capturing its target features: `sink` reads
// rank 0's residual after each captured layer's all-reduce (the complete hidden state) on rank
// 0's stream; rank 1 captures into the peer sink when one is set (set_peer_feature_sink).
void target_verify_batch(const RankTensors& ids, const RankTensors& cache_positions,
                         const RankTensors& rope_positions, const RankTensors& valid_columns,
                         const RankTensors& kv_table_rows,
                         const RankTensors& linear_state_source_slots,
                         ops::CausalAttentionExecutionEnvelope envelope, const RankTensors& hidden,
                         Tensor& logits, Tensor& target_tokens, DFlashFeatureSink& sink);
void mtp_forward_decode_batch(const Tensor& ids, const RankTensors& hidden,
                              const RankTensors& cache_positions, const RankTensors& rope_positions,
                              const RankTensors& valid_columns, const RankTensors& kv_table_rows,
                              ops::CausalAttentionExecutionEnvelope envelope,
                              const RankTensors& mtp_hidden);
// `logits` [V,B] is rank 0's.
void mtp_propose_batch(const RankTensors& hidden, Tensor& logits, Tensor& draft_tokens,
                       Tensor* draft_logprobs = nullptr);
// The MTP bridge over both ranks: `ids` [T] (rank 0's alone) against each rank's copy of the
// target hidden [H,T], appending both ranks' MTP K/V at `positions` through each rank's
// prefill MTP KV row. `rope_positions[r]` are [T], or [T,3] for a visual column. Rank 0 embeds
// `ids` unless `input_embeddings` [H,T] (rank 0's) supplies their composed embeddings. With
// `logits_column` >= 0 rank 0 proposes `draft_token` from that column of `mtp_hidden`, over
// logits rank 0 alone gathers.
void mtp_forward_batch(const Tensor& ids, const RankTensors& hidden, const RankTensors& positions,
                       const RankTensors& rope_positions,
                       ops::CausalAttentionExecutionEnvelope envelope,
                       const RankTensors& mtp_hidden, int logits_column, Tensor* logits,
                       Tensor* draft_token, const Tensor* input_embeddings = nullptr);
// One prompt proposal step through each rank's prefill MTP KV row; rank 0 alone gathers the
// proposal logits.
void mtp_forward_ar_step(const Tensor& token, const RankTensors& previous_hidden,
                         const RankTensors& position,
                         ops::CausalAttentionExecutionEnvelope envelope,
                         const RankTensors& mtp_hidden, Tensor& logits, Tensor& draft_token);
