// Class-body fragment of TextContext (execution/text.h), included at the end of its private
// section: the two-device (tp == 2) bindings, accessors and Qwen3.5 MTP head. The definitions are in
// tp2/text_tp2.inc. Not a header; our file, see docs/maintainer/upstream-merge.md.

private:
// --- tp == 2 -------------------------------------------------------------------------------
// The Text layers run in the Qwen4 forward on both ranks (qwen4_text.cpp); these members hold
// rank 1's bindings and the Qwen3.5 MTP head's split schedule. Rank r's work is issued on
// ExecutionContext::dev[r]'s stream; only the collectives order the two streams.
[[nodiscard]] bool tp2() const noexcept { return tp_ != nullptr; }

void validate_tensor_parallel();
void require_single_device(const char* operation) const;
[[nodiscard]] const Parameters& rank_parameters(int rank) const noexcept;
[[nodiscard]] std::array<WorkspaceArena*, kTensorParallelWidth> workspaces() const noexcept;
[[nodiscard]] cudaStream_t rank_stream(int rank) const noexcept;
// Stream and SM count of `rank`'s device, for Ops whose launch shape depends on the device.
[[nodiscard]] DeviceExecutionView rank_execution_view(int rank) const noexcept;
[[nodiscard]] LinearAttentionStatePool& rank_state(int rank) const noexcept;
[[nodiscard]] const qwen3_5::PagedKVCache& rank_text_cache(int rank) const;
// Both ranks bind valid columns, or neither does (an empty Tensor: every column is live); two
// ranks masking different columns would diverge silently.
[[nodiscard]] Tensor rank_valid_columns(int rank) const;
// The MTP execution rows: the batch binding of an MTP decode call, else the prefill scalar.
[[nodiscard]] const Tensor& rank_backend_kv_table_rows(int rank) const;
[[nodiscard]] const qwen3_5::PagedKVCache& rank_mtp_cache(int rank) const;
[[nodiscard]] qwen3_5::PagedKVCacheView rank_mtp_kv(int rank) const;
[[nodiscard]] const MtpParameters& rank_mtp(int rank) const;
[[nodiscard]] const GdnReplayRecords& rank_replay_records(int rank) const;
void validate_tensor_parallel_mtp();
// Vocabulary-split head: each rank projects its half of the vocabulary from its final hidden
// columns, and rank 0 alone gathers the complete [V, C] logits into its `logits` (one pull of
// rank 1's half).
void logits_tp2(const RankTensors& hidden, Tensor& logits);
void logits_tp2(const RankTensors& hidden, const std::array<const LinearParameters*, 2>& head,
                Tensor& logits);
// MTP head over both ranks: three all-reduces (input projection, attention output, post-mixer
// down projection) leave the replicated residual identical on the two ranks. The attention runs
// over each rank's half of the heads and its own MTP KV pages.
// `input_embeddings`, when set, is rank 0's composed embedding of `ids`.
void mtp_forward_stem_tp2(const Tensor& ids, const RankTensors& hidden,
                          const Tensor* input_embeddings, RankTensors& x, RankTensors& ah,
                          const RankTensors& staging);
void mtp_forward_tail_tp2(RankTensors& x, const RankTensors& ah, const RankTensors& positions,
                          const RankTensors& rope_positions,
                          ops::CausalAttentionExecutionEnvelope envelope,
                          const RankTensors& mtp_hidden, const RankTensors& staging);
void mtp_forward_core_tp2(const Tensor& ids, const RankTensors& hidden,
                          const RankTensors& positions, const RankTensors& rope_positions,
                          ops::CausalAttentionExecutionEnvelope envelope,
                          const RankTensors& mtp_hidden, const Tensor* input_embeddings);
void mtp_prompt_chunk_tp2(std::span<const int> alignment_ids, std::uint32_t base, int len, int T,
                          int base_i, bool is_last, const RankTensors& mtp_input,
                          const RankTensors& positions, const RankTensors& rope_positions,
                          ops::CausalAttentionExecutionEnvelope chunk_envelope, bool multimodal,
                          const VisionChunk& vision_chunk);
void mtp_prefill_chunk_tp2(const Tensor& ids, const RankTensors& hidden,
                           const Tensor* input_embeddings, const RankTensors& positions,
                           const RankTensors& rope_positions,
                           ops::CausalAttentionExecutionEnvelope envelope, bool final_chunk,
                           const RankTensors* final_hidden, Tensor* logits, Tensor* draft_token);
// The optimized proposal head proposes on rank 0 alone when PrimaryOnly, or on both ranks when
// split by vocabulary (execution::proposal_argmax_split, rank 0 combining the two argmaxes);
// the full head is the vocabulary-split output head, gathered before rank 0's argmax.
void proposal_argmax_tp2(const RankTensors& hidden, Tensor& logits, Tensor& proposal_tokens);
// rank 1's StateImage column for set_rewrite_checkpoint_hidden_output() (MTP only).
Tensor* peer_rewrite_checkpoint_hidden_output_ = nullptr;
// The split DFlash2 drafter's rank 1 feature sink (set_peer_feature_sink); null otherwise.
DFlashFeatureSink* peer_feature_sink_ = nullptr;
// Rank 1's calls on peer_feature_sink_, beside the rank 0 tap's (no-ops without a peer sink).
void peer_tap_begin(const Tensor& x1);
void peer_tap_capture_layer(int layer, const Tensor& x1);
void peer_tap_capture_layer_columns(int layer, const Tensor& x1, std::int32_t first);
void peer_tap_capture_positions(const Tensor& positions1);
void peer_tap_consume_prefill_chunk(std::int32_t tokens, bool rewrite_checkpoint);

const TpExecution* tp_ = nullptr;
// Each rank's share of config_ (shard_text_config), present only at tp == 2.
std::optional<TextConfig> shard_config_;
const Tensor* peer_valid_columns_         = nullptr;
const Tensor* peer_backend_kv_table_rows_ = nullptr;
