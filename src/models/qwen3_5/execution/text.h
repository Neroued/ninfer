#pragma once
#include "models/qwen3_5/program/internal.h"


#include "core/arena.h"
#include "core/device.h"
#include "core/gdn_replay_records.h"
#include "core/linear_attention_state.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/softmax_attention.h"
#include "models/qwen3_5/state/decoder_state.h"
#include "models/qwen3_5/execution/tp.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "models/qwen3_5/program/round_buffers.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace ninfer::models::qwen3_5::execution {

enum class GdnStateAction : std::uint8_t {
    UpdateInPlace,
    RecordForReplay,
};

struct PrefillChunkResult {
    std::uint32_t processed_tokens = 0;
    bool finalized                 = false;
    runtime::ExecutionTiming timing;
};

struct DFlashFeatureSink {
    using PrefillConsumer = std::function<void(const Tensor&, const Tensor&, bool)>;

    Tensor* features                  = nullptr;
    Tensor* positions                 = nullptr;
    Tensor* batch_features            = nullptr;
    const Tensor* batch_lanes         = nullptr;
    const Tensor* batch_valid_columns = nullptr;
    std::int32_t batch_width          = 0;
    std::int32_t batch_size           = 0;
    std::span<const std::uint32_t> layers;
    PrefillConsumer consume_prefill;
    std::uint32_t captured_mask = 0;
    std::int32_t active_tokens  = 0;

    void begin(const Tensor& value);
    void capture_layer(int layer, const Tensor& value, cudaStream_t stream);
    // A prefill chunk's layer output for the column window [first, first + value.ne[1]) only (the
    // TP2 prefill overlap captures each half once its exchange has finished). The same bytes, at
    // the same place, as capture_layer copies for those columns.
    void capture_layer_columns(int layer, const Tensor& value, std::int32_t first,
                               cudaStream_t stream);
    void capture_positions(const Tensor& source, cudaStream_t stream);
    void consume_prefill_chunk(std::int32_t tokens, bool rewrite_checkpoint);
};

class VisionPrefillSession;
struct VisionChunk;
struct Qwen4Runtime;
struct Qwen4Columns;
struct Qwen4State;
template <class Arena>
struct Qwen4Rank;
template <class Arena>
class Qwen4Forward;

class TextContext {
public:
    TextContext(DeviceContext& ctx, const execution::Parameters& weights, WorkspaceArena& work,
                qwen3_5::PagedKVCacheView kv, LinearAttentionStatePool& state,
                qwen3_5::RoundState& io, Tensor& prefill_hidden, std::uint32_t prefill_chunk,
                std::uint32_t text_kv_base,
                qwen3_5::PagedKVCacheView mtp_kv           = qwen3_5::PagedKVCacheView(),
                const qwen3_5::PagedKVCache* batch_text_kv = nullptr,
                const qwen3_5::PagedKVCache* batch_mtp_kv  = nullptr,
                const TpExecution* tp = nullptr, std::int32_t rope_delta = 0);
    ~TextContext();

    TextContext(const TextContext&)            = delete;
    TextContext& operator=(const TextContext&) = delete;

    void set_proposal_head(const LinearParameters* weight, const std::int32_t* ids,
                           int count) noexcept {
        proposal_head_     = weight;
        proposal_head_ids_ = ids;
        proposal_head_n_   = count;
    }

    void set_sampling(const ops::SamplingConfig* config) noexcept { sampling_config_ = config; }

    // Structured output: a constrained prompt's first token is never redrawn by the reasoning EOS
    // guard (program/context.h PrefillContext::reasoning_guard).
    void set_reasoning_guard(bool enabled) noexcept { reasoning_guard_ = enabled; }

    void set_prefill_split_frontier(std::int64_t position) noexcept {
        prefill_split_frontier_ = position;
    }

    // `peer_output` is rank 1's StateImage column at tensor-parallel width 2 under MTP, the only
    // backend whose rank 1 reads a retained hidden (the MTP bridge); null otherwise.
    void set_rewrite_checkpoint_hidden_output(Tensor* output,
                                              Tensor* peer_output = nullptr) noexcept {
        rewrite_checkpoint_hidden_output_      = output;
        peer_rewrite_checkpoint_hidden_output_ = peer_output;
    }

    void set_mtp_proposal_extent(std::uint32_t extent) noexcept { mtp_proposal_extent_ = extent; }

    void set_linear_state_slots(std::int32_t source_slot, std::int32_t destination_slot);
    void set_gdn_state_action(GdnStateAction action, const GdnReplayRecords* replay_records);

    // The Program's runtime inputs of the Text forward (expert backend, PLE states and staging;
    // the experts only with routed experts). Required before any Text call.
    void set_qwen4_runtime(const Qwen4Runtime* runtime) noexcept { qwen4_runtime_ = runtime; }

    [[nodiscard]] const LinearParameters* proposal_head() const noexcept { return proposal_head_; }

    [[nodiscard]] int proposal_head_n() const noexcept { return proposal_head_n_; }

    [[nodiscard]] PrefillChunkResult prefill_chunk(std::span<const int> full_ids,
                                                   std::uint32_t begin,
                                                   std::uint32_t nominal_length,
                                                   bool finalize_at_end);
    [[nodiscard]] PrefillChunkResult prefill_chunk(std::span<const int> full_ids,
                                                   std::uint32_t begin,
                                                   std::uint32_t nominal_length,
                                                   bool finalize_at_end, DFlashFeatureSink& sink);
    [[nodiscard]] PrefillChunkResult
    prefill_chunk(const qwen3_5::PreparedPromptData& input, std::uint32_t begin,
                  std::uint32_t nominal_length, VisionPrefillSession& vision, bool finalize_at_end);
    [[nodiscard]] PrefillChunkResult prefill_chunk(const qwen3_5::PreparedPromptData& input,
                                                   std::uint32_t begin,
                                                   std::uint32_t nominal_length,
                                                   VisionPrefillSession& vision,
                                                   bool finalize_at_end, DFlashFeatureSink& sink);
    void ordinary_decode_batch(const Tensor& ids, const Tensor& cache_positions,
                               const Tensor& rope_positions, const Tensor& kv_table_rows,
                               const Tensor& linear_state_source_slots,
                               const Tensor& linear_state_destination_slots,
                               ops::CausalAttentionExecutionEnvelope envelope, Tensor& hidden,
                               Tensor& logits);
    void target_verify_batch(const Tensor& ids, const Tensor& cache_positions,
                             const Tensor& rope_positions, const Tensor& valid_columns,
                             const Tensor& kv_table_rows, const Tensor& linear_state_source_slots,
                             ops::CausalAttentionExecutionEnvelope envelope, Tensor& hidden,
                             Tensor& logits, Tensor& target_tokens);
    void target_verify_batch(const Tensor& ids, const Tensor& cache_positions,
                             const Tensor& rope_positions, const Tensor& valid_columns,
                             const Tensor& kv_table_rows, const Tensor& linear_state_source_slots,
                             ops::CausalAttentionExecutionEnvelope envelope, Tensor& hidden,
                             Tensor& logits, Tensor& target_tokens, DFlashFeatureSink& sink);
    void mtp_forward_decode_batch(const Tensor& ids, const Tensor& hidden,
                                  const Tensor& cache_positions, const Tensor& rope_positions,
                                  const Tensor& valid_columns, const Tensor& kv_table_rows,
                                  ops::CausalAttentionExecutionEnvelope envelope,
                                  Tensor& mtp_hidden);
    // `draft_logprobs` (FP32 [B], optional): each proposal's softmax log-probability over the
    // proposal vocabulary (T41's adaptive window; the draft shortlist's rows under T43).
    void mtp_propose_batch(const Tensor& hidden, Tensor& logits, Tensor& draft_tokens,
                           Tensor* draft_logprobs = nullptr);
    void mtp_forward_batch(const Tensor& ids, const Tensor& hidden, const Tensor& positions,
                           ops::CausalAttentionExecutionEnvelope envelope, Tensor& mtp_hidden,
                           int logits_column, Tensor* logits, Tensor* draft_token,
                           const Tensor* explicit_rope_positions = nullptr,
                           const Tensor* input_embeddings        = nullptr);
    void mtp_forward_ar_step(const Tensor& token, const Tensor& previous_hidden,
                             const Tensor& position, ops::CausalAttentionExecutionEnvelope envelope,
                             Tensor& mtp_hidden, Tensor& logits, Tensor& draft_token);
    // Qwen4Exp: the complete logits [V,1] (on rank 0) of one retained wide residual [S*H,1]
    // through the Text output mixer and head. At tp 2 `peer_hidden` is rank 1's copy, or null to
    // copy rank 0's (the residual is replicated across the ranks).
    void qwen4_logits_from_hidden(const Tensor& hidden, const Tensor* peer_hidden, Tensor& logits);

#include "models/qwen3_5/execution/tp2/text_context_public.inc"
private:
    [[nodiscard]] bool mtp_enabled() const noexcept {
        return mtp_kv_.valid() || batch_mtp_kv_ != nullptr;
    }

    void mtp_forward_stem(const Tensor& ids, const Tensor& hidden, const Tensor* input_embeddings,
                          Tensor& x, Tensor& ah);
    void mtp_forward_tail(Tensor& x, const Tensor& ah, const Tensor& positions,
                          const Tensor& rope_positions,
                          ops::CausalAttentionExecutionEnvelope envelope, Tensor& mtp_hidden);
    void mtp_forward_core(const Tensor& ids, const Tensor& hidden, const Tensor& positions,
                          const Tensor& rope_positions,
                          ops::CausalAttentionExecutionEnvelope envelope, Tensor& mtp_hidden,
                          const Tensor* input_embeddings);
    void mtp_prefill_chunk(const Tensor& ids, const Tensor& hidden, const Tensor* input_embeddings,
                           const Tensor& positions, const Tensor& rope_positions,
                           ops::CausalAttentionExecutionEnvelope envelope, bool final_chunk,
                           Tensor* final_hidden, Tensor* logits, Tensor* draft_token);
    void proposal_argmax(const Tensor& hidden, Tensor& logits, Tensor& proposal_tokens);
    // The Qwen3.5 MTP head's prompt alignment of one prefill chunk (text.cpp; driven by
    // qwen4_text.cpp's prefill).
    void mtp_prompt_chunk(std::span<const int> alignment_ids, std::uint32_t alignment_begin,
                          int len, int total, int base_i, bool is_last, const Tensor& xf,
                          const Tensor& positions, const Tensor& rope_positions,
                          ops::CausalAttentionExecutionEnvelope chunk_envelope, bool multimodal,
                          const VisionChunk& vision_chunk);

    // The MTP head the Qwen4 forward drives: the Qwen4 layer (qwen4_mtp_, hyper-connections) or,
    // for a dense Qwen3.5 model, the Qwen3.5 head (mtp_) through the Qwen3.5 MTP entry points.
    [[nodiscard]] bool qwen4_mtp_layer() const noexcept { return qwen4_mtp_ != nullptr; }

    struct MultimodalPrefill {
        std::span<const int> token_ids;
        std::span<const std::int32_t> positions;
        VisionPrefillSession* vision = nullptr;
        std::uint32_t begin          = 0;
        std::int32_t rope_delta      = 0;
    };

    struct TextPrefill {
        std::span<const int> token_ids;
        std::uint32_t begin = 0;
    };

    // The Text forward (execution/qwen4_text.cpp, qwen4_forward.h).

    // Rows of the hidden a round carries: the wide residual (S*H) with hyper-connections, the
    // final-normed hidden (H) without them.
    [[nodiscard]] std::int32_t carried_hidden_rows() const noexcept {
        return dimension(config_.residual_width());
    }

    [[nodiscard]] const Qwen4Runtime& qwen4_runtime() const;
    [[nodiscard]] std::span<const int> qwen4_prompt(std::span<const int> ids,
                                                    const TextPrefill* text_prefill,
                                                    const MultimodalPrefill* multimodal) const;
    // One prompt chunk of a text (`text_prefill`) or multimodal (`multimodal`) prompt: exactly one
    // of them names the whole prompt.
    // `sink` (plain residual only): a DFlash drafter's target features of the chunk.
    [[nodiscard]] PrefillChunkResult qwen4_prefill(std::span<const int> ids,
                                                   const TextPrefill* text_prefill,
                                                   const MultimodalPrefill* multimodal,
                                                   bool finalize_at_end,
                                                   DFlashFeatureSink* sink = nullptr);
    // A DFlashFeatureSink (and at tp 2 the peer sink) as the Qwen4 forward's feature tap.
    class Qwen4SinkTap;

    // The shifted visual columns of a multimodal chunk's MTP prompt input: each rank's merged
    // Vision embeddings of the chunk's item (rank 1's only at tp 2), the chunk-relative MTP
    // columns whose shifted token is visual and the item's first merged token among them.
    struct Qwen4MtpVisual {
        RankTensors merged;
        std::span<const std::int32_t> columns;
        std::size_t source_begin = 0;
    };

    // The forward's ranks: 2 at tensor-parallel width 2, else 1; each rank's arena (rank 1's null
    // at one device), runtime and device.
    [[nodiscard]] int qwen4_width() const noexcept;
    [[nodiscard]] std::array<WorkspaceArena*, 2> qwen4_arenas() const noexcept;
    void reset_qwen4_arenas();
    [[nodiscard]] std::array<const Qwen4Runtime*, 2> qwen4_runtimes() const;
    [[nodiscard]] int qwen4_device(int rank) const noexcept;
    // Rank `rank`'s operands of a Text forward over `columns` and `state`: at one device the
    // logical config, at tp 2 the rank's shard geometry.
    [[nodiscard]] Qwen4Rank<WorkspaceArena> qwen4_rank(int rank, const Qwen4Columns& columns,
                                                       const Qwen4State& state) const;
    // The Text forward over `ranks` (rank 0 first) for `columns` columns.
    [[nodiscard]] Qwen4Forward<WorkspaceArena>
    qwen4_forward(const std::array<Qwen4Rank<WorkspaceArena>, 2>& ranks,
                  std::int32_t columns) const;
    // Rank `rank`'s Main KV cache, with its QSA index planes for a model with QSA (`call` names
    // the execution in the error).
    [[nodiscard]] const qwen3_5::PagedKVCache& qwen4_text_cache(int rank, const char* call) const;
    // The complete logits on rank 0 from each rank's output head input (vocabulary-split at tp 2).
    void qwen4_logits(const RankTensors& x, Tensor& logits);
    // Each rank's copy of rank 0's device ids (rank 1's: qwen4_ids_on_rank1).
    [[nodiscard]] RankTensors qwen4_rank_ids(const Tensor& ids);
    // Every rank's Qwen4 MTP layer, with (`cache`) the MTP KV cache and its QSA planes required.
    [[nodiscard]] std::array<const Qwen4MtpParameters*, 2>
    qwen4_mtp_layers(const char* call, bool cache = true) const;

    void qwen4_ordinary_decode(const Tensor& ids, const Tensor& cache_positions,
                               const Tensor& rope_positions, const Tensor& kv_table_rows,
                               const Tensor& source_slots, const Tensor& destination_slots,
                               ops::CausalAttentionExecutionEnvelope envelope, Tensor& hidden,
                               Tensor& logits);
    // tp 2 (both ranks in lockstep, rank 1's operands from TpExecution; qwen4_text.cpp).
    void validate_tensor_parallel_qwen4();
    [[nodiscard]] const Qwen4Runtime& qwen4_peer_runtime() const;
    // Rank 1's copy (its transient arena) of rank 0's device ids, ordered after rank 0's stream.
    [[nodiscard]] Tensor qwen4_ids_on_rank1(const Tensor& ids);
    // The speculative calls take each rank's operands (rank 1's only at tp 2); the one-device
    // overloads validate single tensors and run as rank 0. MTP (T33 at tp 2): the verification,
    // the MTP layer per rank (each rank embeds its own copy of the ids), the proposal through the
    // replicated MTP output mixer and the output head, and the prompt's MTP KV and drafts.
    void qwen4_target_verify(const Tensor& ids, const Tensor& cache_positions,
                             const Tensor& rope_positions, const Tensor& valid_columns,
                             const Tensor& kv_table_rows, const Tensor& source_slots,
                             ops::CausalAttentionExecutionEnvelope envelope, Tensor& hidden,
                             Tensor& logits, Tensor& target_tokens,
                             DFlashFeatureSink* sink = nullptr);
    void qwen4_target_verify(const RankTensors& ids, const RankTensors& cache_positions,
                             const RankTensors& rope_positions, const RankTensors& valid_columns,
                             const RankTensors& kv_table_rows, const RankTensors& source_slots,
                             ops::CausalAttentionExecutionEnvelope envelope,
                             const RankTensors& hidden, Tensor& logits, Tensor& target_tokens,
                             DFlashFeatureSink* sink = nullptr);
    void qwen4_mtp_forward_decode(const Tensor& ids, const Tensor& hidden,
                                  const Tensor& cache_positions, const Tensor& rope_positions,
                                  const Tensor& valid_columns, const Tensor& kv_table_rows,
                                  ops::CausalAttentionExecutionEnvelope envelope,
                                  Tensor& mtp_hidden);
    void qwen4_mtp_forward_decode(const Tensor& ids, const RankTensors& hidden,
                                  const RankTensors& cache_positions,
                                  const RankTensors& rope_positions,
                                  const RankTensors& valid_columns,
                                  const RankTensors& kv_table_rows,
                                  ops::CausalAttentionExecutionEnvelope envelope,
                                  const RankTensors& mtp_hidden);
    void qwen4_mtp_forward_batch(const Tensor& ids, const Tensor& hidden, const Tensor& positions,
                                 ops::CausalAttentionExecutionEnvelope envelope, Tensor& mtp_hidden,
                                 int logits_column, Tensor* logits, Tensor* draft_token,
                                 const Tensor* explicit_rope_positions,
                                 const Tensor* input_embeddings);
    // T66b at tp 2: the prefix-reuse bridge column and its proposal steps over both ranks.
    void qwen4_mtp_forward_batch(const Tensor& ids, const RankTensors& hidden,
                                 const RankTensors& positions, const RankTensors& rope_positions,
                                 ops::CausalAttentionExecutionEnvelope envelope,
                                 const RankTensors& mtp_hidden, int logits_column, Tensor* logits,
                                 Tensor* draft_token, const Tensor* input_embeddings);
    void qwen4_mtp_propose(const RankTensors& hidden, Tensor& logits, Tensor& draft_tokens,
                           Tensor* draft_logprobs = nullptr);
    struct Qwen4ChunkInputs;
    [[nodiscard]] Qwen4ChunkInputs qwen4_chunk_inputs(std::span<const int> ids, int len, int base_i,
                                                      std::span<const std::int32_t> rope_rows,
                                                      std::span<const std::int32_t> visual_columns,
                                                      std::int32_t merged_begin,
                                                      const VisionChunk& vision_chunk,
                                                      std::uint32_t ple_width);
    void upload_chunk_ple(Qwen4ChunkInputs& in, int rank, int len, std::uint32_t ple_width);
    void qwen4_prompt_token(Qwen4Forward<WorkspaceArena>& forward, const RankTensors& xf, int len,
                            int total, std::span<const int> prompt);
    void qwen4_retain_chunk(const RankTensors& xf, int len, bool rewrite, DFlashFeatureSink* sink);
    void qwen4_mtp_prompt_chunk(std::span<const int> prompt, std::uint32_t base, int len,
                                bool is_last, const RankTensors& xf, const RankTensors& positions,
                                const RankTensors& rope_positions, const VisionChunk& vision_chunk);
    void qwen4_mtp_columns(const std::array<const Qwen4MtpParameters*, 2>& mtp,
                           const RankTensors& ids, const RankTensors& hidden, RankTensors& out,
                           const RankTensors& positions, const RankTensors& rope_positions,
                           std::int32_t count, bool append_only, std::uint32_t max_visible,
                           const RankTensors* embeddings);
    void qwen4_mtp_drafts(const std::array<const Qwen4MtpParameters*, 2>& mtp,
                          RankTensors& ar_hidden, std::uint32_t max_visible);
    void qwen4_mtp_prefill(const Tensor& ids, const RankTensors& hidden,
                           const RankTensors& positions, const RankTensors& rope_positions,
                           std::uint32_t max_visible, bool propose,
                           const Qwen4MtpVisual* visual = nullptr);
    DeviceContext& ctx_;
    const Parameters& parameters_;
    const TextConfig& config_;
    WorkspaceArena& work_;
    qwen3_5::PagedKVCacheView kv_;
    qwen3_5::PagedKVCacheView mtp_kv_;
    const qwen3_5::PagedKVCache* batch_text_kv_ = nullptr;
    const qwen3_5::PagedKVCache* batch_mtp_kv_  = nullptr;
    LinearAttentionStatePool& state_;
    qwen3_5::RoundState& io_;
    Tensor& prefill_hidden_;
    std::uint32_t prefill_chunk_;
    std::uint32_t text_kv_base_;
    const Tensor* active_valid_columns_         = nullptr;
    const Tensor* active_backend_kv_table_rows_ = nullptr;
    std::int32_t active_sequence_batch_         = 0;
    std::int32_t active_sequence_width_         = 0;
    std::int32_t rope_delta_                    = 0;
    std::int32_t linear_state_source_slot_      = 0;
    std::int32_t linear_state_destination_slot_ = 0;
    GdnStateAction gdn_state_action_            = GdnStateAction::UpdateInPlace;
    const GdnReplayRecords* replay_records_     = nullptr;
    std::int64_t prefill_split_frontier_        = -1;
    Tensor* rewrite_checkpoint_hidden_output_   = nullptr;
    std::uint32_t mtp_proposal_extent_          = 0;

    const Weight* embed_                        = nullptr;
    const LinearParameters* lm_head_            = nullptr;
    const LinearParameters* proposal_head_      = nullptr;
    const std::int32_t* proposal_head_ids_      = nullptr;
    int proposal_head_n_                        = 0;
    const ops::SamplingConfig* sampling_config_ = nullptr;
    bool reasoning_guard_                       = true;
    const MtpParameters* mtp_                   = nullptr;
    const Qwen4Runtime* qwen4_runtime_          = nullptr;
    const Qwen4MtpParameters* qwen4_mtp_        = nullptr;

#include "models/qwen3_5/execution/tp2/text_context_private.inc"
};

} // namespace ninfer::models::qwen3_5::execution
