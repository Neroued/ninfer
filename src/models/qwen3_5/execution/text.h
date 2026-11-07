#pragma once

#include "core/gdn_replay_records.h"
#include "core/linear_attention_state.h"
#include "models/common/moe/schedule.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/state/decoder_state.h"
#include "ninfer/ops/softmax_attention.h"

#include <cstdint>
#include <optional>
#include <span>
#include <variant>

namespace ninfer::models::qwen3_5::execution {

struct TextExecutionView {
    const Parameters& parameters;
    MemoryArena& workspace;
    CudaExecutionView execution;
    const PagedKVCache& cache;
    LinearAttentionStateView& states;
    moe::Schedule* moe = nullptr;
};

// Contiguous ids/cache_positions I32 [W,B], RoPE positions I32 [W*B] or [W*B,3],
// valid_columns/kv_table_rows I32 [B], and State byte offsets I64 [B]. SequenceUpdate
// uses Host byte offsets; ReplayRecords only reads source records. Embeddings, when supplied,
// are immutable BF16 [H,W,B] with visual replacements already applied.
// PackedSequenceUpdate flattens its sequences onto one token axis: ids/cache_positions are I32
// [N,1], RoPE positions I32 [N,3], N the summed segment length, and valid_columns/kv_table_rows
// I32 [S], indexed by SequenceSegment::row. State offsets come from the segments.
// MixedUpdate appends the verify block of decode rows to such a token axis: columns
// [0, steps.token_offset) belong to the segments and the remaining steps.width * steps.rows
// columns to the decode rows, row-major with the width fastest. valid_columns/kv_table_rows are
// I32 [S + rows]; segments address their entry through SequenceSegment::row and decode row b
// through steps.row_offset + b, and the two ranges do not overlap. State byte offsets of the
// decode rows are I64 [steps.rows] (destination offsets only without records).
struct TextInputs {
    Tensor ids;
    Tensor cache_positions;
    Tensor rope_positions;
    Tensor valid_columns;
    Tensor kv_table_rows;
    Tensor state_source_offsets;
    Tensor state_destination_offsets;
    std::optional<Tensor> input_embeddings;
};

// SequenceUpdate publishes a complete B=1 sequence with the chunked recurrence; its sole
// valid extent is W. StepUpdate publishes one transition per row, all with valid extent one.
// ReplayRecords leaves master state unchanged and records every valid transition for the
// caller's later accepted-prefix fold. Missing valid_columns means every column is valid.
struct SequenceUpdate {
    std::int64_t source_offset;
    std::int64_t destination_offset;
};

// One independent sequence inside a packed token axis: tokens [token_offset, token_offset+tokens)
// belong to the sequence whose table row and valid length are entry `row` of kv_table_rows and
// valid_columns. Its State is read at source_offset and published at destination_offset (Host
// byte offsets). Segments are consecutive, cover the token axis and share no destination State.
struct SequenceSegment {
    std::uint32_t row;
    std::uint32_t token_offset;
    std::uint32_t tokens;
    std::int64_t source_offset;
    std::int64_t destination_offset;
};

// Publishes every segment as a complete B=1 sequence with the same chunked recurrence and the
// same attention as SequenceUpdate would apply to it alone. Token-wise stages (embedding,
// normalization, projection, routed experts) run once over the whole token axis, so each segment
// matches its separate execution. The segment storage is borrowed for the traversal.
struct PackedSequenceUpdate {
    std::span<const SequenceSegment> segments;
};

// The decode rows of a mixed traversal as a [width, rows] tail slice of the token axis. The
// envelope bounds the keys visible to these rows and is independent of the segments' envelope.
struct StepRows {
    std::uint32_t token_offset;
    std::uint32_t width;
    std::uint32_t rows;
    std::uint32_t row_offset;
    ops::CausalAttentionExecutionEnvelope envelope;
};

// Publishes every segment as PackedSequenceUpdate does and the decode rows as StepUpdate does
// when `records` is null, or as ReplayRecords does otherwise (`records` is then borrowed for the
// traversal and each decode row's valid extent comes from valid_columns). Token-wise stages run
// once over the whole token axis.
struct MixedUpdate {
    std::span<const SequenceSegment> segments;
    StepRows steps;
    const GdnReplayRecords* records;
};

struct StepUpdate {};

struct ReplayRecords {
    const GdnReplayRecords& records;
};

using TextStateUpdate = std::variant<SequenceUpdate, PackedSequenceUpdate, MixedUpdate,
                                     StepUpdate, ReplayRecords>;

// Selected post-layer residuals concatenate along the feature axis. With residency_slots,
// features is BF16 [H*layers,W,Slots], and valid row prefixes scatter to absolute residency
// slots. Otherwise features is BF16 [H*layers,W*B] and optional positions is I32 [W*B].
// The caller owns storage, the stable layer selection and any later draft context append or
// normalization. Layer IDs are distinct and refer to the model's selected draft configuration.
struct DFlashFeatureSink {
    Tensor features;
    Tensor positions;
    Tensor residency_slots;
    std::span<const std::uint32_t> layers;
};

// One mathematical Main traversal. All borrowed storage stays live through completion or
// cancellation. The arena's residual and pending post-mixer scope remain reserved across
// readiness yields; destroy the traversal before reusing that arena for subsequent work.
class TextRun {
public:
    TextRun(TextExecutionView execution, TextInputs inputs, TextStateUpdate update,
            ops::CausalAttentionExecutionEnvelope envelope, Tensor hidden,
            const DFlashFeatureSink* features = nullptr);
    ~TextRun();
    TextRun(const TextRun&)            = delete;
    TextRun& operator=(const TextRun&) = delete;

    // True means the complete output chain was submitted, not that the GPU has retired.
    [[nodiscard]] bool start();
    [[nodiscard]] bool poll();
    void cancel();

    // Explicit mathematical segments for rank-owned capture. These do not start an MoE
    // activity or acquire offloaded weights. A supplied GPU call binds the captured MoE body.
    void prefix();
    void mixer(std::uint32_t layer);
    void post_mixer(std::uint32_t layer, const moe::GpuCall* call = nullptr);
    void capture_features(std::uint32_t layer);
    void finish();
    void set_attention_envelope(ops::CausalAttentionExecutionEnvelope envelope) noexcept;

    [[nodiscard]] Tensor& residual() noexcept { return residual_; }

private:
    enum class Stage : std::uint8_t { Idle, Layer, PostMixer, Done, Cancelled };

    void attention(const BlockParameters&, std::uint32_t compact_layer);
    void gdn(const BlockParameters&, std::uint32_t compact_layer);
    [[nodiscard]] ops::MoeExecutionHints next_projection_hints(std::uint32_t layer) const;
    void complete_layer();

    TextExecutionView view_;
    TextInputs inputs_;
    TextStateUpdate update_;
    ops::CausalAttentionExecutionEnvelope envelope_;
    Tensor hidden_;
    std::optional<DFlashFeatureSink> features_;
    MemoryArena::Scope traversal_scope_;
    Tensor residual_;
    std::optional<moe::Schedule::Activity> activity_;
    std::optional<MemoryArena::Scope> post_mixer_scope_;
    Tensor post_mixer_hidden_;
    std::uint32_t layer_ = 0;
    Stage stage_         = Stage::Idle;
};

} // namespace ninfer::models::qwen3_5::execution
