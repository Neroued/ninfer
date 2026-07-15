#pragma once

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/gqa_attention.h"
#include "targets/qwen3_6_27b_rtx5090/impl/config.h"
#include "targets/qwen3_6_27b_rtx5090/impl/load/bindings.h"
#include "core/kv_cache.h"
#include "targets/qwen3_6_27b_rtx5090/impl/state/state_store.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::targets::qwen3_6_27b_rtx5090::detail::schedule {

// Target-private compatibility vocabulary for the mechanically preserved fixed schedule. It is
// data-only: TextContext is constructed on the stack for one schedule recording/execution and owns
// neither weights nor device state.
struct ModelConfig {
    static constexpr int hidden              = TextConfig::hidden;
    static constexpr int n_layers            = TextConfig::layers;
    static constexpr int intermediate        = TextConfig::intermediate;
    static constexpr int vocab               = TextConfig::output_rows;
    static constexpr int token_domain        = TextConfig::token_domain;
    static constexpr int gdn_k_heads         = TextConfig::gdn_key_heads;
    static constexpr int gdn_k_dim           = TextConfig::gdn_key_head_dim;
    static constexpr int gdn_v_heads         = TextConfig::gdn_value_heads;
    static constexpr int gdn_v_dim           = TextConfig::gdn_value_head_dim;
    static constexpr int n_q                 = TextConfig::query_heads;
    static constexpr int n_kv                = TextConfig::kv_heads;
    static constexpr int head_dim            = TextConfig::head_dim;
    static constexpr int rotary_dim          = TextConfig::rotary_dim;
    static constexpr int key_dim             = TextConfig::key_dim;
    static constexpr int value_dim           = TextConfig::value_dim;
    static constexpr int conv_dim            = TextConfig::convolution_dim;
    static constexpr int q_size              = TextConfig::query_size;
    static constexpr int kv_size             = TextConfig::kv_size;
    static constexpr int mtp_fc_in           = TextConfig::mtp_input_rows;
    static constexpr int mtp_attn_in         = TextConfig::mtp_attention_input_rows;
    static constexpr int mtp_mlp_gateup_rows = TextConfig::mtp_mlp_gate_up_rows;
    static constexpr float rms_eps           = TextConfig::rms_epsilon;
    static constexpr float rope_theta        = TextConfig::rope_theta;
    static constexpr int mtp_layers          = TextConfig::mtp_layers;

    [[nodiscard]] static constexpr bool is_full(int layer) {
        return TextConfig::is_full_attention(layer);
    }

    [[nodiscard]] static constexpr int n_full() { return TextConfig::full_attention_layers(); }

    [[nodiscard]] static constexpr int n_gdn() { return TextConfig::gdn_layers(); }

    [[nodiscard]] static constexpr int full_idx(int layer) {
        return TextConfig::full_attention_index(layer);
    }

    [[nodiscard]] static constexpr int gdn_idx(int layer) { return TextConfig::gdn_index(layer); }
};

inline constexpr ModelConfig kCfg{};
inline constexpr float kAttnScale                     = kAttentionScale;
inline constexpr std::uint32_t kPrefillChunkAlignment = 128;

struct MlpW {
    const Weight* gate    = nullptr;
    const Weight* up      = nullptr;
    const Weight* gate_up = nullptr;
    const Weight* down    = nullptr;
};

struct FullLayerW {
    const Tensor* input_norm     = nullptr;
    const Weight* q_proj         = nullptr;
    const Weight* gate_proj      = nullptr;
    const Weight* k_proj         = nullptr;
    const Weight* v_proj         = nullptr;
    const Weight* qkv_q4         = nullptr;
    const Weight* gatev_q5       = nullptr;
    const Weight* o_proj         = nullptr;
    const Tensor* q_norm         = nullptr;
    const Tensor* k_norm         = nullptr;
    const Tensor* post_attn_norm = nullptr;
    MlpW mlp;
};

struct GdnLayerW {
    const Tensor* input_norm     = nullptr;
    const Weight* in_q           = nullptr;
    const Weight* in_k           = nullptr;
    const Weight* in_qk_q4       = nullptr;
    const Weight* in_v           = nullptr;
    const Weight* in_z           = nullptr;
    const Weight* in_a           = nullptr;
    const Weight* in_b           = nullptr;
    const Tensor* conv1d         = nullptr;
    const Tensor* a_log          = nullptr;
    const Tensor* dt_bias        = nullptr;
    const Tensor* gdn_norm       = nullptr;
    const Weight* out_proj       = nullptr;
    const Tensor* post_attn_norm = nullptr;
    MlpW mlp;
};

struct MtpW {
    const Weight* fc                    = nullptr;
    const Tensor* pre_fc_norm_embedding = nullptr;
    const Tensor* pre_fc_norm_hidden    = nullptr;
    const Tensor* input_norm            = nullptr;
    const Weight* attn_in               = nullptr;
    const Weight* q_proj                = nullptr;
    const Weight* gate_proj             = nullptr;
    const Weight* k_proj                = nullptr;
    const Weight* v_proj                = nullptr;
    const Tensor* q_norm                = nullptr;
    const Tensor* k_norm                = nullptr;
    const Weight* o_proj                = nullptr;
    const Tensor* post_attn_norm        = nullptr;
    const Weight* gate_up               = nullptr;
    const Weight* down                  = nullptr;
    const Tensor* norm                  = nullptr;
};

struct StepState {
    Tensor token;
    Tensor pos;
    Tensor rope_pos;
    Tensor rope_delta;
    Tensor logits;
    Tensor verify_hidden;
    Tensor prefill_hidden;
    Tensor target_tokens;
    Tensor drafts;
    Tensor sampled_out;
    Tensor num_sampled;
    Tensor verify_ids;
    Tensor shifted_ids;
    Tensor positions;
    Tensor window_base;
    Tensor accepted;
    Tensor gdn_initial_slot;
    Tensor ar_pos;
    Tensor mtp_ar_hidden;
    Tensor stats;
};

enum class Modality : std::uint8_t {
    Image = 1,
    Video = 2,
};

struct VisionGrid {
    int t = 0;
    int h = 0;
    int w = 0;
};

struct TokenSpan {
    std::size_t begin = 0;
    std::size_t count = 0;
};

struct VisionItem {
    Modality modality = Modality::Image;
    VisionGrid grid;
    std::size_t patch_begin = 0;
    std::size_t patch_count = 0;
    std::vector<double> timestamps;
    std::vector<TokenSpan> token_spans;
};

struct PreprocessStats {
    std::size_t media_items       = 0;
    std::uint64_t raw_patches     = 0;
    std::uint64_t vision_tokens   = 0;
    std::uint64_t attention_pairs = 0;
    std::size_t prompt_tokens     = 0;
    std::size_t patch_bytes       = 0;
};

struct ProcessedInput {
    std::vector<int> input_ids;
    std::vector<std::uint8_t> token_types;
    std::vector<std::int32_t> positions;
    std::int32_t rope_delta = 0;
    std::vector<float> patches;
    std::vector<VisionItem> vision_items;
    PreprocessStats stats;
};

enum class Phase {
    Prefill,
    Decode,
    Verify,
};

enum class TapId {
    AfterEmbed,
    AfterMixer,
    AfterMlp,
    AfterFinalNorm,
    AfterLogits,
};

using TextTapCallback = void (*)(void*, TapId, int, Phase, const Tensor&, cudaStream_t);

struct NullTap {
    static constexpr bool enabled = false;
};

class TextContext {
public:
    TextContext(DeviceContext& ctx, const LoadedModelData& weights, WorkspaceArena& work,
                KVCache& kv, GdnState& state, StepState& io, std::uint32_t prefill_chunk,
                std::uint32_t text_kv_base, KVCache* mtp_kv = nullptr);
    ~TextContext();

    TextContext(const TextContext&)            = delete;
    TextContext& operator=(const TextContext&) = delete;

    void set_lm_head_draft(const Weight* weight, const std::int32_t* ids, int count) noexcept {
        lm_head_draft_     = weight;
        lm_head_draft_ids_ = ids;
        lm_head_draft_n_   = count;
    }

    void set_sampling(const ops::SamplingConfig* config) noexcept { sampling_config_ = config; }

    void set_prefill_snapshot_boundary(std::int64_t position) noexcept {
        prefill_snapshot_boundary_ = position;
    }

    void set_boundary_hidden_output(Tensor* output) noexcept { boundary_hidden_output_ = output; }

    [[nodiscard]] const Weight* lm_head_draft() const noexcept { return lm_head_draft_; }

    [[nodiscard]] const std::int32_t* lm_head_draft_ids() const noexcept {
        return lm_head_draft_ids_;
    }

    [[nodiscard]] int lm_head_draft_n() const noexcept { return lm_head_draft_n_; }

    [[nodiscard]] bool mtp_prompt_prepared() const noexcept { return mtp_prompt_prepared_; }

    void prefill(std::span<const int> ids);
    void prefill(const ProcessedInput& input, const Tensor& visual_embeddings);
    void diagnostic_prefill(std::span<const int> ids, void* context, TextTapCallback callback);
    void diagnostic_prefill(const ProcessedInput& input, const Tensor& visual_embeddings,
                            void* context, TextTapCallback callback);
    void target_verify(const Tensor& ids, const Tensor& positions,
                       ops::GqaExecutionEnvelope envelope);
    void diagnostic_target_verify(const Tensor& ids, const Tensor& positions,
                                  ops::GqaExecutionEnvelope envelope, void* context,
                                  TextTapCallback callback);
    void mtp_forward_batch(const Tensor& ids, const Tensor& hidden, const Tensor& positions,
                           ops::GqaExecutionEnvelope envelope, Tensor& mtp_hidden,
                           int logits_column, Tensor* logits, Tensor* draft_token);
    void mtp_forward_ar_step(const Tensor& token, const Tensor& previous_hidden,
                             const Tensor& position, ops::GqaExecutionEnvelope envelope,
                             Tensor& mtp_hidden, Tensor& logits, Tensor& draft_token);
    void mtp_sample_from_hidden_row(const Tensor& mtp_hidden, const Tensor& row, Tensor& out_hidden,
                                    Tensor& logits, Tensor& draft_token);

private:
    void bind();

    [[nodiscard]] bool mtp_enabled() const noexcept { return mtp_kv_ != nullptr; }

    [[nodiscard]] const MtpW& mtp_weights() const;
    void attn_mix(const FullLayerW& weights, Tensor& x, int index, Phase phase);
    void gdn_mix(const GdnLayerW& weights, Tensor& x, int index, Phase phase);
    void mlp_tail(const Tensor* post_norm, const MlpW& weights, Tensor& x, Phase phase);
    void run_layers(Tensor& x, Phase phase);
    template <class Tap>
    void run_layers(Tensor& x, Phase phase, Tap& tap);

    void mtp_forward_stem(const Tensor& ids, const Tensor& hidden, const Tensor* input_embeddings,
                          Tensor& x, Tensor& ah);
    void mtp_forward_tail(Tensor& x, const Tensor& ah, const Tensor& positions,
                          const Tensor& rope_positions, ops::GqaExecutionEnvelope envelope,
                          Tensor& mtp_hidden);
    void mtp_forward_core(const Tensor& ids, const Tensor& hidden, const Tensor& positions,
                          const Tensor& rope_positions, ops::GqaExecutionEnvelope envelope,
                          Tensor& mtp_hidden);
    void mtp_prefill_chunk(const Tensor& ids, const Tensor& hidden, const Tensor* input_embeddings,
                           const Tensor& positions, const Tensor& rope_positions,
                           ops::GqaExecutionEnvelope envelope, bool final_chunk,
                           Tensor* final_hidden, Tensor* logits, Tensor* draft_token);
    void mtp_draft_argmax(const Tensor& hidden, Tensor& logits, Tensor& draft_token);

    struct MultimodalPrefill {
        std::span<const std::int32_t> positions;
        std::span<const std::int32_t> scatter_indices;
        const Tensor* embeddings = nullptr;
        std::int32_t rope_delta  = 0;
    };

    template <class Tap>
    void prefill_impl(std::span<const int> ids, const MultimodalPrefill* multimodal, Tap& tap);
    template <class Tap>
    void target_verify_impl(const Tensor& ids, const Tensor& positions,
                            ops::GqaExecutionEnvelope envelope, Tap& tap);

    DeviceContext& ctx_;
    const LoadedModelData& weights_;
    WorkspaceArena& work_;
    KVCache& kv_;
    KVCache* mtp_kv_;
    GdnState& state_;
    StepState& io_;
    std::uint32_t prefill_chunk_;
    std::uint32_t text_kv_base_;
    const Tensor* active_cache_positions_                 = nullptr;
    const Tensor* active_rope_positions_                  = nullptr;
    const ops::GqaExecutionEnvelope* active_gqa_envelope_ = nullptr;
    std::int32_t rope_delta_                              = 0;
    std::int32_t gdn_prefill_read_slot_                   = 0;
    std::int64_t prefill_snapshot_boundary_               = -1;
    Tensor* boundary_hidden_output_                       = nullptr;
    bool mtp_prompt_prepared_                             = false;

    const Weight* embed_                        = nullptr;
    const Tensor* final_norm_                   = nullptr;
    const Weight* lm_head_                      = nullptr;
    const Weight* lm_head_draft_                = nullptr;
    const std::int32_t* lm_head_draft_ids_      = nullptr;
    int lm_head_draft_n_                        = 0;
    const ops::SamplingConfig* sampling_config_ = nullptr;
    MtpW mtp_;
    std::array<FullLayerW, TextConfig::full_attention_layers()> full_{};
    std::array<GdnLayerW, TextConfig::gdn_layers()> gdn_{};
    std::array<Weight, TextConfig::gdn_layers()> gdn_in_a_{};
    std::array<Weight, TextConfig::gdn_layers()> gdn_in_b_{};
    std::array<Tensor, TextConfig::gdn_layers()> gdn_conv1d_views_{};
};

namespace detail {

struct VisionControl {
    std::vector<std::int32_t> position_ids;
    std::vector<std::int32_t> cu_seqlens;
    std::vector<std::int32_t> scatter_indices;
    std::vector<std::int32_t> pos_indices;
    std::vector<float> pos_weights;
};

VisionControl build_vision_control(const ProcessedInput& input);

} // namespace detail

} // namespace ninfer::targets::qwen3_6_27b_rtx5090::detail::schedule
