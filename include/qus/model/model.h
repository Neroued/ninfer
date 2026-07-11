#pragma once

#include "qus/core/arena.h"
#include "qus/core/device.h"
#include "qus/core/kv_cache.h"
#include "qus/core/state_store.h"
#include "qus/core/tensor.h"
#include "qus/core/weight.h"
#include "qus/core/weight_store.h"
#include "qus/kernels/sampling.h"
#include "qus/model/config.h"
#include "qus/model/processor.h"

#include <array>
#include <cstddef>
#include <filesystem>
#include <span>

namespace qus::model {

inline constexpr const char* kWorkspaceLifetimePolicy = "block_scoped_mixer_mlp_rewind";
inline constexpr std::int32_t kStepStatsCounters      = 9;

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
    // Stable device address read by MTP position kernels captured in CUDA graphs.
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

struct NullTap {
    static constexpr bool enabled = false;
};

class FileTap {
public:
    static constexpr bool enabled = true;

    explicit FileTap(std::filesystem::path out_dir);

    void operator()(TapId id, int layer, Phase phase, const Tensor& x, cudaStream_t stream);

private:
    std::filesystem::path out_dir_;
};

using TapCallback = void (*)(void*, TapId, int, Phase, const Tensor&, cudaStream_t);

class Qwen3_6_27B {
public:
    Qwen3_6_27B(DeviceContext& ctx, WeightStore& weights, WorkspaceArena& work, KVCache& kv,
                GdnState& state, StepState& io, std::uint32_t prefill_chunk,
                KVCache* mtp_kv = nullptr);
    ~Qwen3_6_27B();

    Qwen3_6_27B(const Qwen3_6_27B&)            = delete;
    Qwen3_6_27B& operator=(const Qwen3_6_27B&) = delete;
    Qwen3_6_27B(Qwen3_6_27B&&)                 = delete;
    Qwen3_6_27B& operator=(Qwen3_6_27B&&)      = delete;

    [[nodiscard]] const Weight* embed() const noexcept { return embed_; }

    [[nodiscard]] const Tensor* final_norm() const noexcept { return final_norm_; }

    [[nodiscard]] const Weight* lm_head() const noexcept { return lm_head_; }

    // Optional embedded Q4 draft head (v4 packed file). When set, the MTP draft
    // argmax sites project with this smaller [n,5120] head and remap the shortlist
    // index back to a real vocab id via `ids`. Verify and every non-draft site keep
    // using the full `lm_head_`, so emitted tokens are unchanged; only per-round
    // acceptance (speed) is affected. Pass a null weight to force the baseline head.
    void set_lm_head_draft(const Weight* w, const std::int32_t* ids, int n) noexcept {
        lm_head_draft_     = w;
        lm_head_draft_ids_ = ids;
        lm_head_draft_n_   = n;
    }

    // Device-resident sampling config for the argmax/sample sites. When null the
    // decode/prefill token pick stays exact greedy argmax; when set, sample
    // reads it (temperature == 0 in the config is still exact greedy). The pointer
    // is stable across requests so a captured CUDA graph stays valid; only the
    // buffer contents change between requests.
    void set_sampling(const kernels::SamplingConfig* config) noexcept { sampling_config_ = config; }

    // Absolute token position at which the next prefill snapshots the running GDN conv/SSM state
    // into the dedicated turn-boundary snapshot slot (the last GdnState slot). -1 disables it. The
    // engine sets this per prefill so partial prefix reuse can continue the recurrence from the
    // assistant-content boundary next turn. Consumed and reset by prefill.
    void set_prefill_snapshot_boundary(std::int64_t abs_pos) noexcept {
        prefill_snapshot_boundary_ = abs_pos;
    }

    [[nodiscard]] const kernels::SamplingConfig* sampling_config() const noexcept {
        return sampling_config_;
    }

    [[nodiscard]] const Weight* lm_head_draft() const noexcept { return lm_head_draft_; }

    [[nodiscard]] const std::int32_t* lm_head_draft_ids() const noexcept {
        return lm_head_draft_ids_;
    }

    [[nodiscard]] int lm_head_draft_n() const noexcept { return lm_head_draft_n_; }

    [[nodiscard]] const FullLayerW& full_layer(std::size_t i) const { return full_.at(i); }

    [[nodiscard]] const GdnLayerW& gdn_layer(std::size_t i) const { return gdn_.at(i); }

    [[nodiscard]] bool mtp_enabled() const noexcept { return mtp_kv_ != nullptr; }

    [[nodiscard]] const MtpW& mtp_weights() const;

    void mtp_forward_batch(const Tensor& ids, const Tensor& hidden, const Tensor& positions,
                           Tensor& mtp_hidden, int logits_column, Tensor* logits,
                           Tensor* draft_token);

    void mtp_forward_ar_step(const Tensor& token, const Tensor& previous_hidden,
                             const Tensor& position, Tensor& mtp_hidden, Tensor& logits,
                             Tensor& draft_token);

    void mtp_sample_from_hidden_row(const Tensor& mtp_hidden, const Tensor& row, Tensor& out_hidden,
                                    Tensor& logits, Tensor& draft_token);

    void target_verify(const Tensor& ids, const Tensor& positions);

    void prefill(std::span<const int> ids);
    void prefill(const ProcessedInput& input, const Tensor& visual_embeddings);

    template <class Tap>
    void prefill(std::span<const int> ids, Tap& tap) {
        if constexpr (Tap::enabled) {
            prefill_erased(ids, &tap,
                           [](void* ctx, TapId id, int layer, Phase phase, const Tensor& x,
                              cudaStream_t stream) {
                               (*static_cast<Tap*>(ctx))(id, layer, phase, x, stream);
                           });
        } else {
            prefill(ids);
        }
    }

    void decode_step();
    void decode_step_record();

    template <class Tap>
    void decode_step(Tap& tap) {
        if constexpr (Tap::enabled) {
            decode_step_erased(&tap, [](void* ctx, TapId id, int layer, Phase phase,
                                        const Tensor& x, cudaStream_t stream) {
                (*static_cast<Tap*>(ctx))(id, layer, phase, x, stream);
            });
        } else {
            decode_step();
        }
    }

#ifdef QUS_MODEL_TESTING
    struct TestScheduleEntry {
        bool is_full = false;
        int index    = 0;
    };

    void test_attn_mix(const FullLayerW& w, Tensor& x, int fidx, Phase ph) {
        attn_mix(w, x, fidx, ph);
    }

    void test_gdn_mix(const GdnLayerW& w, Tensor& x, int gidx, Phase ph) {
        gdn_mix(w, x, gidx, ph);
    }

    void test_mlp_tail(const Tensor* post_norm, const MlpW& m, Tensor& x, Phase ph) {
        mlp_tail(post_norm, m, x, ph);
    }

    void test_run_layers(Tensor& x, Phase ph) { run_layers(x, ph); }

    [[nodiscard]] static constexpr TestScheduleEntry test_schedule_entry(int layer) {
        return TestScheduleEntry{
            ModelConfig::is_full(layer),
            ModelConfig::is_full(layer) ? ModelConfig::full_idx(layer)
                                        : ModelConfig::gdn_idx(layer),
        };
    }
#endif

private:
    void bind();
    void attn_mix(const FullLayerW& w, Tensor& x, int fidx, Phase ph);
    void gdn_mix(const GdnLayerW& w, Tensor& x, int gidx, Phase ph);
    void mlp_tail(const Tensor* post_norm, const MlpW& m, Tensor& x, Phase ph);
    void mtp_forward_stem(const Tensor& ids, const Tensor& hidden, Tensor& x, Tensor& ah);
    void mtp_forward_tail(Tensor& x, const Tensor& ah, const Tensor& positions,
                          const Tensor& rope_positions, Tensor& mtp_hidden);
    void mtp_forward_core(const Tensor& ids, const Tensor& hidden, const Tensor& positions,
                          const Tensor& rope_positions, Tensor& mtp_hidden);
    void mtp_prefill_chunk(const Tensor& ids, const Tensor& hidden, const Tensor& positions,
                           const Tensor& rope_positions, bool final_chunk, Tensor* final_hidden,
                           Tensor* logits, Tensor* draft_token);
    // Draft-proposal argmax at an MTP draft site. Uses the full `lm_head_` (writing
    // into `logits`) unless a draft head is set, in which case it projects with the
    // smaller `lm_head_draft_` into a work-scoped [n,1] buffer, argmaxes to a
    // shortlist index, and remaps it to a real vocab id. `draft_token` receives the
    // vocab id either way. Caller owns the surrounding work_ mark/rewind.
    void mtp_draft_argmax(const Tensor& hidden_col, Tensor& logits, Tensor& draft_token);
    void prefill_erased(std::span<const int> ids, void* tap, TapCallback callback);
    void decode_step_erased(void* tap, TapCallback callback);

    struct MultimodalPrefill {
        std::span<const std::int32_t> positions;
        std::span<const std::int32_t> scatter_indices;
        const Tensor* embeddings = nullptr;
        std::int32_t rope_delta  = 0;
    };

    template <class Tap>
    void prefill_impl(std::span<const int> ids, const MultimodalPrefill* multimodal, Tap& tap);
    template <class Tap>
    void decode_step_impl(Tap& tap);
    template <class Tap>
    void run_layers(Tensor& x, Phase ph, Tap& tap);
    void run_layers(Tensor& x, Phase ph);

    DeviceContext& ctx_;
    WeightStore& weights_;
    WorkspaceArena& work_;
    KVCache& kv_;
    KVCache* mtp_kv_ = nullptr;
    GdnState& state_;
    StepState& io_;
    std::uint32_t prefill_chunk_          = kDefaultPrefillChunk;
    const Tensor* active_cache_positions_ = nullptr;
    const Tensor* active_rope_positions_  = nullptr;
    std::int32_t rope_delta_              = 0;
    // GDN snapshot slot the current prefill chunk reads its initial recurrent/conv state from.
    // Slot 0 for the reset path and for every chunk after the first; the committed snapshot slot
    // for chunk 0 of a prefix-append prefill (MTP). The running state is always written to slot 0.
    std::int32_t gdn_prefill_read_slot_ = 0;
    // Absolute position at which the current prefill snapshots the running GDN state into the
    // turn-boundary slot; -1 disables. Set via set_prefill_snapshot_boundary and reset by prefill.
    std::int64_t prefill_snapshot_boundary_ = -1;

    const Weight* embed_                            = nullptr;
    const Tensor* final_norm_                       = nullptr;
    const Weight* lm_head_                          = nullptr;
    const Weight* lm_head_draft_                    = nullptr;
    const std::int32_t* lm_head_draft_ids_          = nullptr;
    int lm_head_draft_n_                            = 0;
    const kernels::SamplingConfig* sampling_config_ = nullptr;
    MtpW mtp_{};
    std::array<FullLayerW, ModelConfig::n_full()> full_{};
    std::array<GdnLayerW, ModelConfig::n_gdn()> gdn_{};
    std::array<Weight, ModelConfig::n_gdn()> gdn_in_a_{};
    std::array<Weight, ModelConfig::n_gdn()> gdn_in_b_{};
    std::array<Tensor, ModelConfig::n_gdn()> gdn_conv1d_views_{};
};

} // namespace qus::model
