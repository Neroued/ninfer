#pragma once

#include "qus/core/arena.h"
#include "qus/core/device.h"
#include "qus/core/kv_cache.h"
#include "qus/core/state_store.h"
#include "qus/core/tensor.h"
#include "qus/core/weight.h"
#include "qus/core/weight_store.h"
#include "qus/model/config.h"

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

    [[nodiscard]] const FullLayerW& full_layer(std::size_t i) const { return full_.at(i); }

    [[nodiscard]] const GdnLayerW& gdn_layer(std::size_t i) const { return gdn_.at(i); }

    [[nodiscard]] bool mtp_enabled() const noexcept { return mtp_kv_ != nullptr; }

    [[nodiscard]] const MtpW& mtp_weights() const;

    void mtp_set_cache_position(std::uint32_t position);

    void mtp_forward_batch(const Tensor& ids, const Tensor& hidden, const Tensor& positions,
                           std::uint32_t cache_offset, Tensor& mtp_hidden, int logits_column,
                           Tensor* logits, Tensor* draft_token);

    void mtp_forward_ar_step(const Tensor& token, const Tensor& previous_hidden,
                             const Tensor& position, Tensor& mtp_hidden, Tensor& logits,
                             Tensor& draft_token);

    void mtp_sample_from_hidden_row(const Tensor& mtp_hidden, const Tensor& row, Tensor& out_hidden,
                                    Tensor& logits, Tensor& draft_token);

    void target_verify(const Tensor& ids, const Tensor& positions, std::uint32_t cache_offset);

    void prefill(std::span<const int> ids);

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
    void mtp_forward_core(const Tensor& ids, const Tensor& hidden, const Tensor& positions,
                          Tensor& mtp_hidden);
    void prefill_erased(std::span<const int> ids, void* tap, TapCallback callback);
    void decode_step_erased(void* tap, TapCallback callback);
    template <class Tap>
    void prefill_impl(std::span<const int> ids, Tap& tap);
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
    std::uint32_t prefill_chunk_    = kDefaultPrefillChunk;
    const Tensor* active_positions_ = nullptr;

    const Weight* embed_      = nullptr;
    const Tensor* final_norm_ = nullptr;
    const Weight* lm_head_    = nullptr;
    MtpW mtp_{};
    std::array<FullLayerW, ModelConfig::n_full()> full_{};
    std::array<GdnLayerW, ModelConfig::n_gdn()> gdn_{};
    std::array<Weight, ModelConfig::n_gdn()> gdn_in_a_{};
    std::array<Weight, ModelConfig::n_gdn()> gdn_in_b_{};
    std::array<Tensor, ModelConfig::n_gdn()> gdn_conv1d_views_{};
};

} // namespace qus::model
