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
#include <span>

namespace qus::model {

struct MlpW {
    const Weight* gate = nullptr;
    const Weight* up   = nullptr;
    const Weight* down = nullptr;
};

struct FullLayerW {
    const Tensor* input_norm     = nullptr;
    const Weight* q_proj         = nullptr;
    const Weight* gate_proj      = nullptr;
    const Weight* k_proj         = nullptr;
    const Weight* v_proj         = nullptr;
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

struct StepState {
    Tensor token;
    Tensor pos;
    Tensor logits;
};

struct NullTap {};

enum class Phase {
    Prefill,
    Decode,
};

class Qwen3_6_27B {
public:
    Qwen3_6_27B(DeviceContext& ctx, WeightStore& weights, WorkspaceArena& work, KVCache& kv,
                GdnState& state, StepState& io);

    [[nodiscard]] const Weight* embed() const noexcept { return embed_; }

    [[nodiscard]] const Tensor* final_norm() const noexcept { return final_norm_; }

    [[nodiscard]] const Weight* lm_head() const noexcept { return lm_head_; }

    [[nodiscard]] const FullLayerW& full_layer(std::size_t i) const { return full_.at(i); }

    [[nodiscard]] const GdnLayerW& gdn_layer(std::size_t i) const { return gdn_.at(i); }

    void prefill(std::span<const int> ids);
    void decode_step();

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
    void run_layers(Tensor& x, Phase ph);

    DeviceContext& ctx_;
    WeightStore& weights_;
    WorkspaceArena& work_;
    KVCache& kv_;
    GdnState& state_;
    StepState& io_;
    const Tensor* active_positions_ = nullptr;
    int pos_upload_                 = 0;

    const Weight* embed_      = nullptr;
    const Tensor* final_norm_ = nullptr;
    const Weight* lm_head_    = nullptr;
    std::array<FullLayerW, ModelConfig::n_full()> full_{};
    std::array<GdnLayerW, ModelConfig::n_gdn()> gdn_{};
    std::array<Weight, ModelConfig::n_gdn()> gdn_in_a_{};
    std::array<Weight, ModelConfig::n_gdn()> gdn_in_b_{};
    std::array<Tensor, ModelConfig::n_gdn()> gdn_conv1d_views_{};
};

} // namespace qus::model
