#pragma once

#include <ninfer/targets/qwen3_6_35b_a3b/package.h>
#include <ninfer/targets/qwen3_6/frontend_resources.h>
#include <ninfer/targets/qwen3_6/model_view.h>
#include <ninfer/targets/qwen3_6/startup_features.h>
#include <ninfer/targets/qwen3_6/vision.h>

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "core/tensor.h"
#include "ninfer/ops/sparse_moe.h"

#include <array>
#include <cstddef>
#include <utility>

namespace ninfer::targets::qwen3_6_35b_a3b::detail {

inline constexpr std::size_t kTextLayers          = 40;
inline constexpr std::size_t kFullAttentionLayers = 10;
inline constexpr std::size_t kGdnLayers           = 30;

struct MoePlan {
    artifact::ObjectHandle router_shared_gate;
    artifact::ObjectHandle routed_gate_up;
    artifact::ObjectHandle routed_down;
    artifact::ObjectHandle shared_gate_up;
    artifact::ObjectHandle shared_down;
};

struct FullAttentionPlan {
    artifact::ObjectHandle query_key_gate_value;
    artifact::ObjectHandle query_norm;
    artifact::ObjectHandle key_norm;
    artifact::ObjectHandle output;
};

struct GdnPlan {
    artifact::ObjectHandle a_log;
    artifact::ObjectHandle dt_bias;
    artifact::ObjectHandle convolution;
    artifact::ObjectHandle a_b_projection;
    artifact::ObjectHandle query_key_value_z;
    artifact::ObjectHandle norm;
    artifact::ObjectHandle output;
};

struct TextLayerPlan {
    artifact::ObjectHandle input_norm;
    FullAttentionPlan attention{};
    GdnPlan gdn{};
    bool is_full_attention = false;
    artifact::ObjectHandle post_attention_norm;
    MoePlan moe;
};

struct MtpPlan {
    artifact::ObjectHandle input_projection;
    artifact::ObjectHandle embedding_norm;
    artifact::ObjectHandle hidden_norm;
    artifact::ObjectHandle input_norm;
    FullAttentionPlan attention;
    artifact::ObjectHandle post_attention_norm;
    MoePlan moe;
    artifact::ObjectHandle final_norm;
};

struct BindingPlan {
    qwen3_6::FrontendResourcePlan frontend;
    qwen3_6::StartupFeatures features;
    artifact::ObjectHandle token_embedding;
    std::array<TextLayerPlan, kTextLayers> text_layers;
    artifact::ObjectHandle final_norm;
    artifact::ObjectHandle output_head;
    artifact::ObjectHandle draft_head;
    artifact::ObjectHandle draft_head_token_ids;
    MtpPlan mtp;
    qwen3_6::VisionBackbonePlan vision_backbone;
    qwen3_6::VisionMergerInputPlan vision_merger_input;
    artifact::ObjectHandle vision_merger_fc2;
    artifact::ObjectHandle vision_merger_fc2_bias;
    qwen3_6::VisionMergerNormPlan vision_merger_norm;
};

struct ArtifactLoadPlan {
    BindingPlan bindings;
    artifact::MaterializationPlan materialization;
};

ArtifactLoadPlan bind_artifact(artifact::Binder& binder, qwen3_6::StartupFeatures features);

struct SparseMoePayload {
    ops::SparseMoeWeights op;
};

struct AttentionProjectionPayload {
    Weight query_key_gate_value;
};

struct GdnProjectionPayload {
    Tensor a_log;
    Tensor dt_bias;
    Weight a_b_projection;
    Weight query_key_value_z;
};

using RuntimeModelView     = qwen3_6::ModelView<AttentionProjectionPayload, GdnProjectionPayload,
                                                SparseMoePayload, AttentionProjectionPayload,
                                                SparseMoePayload, kFullAttentionLayers, kGdnLayers>;
using FullAttentionWeights = RuntimeModelView::FullLayer;
using GdnWeights           = RuntimeModelView::GdnLayer;
using MtpWeights           = RuntimeModelView::MtpLayer;

class LoadedModelData {
public:
    LoadedModelData(BindingPlan plan, artifact::MaterializedArtifact materialized);

    LoadedModelData(const LoadedModelData&)            = delete;
    LoadedModelData& operator=(const LoadedModelData&) = delete;
    LoadedModelData(LoadedModelData&&)                 = delete;
    LoadedModelData& operator=(LoadedModelData&&)      = delete;

    artifact::MaterializedArtifact backing;
    qwen3_6::FrontendResources frontend;
    RuntimeModelView runtime;
};

class LoadedModel::Impl {
public:
    Impl(BindingPlan plan, artifact::MaterializedArtifact materialized)
        : data(std::move(plan), std::move(materialized)) {}

    LoadedModelData data;
};

} // namespace ninfer::targets::qwen3_6_35b_a3b::detail
