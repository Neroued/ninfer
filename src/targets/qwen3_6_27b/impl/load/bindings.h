#pragma once

#include <ninfer/targets/qwen3_6_27b/package.h>
#include <ninfer/targets/qwen3_6/frontend_resources.h>
#include <ninfer/targets/qwen3_6/model_view.h>
#include <ninfer/targets/qwen3_6/startup_features.h>
#include <ninfer/targets/qwen3_6/vision.h>

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "core/tensor.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace ninfer::targets::qwen3_6_27b::detail {

inline constexpr std::size_t kTextLayers          = 64;
inline constexpr std::size_t kFullAttentionLayers = 16;
inline constexpr std::size_t kGdnLayers           = 48;

struct MlpPlan {
    artifact::ObjectHandle gate_up;
    artifact::ObjectHandle down;
};

struct FullAttentionPlan {
    artifact::ObjectHandle query_key;
    artifact::ObjectHandle gate_value;
    artifact::ObjectHandle query_norm;
    artifact::ObjectHandle key_norm;
    artifact::ObjectHandle output;
};

struct GdnPlan {
    artifact::ObjectHandle a_log;
    artifact::ObjectHandle dt_bias;
    artifact::ObjectHandle convolution;
    artifact::ObjectHandle a_projection;
    artifact::ObjectHandle b_projection;
    artifact::ObjectHandle query_key;
    artifact::ObjectHandle value_z;
    artifact::ObjectHandle norm;
    artifact::ObjectHandle output;
};

struct TextLayerPlan {
    artifact::ObjectHandle input_norm;
    FullAttentionPlan attention{};
    GdnPlan gdn{};
    bool is_full_attention = false;
    artifact::ObjectHandle post_attention_norm;
    MlpPlan mlp;
};

struct MtpPlan {
    artifact::ObjectHandle input_projection;
    artifact::ObjectHandle embedding_norm;
    artifact::ObjectHandle hidden_norm;
    artifact::ObjectHandle input_norm;
    artifact::ObjectHandle query_key_gate_value;
    artifact::ObjectHandle query_norm;
    artifact::ObjectHandle key_norm;
    artifact::ObjectHandle output;
    artifact::ObjectHandle post_attention_norm;
    MlpPlan mlp;
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

struct DensePostMixerPayload {
    Weight gate_up;
    Weight down;
};

struct FullAttentionProjectionPayload {
    Weight query_key;
    Weight gate_value;
};

struct GdnProjectionPayload {
    Tensor a_log;
    Tensor dt_bias;
    Weight a_projection;
    Weight b_projection;
    Weight query_key;
    Weight value;
    Weight z;
};

struct MtpAttentionPayload {
    Weight packed;
    Weight query;
    Weight key;
    Weight output_gate;
    Weight value;
};

using RuntimeModelView =
    qwen3_6::ModelView<FullAttentionProjectionPayload, GdnProjectionPayload, DensePostMixerPayload,
                       MtpAttentionPayload, DensePostMixerPayload, kFullAttentionLayers,
                       kGdnLayers>;
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

} // namespace ninfer::targets::qwen3_6_27b::detail
