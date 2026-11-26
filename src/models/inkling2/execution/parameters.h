#pragma once

#include "models/inkling2/execution/profile.h"
#include "models/inkling2/model.h"
#include "ninfer/ops/loop_message.h"
#include "ninfer/ops/moe.h"
#include "ninfer/ops/projection_weights.h"

#include <memory>
#include <optional>
#include <vector>

namespace ninfer::models::inkling2::execution {

using LinearParameters = ops::SingleProjectionWeight;

struct AttentionPassParameters {
    std::optional<ops::ProjectionWeights> qkv_split;
    ops::ProjectionWeights qrg_split;
    LinearParameters key;
    LinearParameters value;
    std::optional<LinearParameters> k_conv_proj;
    std::optional<LinearParameters> v_conv_proj;
    LinearParameters output;
};

struct LoopParameters {
    Tensor message_queries; // FP32 [H, 4]
    ops::LoopMessageWeights attn_kv;
    ops::LoopMessageWeights mlp;
};

struct MoePassParameters {
    Weight router;
    ops::CudaSwiGLUExperts routed;
    ops::ExpertSubset subset;
};

struct PhysicalLayerParameters {
    bool is_swa  = false;
    bool in_core = false;
    Tensor attn_res_query;  // FP32 [H, 1] (SWA) or [H, 3] (Full); empty on layer 0
    Tensor attn_input_norm; // BF16 [H, 1] (SWA) or [H, 3] (Full)
    Tensor sink;            // BF16 [H, 24]
    Tensor q_scale;         // BF16 [128, 24]
    Tensor r_scale;         // BF16 [16, 24]
    Tensor rel_logits_proj; // BF16 [16, 1024]
    Tensor k_conv;          // BF16 [1024, 4]
    Tensor v_conv;          // BF16 [1024, 4]
    Tensor attn_short_conv; // BF16 [H, 4]

    std::optional<LoopParameters> loop;

    Tensor mlp_attn_res_query; // FP32 [H, 1]
    Tensor mlp_input_norm;     // BF16 [H, 1]
    Tensor mlp_short_conv;     // BF16 [H, 4]
    Tensor correction_bias;    // FP32 [E]

    std::vector<AttentionPassParameters> attention_passes;
    std::vector<MoePassParameters> moe_passes;
};

struct TextParameters {
    Weight embedding;
    Tensor embed_norm;   // BF16 [H]
    Tensor final_norm;   // BF16 [H, 1]
    Tensor output_query; // FP32 [H, 1]
    LinearParameters head;
    std::vector<PhysicalLayerParameters> layers;
};

struct VisionLayerParameters {
    Tensor ln_1_weight, ln_1_bias, ln_2_weight, ln_2_bias;
    Tensor ls_1_gamma, ls_2_gamma;
    LinearParameters qkv, output, fc1, fc2;
    Tensor qkv_bias, output_bias, fc1_bias, fc2_bias;
};

struct VisionParameters {
    LinearParameters patch, projector_in, projector_out;
    Tensor position_embedding, class_embedding, cls_position_embedding;
    Tensor pre_norm_weight, pre_norm_bias;
    std::optional<Tensor> projector_in_bias, projector_out_bias;
    std::vector<VisionLayerParameters> layers;
};

struct Parameters {
    Parameters(const ModelShard&, std::shared_ptr<const ModelProfile>);

    const ModelShard& model;
    std::shared_ptr<const ModelProfile> profile;
    TextParameters text;
    std::optional<VisionParameters> vision;
};

} // namespace ninfer::models::inkling2::execution
