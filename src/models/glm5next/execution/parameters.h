#pragma once

#include "models/glm5next/execution/profile.h"
#include "models/glm5next/model.h"

namespace ninfer::models::glm5next::execution {

using LinearParameters = ops::SingleProjectionWeight;

struct KdaParameters {
    LinearParameters qkv, control, forget, gate, output;
    Tensor convolution, a_log, dt_bias, norm;
};

struct MlaParameters {
    std::array<ops::KpoolMlaPrepareParameters, 3> prepare;
    ops::ProjectionWeights query_index;
    LinearParameters query, output;
    ops::KpoolMlaFactors factors;
};

struct MoeParameters {
    ops::SigmoidMoeWeights weights;
    ops::ExpertSubset subset;
};

struct BlockParameters {
    ops::MhcPreWeights attention_hc, mlp_hc;
    std::variant<KdaParameters, MlaParameters> mixer;
    std::variant<ops::SwiGLUWeights, MoeParameters> ffn;
};

struct TextParameters {
    Weight embedding;
    LinearParameters head;
    Tensor final_norm;
    std::vector<BlockParameters> layers;
};

struct MtpParameters {
    Weight embedding;
    LinearParameters stem, head;
    Tensor embedding_norm, hidden_norm, input_norm, post_attention_norm, final_norm;
    MlaParameters attention;
    MoeParameters moe;
};

struct DynamicConvParameters {
    Tensor base;
    LinearParameters projection;
};

struct DraftBlockParameters {
    Tensor input_norm, post_attention_norm, query_norm, key_norm;
    LinearParameters kvq, output, gate_up, down;
    Weight context_key, context_value;
    DynamicConvParameters attention_conv, mlp_conv;
};

struct DraftParameters {
    Weight embedding;
    LinearParameters feature_projection, head, selector;
    Tensor context_norm, final_norm, predecessor_codebook, successor_codebook;
    std::vector<DraftBlockParameters> layers;
};

struct VisionBlockParameters {
    Tensor input_norm, post_attention_norm, query_norm, key_norm;
    LinearParameters qkv, output, gate_up, down;
    Tensor qkv_bias, output_bias, gate_up_bias, down_bias;
};

struct VisionParameters {
    LinearParameters patch, downsample, merger_projection, merger_gate_up, merger_down;
    Tensor patch_bias, downsample_bias, post_norm, merger_norm_weight, merger_norm_bias;
    std::vector<VisionBlockParameters> layers;
};

class Parameters {
public:
    explicit Parameters(const ModelShard&, std::shared_ptr<const ModelProfile>);
    Parameters(const Parameters&)            = delete;
    Parameters& operator=(const Parameters&) = delete;

    const ModelShard& model;
    const std::shared_ptr<const ModelProfile> profile;
    TextParameters text;
    std::optional<MtpParameters> mtp;
    std::optional<DraftParameters> draft;
    std::optional<VisionParameters> vision;
};

} // namespace ninfer::models::glm5next::execution
