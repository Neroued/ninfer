#pragma once

#include "core/weight_bank.h"
#include "models/common/weights.h"

#include <optional>
#include <vector>

namespace ninfer::models::inkling2 {

struct AttentionWeights {
    std::optional<WeightId> attn_res_query;
    WeightId input_norm;
    WeightId query, relative, output_gate;
    WeightId key, value;
    std::optional<WeightId> k_conv_proj, v_conv_proj;
    WeightId sink;
    WeightId output;
    WeightId q_scale, r_scale, rel_logits_proj;
    WeightId k_conv, v_conv, short_conv;
};

struct LoopWeights {
    WeightId message_queries;
    WeightId attn_kv_down, attn_kv_up;
    WeightId mlp_down, mlp_up;
};

struct MoeWeights {
    WeightId attn_res_query;
    WeightId input_norm;
    WeightId short_conv;
    WeightId router, correction_bias;
    WeightId gate, up, down;
    ExpertBankId bank;
};

struct PhysicalLayerWeights {
    AttentionWeights attention;
    std::optional<LoopWeights> loop;
    MoeWeights moe;
};

struct TextWeights {
    WeightId embedding, embed_norm, final_norm, output_query, head;
    std::vector<PhysicalLayerWeights> layers;
};

struct VisionBlockWeights {
    WeightId ln_1_weight, ln_1_bias, ln_2_weight, ln_2_bias;
    WeightId ls_1_gamma, ls_2_gamma;
    WeightId query, key, value;
    WeightId query_bias, key_bias, value_bias;
    WeightId output, output_bias;
    WeightId fc1, fc1_bias, fc2, fc2_bias;
};

struct VisionWeights {
    WeightId patch_embedding, position_embedding, class_embedding, cls_position_embedding;
    WeightId pre_norm_weight, pre_norm_bias;
    std::vector<VisionBlockWeights> layers;
    WeightId projector_in, projector_out;
    std::optional<WeightId> projector_in_bias, projector_out_bias;
};

struct DSparkLayerWeights {
    WeightId dflare_norm, input_norm, post_attention_norm;
    WeightId query, key, value;
    WeightId context_key, context_value;
    WeightId query_norm, key_norm, context_key_norm;
    WeightId output;
    DenseWeights mlp;
};

struct DSparkMarkovWeights {
    WeightId dflare_norm, w1_embed;
    WeightId w_recurrent, w_hidden, w_frnn, rnn_bias;
    WeightId fold_weight, fold_bias;
};

struct DSparkWeights {
    WeightId embedding, embed_norm, output_head;
    WeightId fuse_alpha, mask_embed, final_norm;
    std::vector<DSparkLayerWeights> layers;
    DSparkMarkovWeights markov;
};

struct ModelWeights {
    TextWeights text;
    std::optional<VisionWeights> vision;
    std::optional<DSparkWeights> dspark;
};

} // namespace ninfer::models::inkling2
