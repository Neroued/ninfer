#pragma once

#include "core/weight_bank.h"
#include "models/common/weights.h"

#include <optional>
#include <variant>

namespace ninfer::models::glm5next {

struct MhcWeights {
    WeightId projection, base, scale;
};

struct KdaWeights {
    WeightId query, key, value, output;
    WeightId beta, forget_a, forget_b, gate_a, gate_b;
    WeightId convolution, a_log, dt_bias, norm;
};

struct MlaWeights {
    WeightId latent, index_key, pool_gate, query_a, index_weight;
    WeightId query_b, index_query, key_up, value_up, output;
    WeightId query_norm, latent_norm, index_norm, index_bias, pool_ape;
};

struct DenseWeights {
    WeightId gate, up, down;
};

struct MoeWeights {
    WeightId router, correction_bias;
    DenseWeights shared;
    WeightId gate, up, down;
    ExpertBankId bank;
};

struct BlockWeights {
    WeightId input_norm, post_attention_norm;
    MhcWeights attention_hc, mlp_hc;
    std::variant<KdaWeights, MlaWeights> mixer;
    std::variant<DenseWeights, MoeWeights> ffn;
};

struct TextWeights {
    WeightId embedding, head, final_norm;
    std::vector<BlockWeights> layers;
};

struct MtpWeights {
    WeightId embedding, head, embedding_norm, hidden_norm, fc, final_norm;
    WeightId input_norm, post_attention_norm;
    MlaWeights attention;
    MoeWeights moe;
};

struct VisionBlockWeights {
    WeightId input_norm, post_attention_norm;
    WeightId query, key, value, query_bias, key_bias, value_bias;
    WeightId query_norm, key_norm, output, output_bias;
    DenseWeights mlp;
    WeightId gate_bias, up_bias, down_bias;
};

struct VisionWeights {
    WeightId patch_embedding, patch_embedding_bias, post_norm;
    std::vector<VisionBlockWeights> layers;
    WeightId downsample, downsample_bias;
    WeightId merger_projection, merger_norm_weight, merger_norm_bias;
    DenseWeights merger;
};

struct ModelWeights {
    TextWeights text;
    std::optional<MtpWeights> mtp;
    std::optional<VisionWeights> vision;
};

} // namespace ninfer::models::glm5next
