#pragma once

#include "models/common/weights.h"
#include "models/common/dflash/config.h"

namespace ninfer::models {
class Bindings;
}

namespace ninfer::models::dflash {

struct AttentionWeights {
    WeightId query, key, value, context_key, context_value;
    WeightId query_norm, key_norm, output;
};

struct DynamicConvWeights {
    WeightId base_kernel, kernel_projection;
};

struct BlockWeights {
    WeightId input_norm, post_attention_norm;
    AttentionWeights attention;
    DenseWeights mlp;
    std::optional<DynamicConvWeights> attention_conv, mlp_conv;
};

struct SelectorWeights {
    WeightId hidden_projection, predecessor_codebook, successor_codebook;
};

struct Weights {
    WeightId feature_projection, context_norm, final_norm;
    std::vector<BlockWeights> layers;
    std::optional<SelectorWeights> selector;
    WeightId token_embedding, output_head;
    WeightUseId output_head_use;
};

[[nodiscard]] Weights bind_weights(Bindings&, const Config&, TargetGeometry,
                                   const std::string& component);

} // namespace ninfer::models::dflash
