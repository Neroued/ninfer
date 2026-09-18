#pragma once

#include "core/weight_view.h"
#include "ninfer/ops/linear.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace ninfer::models::qwen3_5 {

struct WeightId {
    std::size_t index                          = std::numeric_limits<std::size_t>::max();
    friend bool operator==(WeightId, WeightId) = default;
};

struct WeightUseId {
    WeightId parameter;
    std::size_t use_index = std::numeric_limits<std::size_t>::max();
};

struct WeightUse {
    std::string input;
    ops::LinearPolicy policy = ops::LinearPolicy::A16Only;
    std::optional<float> activation_input_divisor;
};

struct BoundWeight {
    std::string name;
    std::vector<std::string> source_objects;
    WeightView view;
    std::vector<WeightUse> uses;
};

struct AttentionWeights {
    WeightId query, key, gate, value;
    WeightId query_norm, key_norm, output;
};

struct GdnWeights {
    WeightId query, key, value, z;
    WeightId a_projection, b_projection, a_log, dt_bias;
    WeightId convolution, norm, output;
};

struct DenseWeights {
    WeightId gate, up, down;
};

struct MoeWeights {
    WeightId router, shared_score;
    std::vector<DenseWeights> experts;
    DenseWeights shared;
};

struct BlockWeights {
    WeightId input_norm, post_attention_norm;
    std::variant<AttentionWeights, GdnWeights> mixer;
    std::variant<DenseWeights, MoeWeights> ffn;
};

struct TextWeights {
    WeightId token_embedding, output_head, final_norm;
    WeightUseId output_head_use;
    std::vector<BlockWeights> layers;
};

struct MtpWeights {
    WeightId input_projection, embedding_norm, hidden_norm, final_norm;
    BlockWeights layer;
    WeightId token_embedding, output_head;
    WeightUseId output_head_use;
};

struct NormWeights {
    WeightId weight, bias;
};

struct VisionBlockWeights {
    NormWeights norm1, norm2;
    WeightId query, key, value, query_bias, key_bias, value_bias;
    WeightId output, output_bias;
    WeightId fc1, fc1_bias, fc2, fc2_bias;
};

struct VisionWeights {
    WeightId patch_embedding, patch_embedding_bias, position_embedding;
    std::vector<VisionBlockWeights> layers;
    NormWeights merger_norm;
    WeightId merger_fc1, merger_fc1_bias, merger_fc2, merger_fc2_bias;
};

struct DraftAttentionWeights {
    WeightId query, key, value, context_key, context_value;
    WeightId query_norm, key_norm, output;
};

struct DynamicConvWeights {
    WeightId base_kernel, kernel_projection;
};

struct DraftBlockWeights {
    WeightId input_norm, post_attention_norm;
    DraftAttentionWeights attention;
    DenseWeights mlp;
    std::optional<DynamicConvWeights> attention_conv, mlp_conv;
};

struct SelectorWeights {
    WeightId hidden_projection, predecessor_codebook, successor_codebook;
};

struct DraftWeights {
    WeightId feature_projection, context_norm, final_norm;
    std::vector<DraftBlockWeights> layers;
    std::optional<SelectorWeights> selector;
    WeightId token_embedding, output_head;
    WeightUseId output_head_use;
};

struct ProposalWeights {
    WeightId head;
    std::optional<WeightId> token_ids;
    std::uint32_t rows = 0;
    std::vector<std::int32_t> global_token_ids;
};

// Handles refer to the frozen model's weight array. No artifact ID lookup is needed in execution.
struct ModelWeights {
    TextWeights text;
    std::optional<VisionWeights> vision;
    std::optional<MtpWeights> mtp;
    std::optional<DraftWeights> draft;
    std::optional<ProposalWeights> proposal;
};

} // namespace ninfer::models::qwen3_5
