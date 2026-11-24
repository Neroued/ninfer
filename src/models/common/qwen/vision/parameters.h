#pragma once

#include "models/common/weights.h"
#include "ninfer/ops/weight_input.h"

#include <array>
#include <vector>

namespace ninfer::models {
class Bindings;
}

namespace ninfer::models::qwen {

struct VisionConfig {
    std::uint32_t depth                   = 0;
    std::uint32_t hidden_size             = 0;
    std::uint32_t intermediate_size       = 0;
    std::uint32_t num_heads               = 0;
    std::uint32_t patch_size              = 0;
    std::uint32_t temporal_patch_size     = 0;
    std::uint32_t spatial_merge_size      = 0;
    std::uint32_t num_position_embeddings = 0;
    std::uint32_t position_grid_side      = 0;
    [[nodiscard]] std::uint64_t patch_width() const;
    [[nodiscard]] std::uint64_t merger_width() const;
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

struct NormParameters {
    Tensor weight, bias;
};

struct VisionBlockParameters {
    NormParameters norm1, norm2;
    ops::SingleProjectionWeight qkv;
    Tensor qkv_bias;
    ops::SingleProjectionWeight output, fc1, fc2;
    Tensor output_bias, fc1_bias, fc2_bias;
};

struct VisionParameters {
    ops::SingleProjectionWeight patch_embedding;
    Tensor patch_embedding_bias, position_embedding;
    std::vector<VisionBlockParameters> layers;
    NormParameters merger_norm;
    ops::SingleProjectionWeight merger_fc1, merger_fc2;
    Tensor merger_fc1_bias, merger_fc2_bias;
};

struct VisionBlockProfile {
    ops::LinearProfile qkv, output, fc1, fc2;
};

struct VisionProfile {
    ops::LinearProfile patch_embedding, merger_fc1, merger_fc2;
    std::vector<VisionBlockProfile> layers;
};

[[nodiscard]] VisionWeights bind_vision(Bindings&, const VisionConfig&,
                                        std::uint32_t output_hidden);
[[nodiscard]] VisionParameters prepare_vision(std::span<const BoundWeight>, const VisionWeights&);

template <class Input>
[[nodiscard]] VisionProfile describe_vision(const VisionWeights& weights, Input&& input) {
    const auto linear = [&](WeightId id) {
        const auto value = input(id);
        return ops::describe_linear_weight({&value, 1});
    };
    VisionProfile result{linear(weights.patch_embedding),
                         linear(weights.merger_fc1),
                         linear(weights.merger_fc2),
                         {}};
    for (const auto& layer : weights.layers) {
        const std::array inputs{input(layer.query), input(layer.key), input(layer.value)};
        result.layers.push_back({ops::describe_linear_weight(inputs), linear(layer.output),
                                 linear(layer.fc1), linear(layer.fc2)});
    }
    return result;
}

} // namespace ninfer::models::qwen
