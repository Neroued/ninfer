#include "models/common/dflash/config.h"
#include "artifact/schema.h"

#include <cmath>
#include <limits>
#include <set>

namespace ninfer::models::dflash {
namespace {
using artifact::ArtifactError;
using artifact::Json;
using artifact::require_members;

std::uint32_t integer(const Json& value, std::string_view label, bool positive = true) {
    const auto n = artifact::require_u64(value, label, positive);
    if (n > std::numeric_limits<std::uint32_t>::max()) {
        throw ArtifactError(std::string(label) + ": config dimension exceeds u32");
    }
    return static_cast<std::uint32_t>(n);
}

std::uint32_t dimension(const Json& config, const char* name) {
    return integer(config.at(name), name);
}

float positive_float(const Json& config, const char* name) {
    const auto& value = config.at(name);
    if (!value.is_number()) { throw ArtifactError(std::string(name) + " must be a real value"); }
    const float result = value.get<float>();
    if (!std::isfinite(result) || result <= 0) {
        throw ArtifactError(std::string(name) + " must be positive finite FP32");
    }
    return result;
}

std::string architecture(const Json& config) {
    const auto& names = config.at("architectures");
    if (!names.is_array() || names.size() != 1) {
        throw ArtifactError("architectures must name one implementation");
    }
    return artifact::require_id(names[0], "architecture");
}

AttentionConfig attention(const Json& value) {
    AttentionConfig out{dimension(value, "num_attention_heads"),
                        dimension(value, "num_key_value_heads"), dimension(value, "head_dim")};
    if (out.num_attention_heads % out.num_key_value_heads) {
        throw ArtifactError("attention query heads must be divisible by KV heads");
    }
    return out;
}

} // namespace

Config parse_config(const artifact::Component& component, TargetGeometry target, bool dflash2) {
    if (component.target != "text") throw ArtifactError("draft component target must be text");
    const auto& value = component.config;
    require_members(value,
                    {"architectures", "model_type", "intermediate_size", "num_attention_heads",
                     "num_key_value_heads", "head_dim", "num_hidden_layers",
                     "max_position_embeddings", "rms_norm_eps", "rope_parameters", "layer_types",
                     "dflash_config"},
                    {"sliding_window"}, "draft config");
    if (architecture(value) != (dflash2 ? "DFlash2DraftModel" : "DFlashDraftModel") ||
        value.at("model_type") != "qwen3") {
        throw ArtifactError("draft architecture/model_type mismatch");
    }
    Config out;
    out.attention = attention(value);
    if (out.attention.head_dim % 2) { throw ArtifactError("draft head dimension must be even"); }
    out.intermediate_size       = dimension(value, "intermediate_size");
    out.num_hidden_layers       = dimension(value, "num_hidden_layers");
    out.max_position_embeddings = dimension(value, "max_position_embeddings");
    out.rms_norm_eps            = positive_float(value, "rms_norm_eps");
    require_members(value.at("rope_parameters"), {"rope_theta"}, {}, "draft RoPE");
    out.rope_theta     = positive_float(value.at("rope_parameters"), "rope_theta");
    const auto& layers = value.at("layer_types");
    if (!layers.is_array() || layers.size() != out.num_hidden_layers) {
        throw ArtifactError("draft layer_types length mismatch");
    }
    bool sliding = false;
    for (const auto& layer : layers) {
        if (layer == "full_attention") {
            out.layer_types.push_back(AttentionKind::FullAttention);
        } else if (layer == "sliding_attention") {
            out.layer_types.push_back(AttentionKind::SlidingAttention);
            sliding = true;
        } else {
            throw ArtifactError("unknown draft layer_type");
        }
    }
    if (sliding || value.contains("sliding_window")) {
        out.sliding_window = dimension(value, "sliding_window");
    }
    const auto& config = value.at("dflash_config");
    if (dflash2) {
        require_members(config,
                        {"target_layer_ids", "mask_token_id", "conv_kernel_size", "conv_group_size",
                         "selector_rank", "selector_top_k"},
                        {"block_size"}, "DFlash2 config");
    } else {
        require_members(config, {"target_layer_ids", "mask_token_id"}, {"block_size"},
                        "DFlash config");
    }
    if (config.contains("block_size")) out.training_block_size = dimension(config, "block_size");
    const auto& taps = config.at("target_layer_ids");
    if (!taps.is_array() || taps.empty()) {
        throw ArtifactError("target_layer_ids must be nonempty");
    }
    std::set<std::uint32_t> unique;
    for (const auto& tap : taps) {
        const auto index = integer(tap, "target layer index", false);
        if (index >= target.layers || !unique.insert(index).second) {
            throw ArtifactError("invalid target layer index");
        }
        out.target_layer_ids.push_back(index);
    }
    out.mask_token_id = integer(config.at("mask_token_id"), "mask token", false);
    if (out.mask_token_id >= target.vocabulary) {
        throw ArtifactError("mask token exceeds target embedding rows");
    }
    if (dflash2) {
        DFlash2Config extra{
            dimension(config, "conv_kernel_size"), dimension(config, "conv_group_size"),
            dimension(config, "selector_rank"), dimension(config, "selector_top_k")};
        if (target.hidden_size % extra.conv_group_size) {
            throw ArtifactError("DFlash2 conv group does not divide hidden width");
        }
        out.dflash2 = extra;
    }
    return out;
}

} // namespace ninfer::models::dflash
