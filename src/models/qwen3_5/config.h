#pragma once

#include "models/load_options.h"
#include "models/registry.h"

#include <array>
#include <algorithm>
#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

namespace ninfer::artifact {
struct Directory;
}

namespace ninfer::models::qwen3_5 {

enum class MixerKind { FullAttention, LinearAttention };
enum class DraftAttentionKind { FullAttention, SlidingAttention };

struct AttentionConfig {
    std::uint32_t num_attention_heads = 0;
    std::uint32_t num_key_value_heads = 0;
    std::uint32_t head_dim            = 0;

    [[nodiscard]] std::uint64_t query_width() const noexcept {
        return std::uint64_t(num_attention_heads) * head_dim;
    }

    [[nodiscard]] std::uint64_t key_width() const noexcept {
        return std::uint64_t(num_key_value_heads) * head_dim;
    }
};

struct RopeConfig {
    float rope_theta            = 0;
    float partial_rotary_factor = 0;
    std::array<std::uint32_t, 3> mrope_section{};
    std::uint32_t rotary_dim = 0;
    std::vector<std::uint8_t> pair_axes;
};

struct GdnConfig {
    std::uint32_t linear_num_key_heads   = 0;
    std::uint32_t linear_key_head_dim    = 0;
    std::uint32_t linear_num_value_heads = 0;
    std::uint32_t linear_value_head_dim  = 0;
    std::uint32_t linear_conv_kernel_dim = 0;

    [[nodiscard]] std::uint64_t key_width() const noexcept {
        return std::uint64_t(linear_num_key_heads) * linear_key_head_dim;
    }

    [[nodiscard]] std::uint64_t value_width() const noexcept {
        return std::uint64_t(linear_num_value_heads) * linear_value_head_dim;
    }

    [[nodiscard]] std::uint64_t conv_channels() const;
};

struct DenseConfig {
    std::uint32_t intermediate_size = 0;
};

struct MoeConfig {
    std::uint32_t num_experts                     = 0;
    std::uint32_t num_experts_per_tok             = 0;
    std::uint32_t moe_intermediate_size           = 0;
    std::uint32_t shared_expert_intermediate_size = 0;
};

struct TextConfig {
    Architecture architecture             = Architecture::Qwen3_5;
    std::uint32_t hidden_size             = 0;
    std::uint32_t vocab_size              = 0;
    std::uint32_t num_hidden_layers       = 0;
    std::uint32_t max_position_embeddings = 0;
    bool tie_word_embeddings              = false;
    float rms_norm_eps                    = 0;
    std::vector<MixerKind> layer_types;
    std::vector<std::uint32_t> compact_layer_indices;
    std::uint32_t full_attention_layers   = 0;
    std::uint32_t linear_attention_layers = 0;
    std::optional<AttentionConfig> attention;
    std::optional<RopeConfig> rope_parameters;
    std::optional<GdnConfig> gdn;
    std::variant<DenseConfig, MoeConfig> ffn;
};

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

struct DFlash2Config {
    std::uint32_t conv_kernel_size = 0;
    std::uint32_t conv_group_size  = 0;
    std::uint32_t selector_rank    = 0;
    std::uint32_t selector_top_k   = 0;
};

struct DraftConfig {
    AttentionConfig attention;
    std::uint32_t intermediate_size       = 0;
    std::uint32_t num_hidden_layers       = 0;
    std::uint32_t max_position_embeddings = 0;
    float rms_norm_eps                    = 0;
    float rope_theta                      = 0;
    std::vector<DraftAttentionKind> layer_types;
    std::optional<std::uint32_t> sliding_window;
    std::vector<std::uint32_t> target_layer_ids;
    std::uint32_t mask_token_id = 0;
    std::optional<DFlash2Config> dflash2;

    [[nodiscard]] std::uint32_t local_layer_count() const noexcept {
        return static_cast<std::uint32_t>(std::count(layer_types.begin(), layer_types.end(),
                                                     DraftAttentionKind::SlidingAttention));
    }

    [[nodiscard]] std::uint32_t full_layer_count() const noexcept {
        return num_hidden_layers - local_layer_count();
    }

    [[nodiscard]] std::uint32_t compact_layer_index(std::uint32_t layer) const {
        const auto kind = layer_types.at(layer);
        return static_cast<std::uint32_t>(
            std::count(layer_types.begin(), layer_types.begin() + layer, kind));
    }
};

struct Config {
    TextConfig text;
    std::optional<VisionConfig> vision;
    bool mtp = false;
    std::optional<DraftConfig> draft;
};

[[nodiscard]] Config parse_config(const artifact::Directory& directory, const LoadOptions& options);

} // namespace ninfer::models::qwen3_5
