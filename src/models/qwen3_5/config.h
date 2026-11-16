#pragma once

#include "models/common/dflash/config.h"
#include "models/common/qwen/vision/parameters.h"
#include "models/load_options.h"
#include "models/registry.h"

#include <array>
#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

namespace ninfer::artifact {
struct Directory;
}

namespace ninfer::models::qwen3_5 {

enum class MixerKind { FullAttention, LinearAttention };

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

struct Config {
    TextConfig text;
    std::optional<qwen::VisionConfig> vision;
    bool mtp = false;
    std::optional<dflash::Config> draft;
};

[[nodiscard]] ModelSamplingDefaults default_sampling(Architecture architecture);

[[nodiscard]] Config parse_config(const artifact::Directory& directory, const LoadOptions& options);

} // namespace ninfer::models::qwen3_5
