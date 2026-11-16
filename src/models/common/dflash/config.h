#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <vector>

namespace ninfer::artifact {
struct Component;
}

namespace ninfer::models::dflash {

enum class AttentionKind { FullAttention, SlidingAttention };

struct TargetGeometry {
    std::uint32_t hidden_size, vocabulary, layers;
};

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

struct DFlash2Config {
    std::uint32_t conv_kernel_size = 0;
    std::uint32_t conv_group_size  = 0;
    std::uint32_t selector_rank    = 0;
    std::uint32_t selector_top_k   = 0;
};

struct Config {
    AttentionConfig attention;
    std::uint32_t intermediate_size       = 0;
    std::uint32_t num_hidden_layers       = 0;
    std::uint32_t max_position_embeddings = 0;
    float rms_norm_eps                    = 0;
    float rope_theta                      = 0;
    std::vector<AttentionKind> layer_types;
    std::optional<std::uint32_t> sliding_window;
    std::vector<std::uint32_t> target_layer_ids;
    std::uint32_t mask_token_id = 0;
    std::optional<std::uint32_t> training_block_size;
    std::optional<DFlash2Config> dflash2;

    [[nodiscard]] std::uint32_t local_layer_count() const noexcept {
        return static_cast<std::uint32_t>(
            std::count(layer_types.begin(), layer_types.end(), AttentionKind::SlidingAttention));
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

[[nodiscard]] Config parse_config(const artifact::Component&, TargetGeometry, bool dflash2);

} // namespace ninfer::models::dflash
