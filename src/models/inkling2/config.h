#pragma once

#include "models/load_options.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

namespace ninfer::artifact {
struct Directory;
}

namespace ninfer::models::inkling2 {

[[nodiscard]] inline std::int32_t extent(std::uint64_t value) {
    if (value > std::uint64_t(std::numeric_limits<std::int32_t>::max()))
        throw std::overflow_error("Inkling2 tensor extent is outside int32");
    return static_cast<std::int32_t>(value);
}

[[nodiscard]] inline std::int32_t dimension(std::uint64_t value) {
    if (!value) throw std::overflow_error("Inkling2 tensor extent is outside int32");
    return extent(value);
}

struct AttentionGeometry {
    std::uint32_t heads    = 0;
    std::uint32_t kv_heads = 0;
    std::uint32_t head_dim = 0;

    [[nodiscard]] std::uint64_t query_width() const noexcept {
        return std::uint64_t(heads) * head_dim;
    }

    [[nodiscard]] std::uint64_t kv_width() const noexcept {
        return std::uint64_t(kv_heads) * head_dim;
    }
};

struct LoopConfig {
    std::uint32_t core_start = 0;
    std::uint32_t core_end   = 0;
    std::uint32_t num_passes = 1;
    std::uint32_t gate_rank  = 0;

    [[nodiscard]] std::uint32_t core_layers() const noexcept { return core_end - core_start; }
};

struct MoeConfig {
    std::uint32_t experts      = 0;
    std::uint32_t top_k        = 0;
    std::uint32_t intermediate = 0;
    float gate_softcap         = 0.0F;
    float up_softcap           = 0.0F;
    float route_scale          = 0.0F;
};

struct PhysicalLayerConfig {
    std::uint32_t physical_index = 0;
    bool is_swa                  = false;
    bool in_core                 = false;
    std::uint32_t num_passes     = 1;
};

struct LogicalLayerConfig {
    std::uint32_t logical_index         = 0;
    std::uint32_t physical_layer        = 0;
    std::uint32_t pass_index            = 0;
    bool is_swa                         = false;
    bool in_core                        = false;
    std::uint32_t attention_state_index = 0;
    std::uint32_t block_index           = 0;
    bool starts_new_block               = false;
    std::optional<std::uint32_t> capture_aux_slot;
};

struct TextConfig {
    std::uint32_t hidden                  = 0;
    std::uint32_t num_layers              = 0;
    std::uint32_t vocabulary              = 0;
    std::uint32_t unpadded_vocabulary     = 0;
    std::uint32_t max_context             = 0;
    std::uint32_t eos_token_id            = 0;
    std::uint32_t attn_res_block_size     = 0;
    std::uint32_t sconv_kernel_size       = 0;
    std::uint32_t d_rel                   = 0;
    std::uint32_t rel_extent              = 0;
    std::uint32_t sliding_window          = 0;
    float norm_epsilon                    = 0.0F;
    float qk_l2_clip_threshold            = 0.0F;
    float rel_l2_clip_threshold           = 0.0F;
    std::optional<std::uint32_t> log_scaling_n_floor;
    float log_scaling_alpha               = 0.0F;
    float final_logit_softcapping         = 0.0F;
    AttentionGeometry full_attention;
    AttentionGeometry swa_attention;
    MoeConfig moe;
    std::optional<LoopConfig> loop;
    std::vector<PhysicalLayerConfig> physical_layers;
    std::vector<LogicalLayerConfig> logical_layers;
    std::uint32_t swa_layers  = 0;
    std::uint32_t full_layers = 0;

    [[nodiscard]] const AttentionGeometry& attention(bool is_swa) const noexcept {
        return is_swa ? swa_attention : full_attention;
    }

    [[nodiscard]] std::uint64_t rel_width(bool is_swa) const noexcept {
        return std::uint64_t(attention(is_swa).heads) * d_rel;
    }

    [[nodiscard]] std::uint32_t logical_layer_count() const noexcept {
        return static_cast<std::uint32_t>(logical_layers.size());
    }
};

struct VisionConfig {
    std::uint32_t depth                 = 0;
    std::uint32_t hidden                = 0;
    std::uint32_t heads                 = 0;
    std::uint32_t head_dim              = 0;
    std::uint32_t intermediate          = 0;
    std::uint32_t native_patch_size     = 0;
    std::uint32_t merge_size            = 0;
    std::uint32_t temporal_patch_size   = 0;
    std::uint32_t posemb_grid_size      = 0;
    std::uint32_t projection_hidden_dim = 0;
    std::uint32_t text_hidden           = 0;
    bool projector_bias                 = false;
    float norm_epsilon                  = 0.0F;

    [[nodiscard]] std::uint64_t patch_width() const;
    [[nodiscard]] std::uint64_t merge_width() const;
};

struct DSparkLayerConfig {
    bool is_swa              = false;
    std::uint32_t state_slot = 0;
};

struct DSparkConfig {
    std::uint32_t hidden              = 0;
    std::uint32_t num_layers          = 0;
    std::uint32_t heads               = 0;
    std::uint32_t kv_heads            = 0;
    std::uint32_t head_dim            = 0;
    std::uint32_t intermediate        = 0;
    std::uint32_t max_context         = 0;
    std::uint32_t vocabulary          = 0;
    std::uint32_t unpadded_vocabulary = 0;
    std::uint32_t sliding_window      = 0;
    std::uint32_t block_size          = 0;
    std::uint32_t markov_rank         = 0;
    float norm_epsilon                = 0.0F;
    float rope_theta                  = 0.0F;
    float final_logit_softcapping     = 0.0F;
    std::vector<DSparkLayerConfig> layers;
    std::uint32_t swa_layers  = 0;
    std::uint32_t full_layers = 0;
    std::vector<std::uint32_t> target_layer_ids;
    std::optional<std::uint32_t> final_aux_slot;

    [[nodiscard]] std::uint64_t query_width() const noexcept {
        return std::uint64_t(heads) * head_dim;
    }

    [[nodiscard]] std::uint64_t kv_width() const noexcept {
        return std::uint64_t(kv_heads) * head_dim;
    }

    [[nodiscard]] std::uint32_t max_draft_tokens() const noexcept { return block_size - 1U; }
};

struct Config {
    TextConfig text;
    std::optional<VisionConfig> vision;
    std::optional<DSparkConfig> dspark;
};

[[nodiscard]] Config parse_config(const artifact::Directory&, const LoadOptions&);

} // namespace ninfer::models::inkling2
