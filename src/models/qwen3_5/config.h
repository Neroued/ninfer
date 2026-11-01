#pragma once

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

enum class GateActivation { Silu, Sigmoid };

// Qwen4Exp wide residual: hc_count streams of hidden_size, low-rank input mixing per block.
struct HyperConnectionConfig {
    std::uint32_t hc_count   = 0;
    std::uint32_t hc_lowrank = 0;
};

// Qwen Sparse Attention indexer: one raw key head per token, pooled over compress_ratio tokens.
struct SparseAttentionConfig {
    std::uint32_t indexer_heads    = 0;
    std::uint32_t indexer_head_dim = 0;
    std::uint32_t budget           = 0;
    std::uint32_t compress_ratio   = 0;

    [[nodiscard]] std::uint32_t block_topk() const noexcept { return budget / compress_ratio; }

    // Budget blocks plus the incomplete tail.
    [[nodiscard]] std::uint32_t max_selected_tokens() const noexcept {
        return budget + compress_ratio - 1;
    }
};

// Qwen4Exp n-gram hash embedding injected before one GDN block (absent when the config names no
// PLE layer).
struct PleConfig {
    std::uint32_t layer           = 0; // zero-based block index
    std::uint32_t ngram_size      = 0;
    std::uint32_t heads_per_ngram = 0;
    std::uint32_t embed_dim       = 0;
    std::uint32_t conv_kernel     = 0;
    std::uint32_t eos_token_id    = 0;
    std::uint64_t table_rows      = 0;
    std::vector<std::uint64_t> multipliers;  // ngram_size odd hash multipliers
    std::vector<std::uint64_t> head_moduli;  // prime modulus per hash head
    std::vector<std::uint64_t> head_offsets; // table row offset per hash head

    [[nodiscard]] std::uint32_t heads() const noexcept {
        return (ngram_size - 1) * heads_per_ngram;
    }

    [[nodiscard]] std::uint32_t row_width() const noexcept { return embed_dim / heads(); }

    [[nodiscard]] std::uint32_t conv_dilation() const noexcept { return ngram_size; }

    [[nodiscard]] std::uint32_t conv_history() const noexcept {
        return (conv_kernel - 1) * conv_dilation();
    }
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
    // GDN output norm gate (Qwen4Exp output_gate_type; Qwen3.5 is SiLU).
    GateActivation gdn_gate = GateActivation::Silu;
    std::optional<HyperConnectionConfig> hyper_connection; // Qwen4Exp only
    std::optional<SparseAttentionConfig> sparse_attention; // Qwen4Exp only
    std::optional<PleConfig> ple;                          // Qwen4Exp only, optional

    [[nodiscard]] bool qwen4() const noexcept { return architecture == Architecture::Qwen4Exp; }

    // Qwen4Exp with the routed-expert MoE (Qwen3.8-Flash-Next); false for a dense Qwen4Exp FFN,
    // which instantiates no expert backend, expert planning or host MoE machinery.
    [[nodiscard]] bool qwen4_routed_experts() const noexcept {
        return qwen4() && std::holds_alternative<MoeConfig>(ffn);
    }

    // Width of the residual carried between blocks and handed to MTP.
    [[nodiscard]] std::uint64_t residual_width() const noexcept {
        return hyper_connection ? std::uint64_t(hyper_connection->hc_count) * hidden_size
                                : hidden_size;
    }
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
    // Every layer attends over this sliding window (DFlash2 drafters have no full-attention layer).
    std::uint32_t sliding_window = 0;
    std::vector<std::uint32_t> target_layer_ids;
    std::uint32_t mask_token_id = 0;
    DFlash2Config dflash2;
};

struct Config {
    TextConfig text;
    std::optional<VisionConfig> vision;
    bool mtp = false;
    std::optional<DraftConfig> draft;
};

[[nodiscard]] Config parse_config(const artifact::Directory& directory, const LoadOptions& options);

} // namespace ninfer::models::qwen3_5
