#pragma once

#include "models/load_options.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace ninfer::artifact {
struct Directory;
}

namespace ninfer::models::glm5next {

enum class MixerKind { Kda, Mla };
enum class FfnKind { Dense, Moe };

struct LayerConfig {
    MixerKind mixer;
    FfnKind ffn;
    std::uint32_t state_index;
};

struct MhcConfig {
    std::uint32_t streams, sinkhorn_iterations;
    float epsilon;

    [[nodiscard]] std::uint64_t coefficient_width() const {
        return std::uint64_t(streams) * (streams + std::uint64_t{2});
    }
};

struct KdaConfig {
    std::uint32_t heads, head_dim, conv_width, gate_lowrank;
    float gate_lower_bound;

    [[nodiscard]] std::uint64_t width() const { return std::uint64_t(heads) * head_dim; }

    [[nodiscard]] std::uint64_t conv_channels() const { return 3 * width(); }
};

struct MlaConfig {
    std::uint32_t heads, query_dim, value_dim, query_lowrank, latent_dim;
    std::uint32_t index_heads, index_dim, pool_size, pool_budget;
    bool share_mtp_selection;

    [[nodiscard]] std::uint64_t query_width() const { return std::uint64_t(heads) * query_dim; }

    [[nodiscard]] std::uint64_t output_width() const { return std::uint64_t(heads) * value_dim; }

    [[nodiscard]] std::uint64_t index_query_width() const {
        return std::uint64_t(index_heads) * index_dim;
    }
};

struct MoeConfig {
    std::uint32_t experts, top_k, intermediate, shared_experts, groups, selected_groups;
    float routed_scaling_factor;

    [[nodiscard]] std::uint64_t shared_intermediate() const {
        return std::uint64_t(shared_experts) * intermediate;
    }
};

struct TextConfig {
    std::uint32_t hidden, vocabulary, max_context, dense_intermediate;
    float norm_epsilon, swiglu_limit;
    MhcConfig mhc;
    KdaConfig kda;
    MlaConfig mla;
    MoeConfig moe;
    std::vector<LayerConfig> layers;
    std::uint32_t kda_layers = 0, mla_layers = 0;

    [[nodiscard]] std::uint64_t residual_width() const {
        return std::uint64_t(hidden) * mhc.streams;
    }
};

struct VisionConfig {
    std::uint32_t depth, hidden, heads, intermediate;
    std::uint32_t channels, patch_size, temporal_patch_size, merge_size;
    std::uint32_t output_hidden, merger_intermediate;
    float norm_epsilon, swiglu_limit;

    [[nodiscard]] std::uint64_t patch_width() const;
    [[nodiscard]] std::uint64_t merge_width() const;

    [[nodiscard]] std::uint32_t head_dim() const { return hidden / heads; }
};

struct Config {
    TextConfig text;
    std::optional<VisionConfig> vision;
    bool mtp = false;
};

[[nodiscard]] Config parse_config(const artifact::Directory&, const LoadOptions&);

} // namespace ninfer::models::glm5next
