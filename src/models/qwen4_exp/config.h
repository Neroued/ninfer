#pragma once

#include "models/common/qwen/vision/parameters.h"
#include "models/load_options.h"

#include <array>
#include <limits>
#include <stdexcept>
#include <optional>
#include <vector>

namespace ninfer::artifact {
struct Directory;
}

namespace ninfer::models::qwen4_exp {

[[nodiscard]] inline std::int32_t dimension(std::uint64_t value) {
    if (value > std::uint64_t(std::numeric_limits<std::int32_t>::max()))
        throw std::overflow_error("model dimension exceeds the Tensor integer domain");
    return static_cast<std::int32_t>(value);
}

enum class MixerKind { Gdn, Qsa };

struct LayerConfig {
    MixerKind mixer;
    std::uint32_t state_index;
};

struct HcConfig {
    std::uint32_t streams, lowrank;
};

struct GdnConfig {
    std::uint32_t key_heads, value_heads, key_dim, value_dim, conv_width;

    [[nodiscard]] std::uint64_t key_width() const { return std::uint64_t(key_heads) * key_dim; }

    [[nodiscard]] std::uint64_t value_width() const {
        return std::uint64_t(value_heads) * value_dim;
    }

    [[nodiscard]] std::uint64_t conv_channels() const;
};

struct QsaConfig {
    std::uint32_t query_heads, kv_heads, head_dim;
    std::uint32_t index_heads, index_dim, compression, block_budget;
    float rope_theta;
    std::uint32_t index_kv_heads, rotary_dim;
    std::array<std::uint32_t, 3> rope_sections;
    bool rope_interleaved;

    [[nodiscard]] std::uint64_t query_width() const {
        return std::uint64_t(query_heads) * head_dim;
    }

    [[nodiscard]] std::uint64_t kv_width() const { return std::uint64_t(kv_heads) * head_dim; }

    [[nodiscard]] std::uint64_t index_query_width() const {
        return std::uint64_t(index_heads) * index_dim;
    }

    [[nodiscard]] std::uint64_t index_key_width() const {
        return std::uint64_t(index_kv_heads) * index_dim;
    }
};

struct MoeConfig {
    std::uint32_t experts, top_k, intermediate, shared_intermediate;
};

struct PleConfig {
    std::uint32_t layer, shards, rows_per_shard;
    std::int32_t eos;
    std::uint32_t embedding, ngram_size, heads_per_ngram, row_width, conv_width, dilation;

    [[nodiscard]] std::uint64_t history_columns() const {
        return std::uint64_t(conv_width - 1) * dilation;
    }
};

struct TextConfig {
    std::uint32_t hidden, vocabulary, max_context;
    float norm_epsilon;
    HcConfig hc;
    GdnConfig gdn;
    QsaConfig qsa;
    MoeConfig moe;
    PleConfig ple;
    std::vector<LayerConfig> layers;
    std::uint32_t gdn_layers = 0, qsa_layers = 0;

    [[nodiscard]] std::uint64_t residual_width() const {
        return std::uint64_t(hidden) * hc.streams;
    }
};

struct Config {
    TextConfig text;
    std::optional<qwen::VisionConfig> vision;
    bool mtp = false;
};

[[nodiscard]] Config parse_config(const artifact::Directory&, const LoadOptions&);

} // namespace ninfer::models::qwen4_exp
