#pragma once

#include <ninfer/targets/qwen3_6/frontend.h>
#include <ninfer/targets/qwen3_6/hybrid_topology.h>
#include <ninfer/targets/qwen3_6/vision.h>

#include <cstdint>

namespace ninfer::targets::qwen3_6_35b_a3b::detail {

struct TextConfig {
    static constexpr int hidden       = 2048;
    static constexpr int layers       = 40;
    static constexpr int intermediate = 512;

    static constexpr int output_rows  = 248320;
    static constexpr int token_domain = static_cast<int>(qwen3_6::kTokenDomain);

    static constexpr int gdn_conv_kernel      = 4;
    static constexpr int gdn_conv_state_width = gdn_conv_kernel - 1;
    static constexpr int gdn_key_heads        = 16;
    static constexpr int gdn_key_head_dim     = 128;
    static constexpr int gdn_value_heads      = 32;
    static constexpr int gdn_value_head_dim   = 128;

    static constexpr int query_heads = 16;
    static constexpr int kv_heads    = 2;
    static constexpr int head_dim    = 256;
    static constexpr int rotary_dim  = 64;

    static constexpr float rms_epsilon = 1.0e-6F;
    static constexpr float rope_theta  = 1.0e7F;

    static constexpr int key_dim               = gdn_key_heads * gdn_key_head_dim;
    static constexpr int value_dim             = gdn_value_heads * gdn_value_head_dim;
    static constexpr int convolution_dim       = 2 * key_dim + value_dim;
    static constexpr int query_size            = query_heads * head_dim;
    static constexpr int kv_size               = kv_heads * head_dim;
    static constexpr int query_projection_rows = 2 * query_size;

    static constexpr int mtp_layers               = 1;
    static constexpr int mtp_input_rows           = 2 * hidden;
    static constexpr int mtp_attention_input_rows = 2 * query_size + 2 * kv_size;
    static constexpr int mtp_mlp_gate_up_rows     = 2 * intermediate;

    [[nodiscard]] static constexpr bool is_full_attention(int layer) {
        return qwen3_6::is_full_attention_layer(layer);
    }

    [[nodiscard]] static constexpr int full_attention_layers() {
        return qwen3_6::full_attention_layers(layers);
    }

    [[nodiscard]] static constexpr int gdn_layers() { return qwen3_6::gdn_layers(layers); }

    [[nodiscard]] static constexpr int full_attention_index(int layer) {
        return qwen3_6::full_attention_index(layer);
    }

    [[nodiscard]] static constexpr int gdn_index(int layer) { return qwen3_6::gdn_index(layer); }
};

static_assert(TextConfig::full_attention_layers() == 10);
static_assert(TextConfig::gdn_layers() == 30);

struct VisionConfig : qwen3_6::VisionBackboneConfig {
    static constexpr int output_hidden = TextConfig::hidden;
};

inline constexpr float kAttentionScale                = 0.0625F;
inline constexpr float kGdnScale                      = 0.08838834764831845F;
inline constexpr std::uint32_t kPrefillChunkAlignment = 128;
inline constexpr std::uint32_t kMaximumMtpDraftTokens = 5;
inline constexpr std::uint32_t kNativeContext         = 262144;

} // namespace ninfer::targets::qwen3_6_35b_a3b::detail
