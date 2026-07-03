#pragma once

#include <cstdint>

namespace qus::model {

struct ModelConfig {
    static constexpr int hidden       = 5120;
    static constexpr int n_layers     = 64;
    static constexpr int intermediate = 17408;
    static constexpr int vocab        = 248320;

    static constexpr int gdn_conv_k           = 4;
    static constexpr int gdn_conv_state_width = gdn_conv_k - 1;
    static constexpr int gdn_k_heads          = 16;
    static constexpr int gdn_k_dim            = 128;
    static constexpr int gdn_v_heads          = 48;
    static constexpr int gdn_v_dim            = 128;

    static constexpr int n_q        = 24;
    static constexpr int n_kv       = 4;
    static constexpr int head_dim   = 256;
    static constexpr int rotary_dim = 64;

    static constexpr int full_interval = 4;
    static constexpr float rms_eps     = 1.0e-6f;
    static constexpr float rope_theta  = 1.0e7f;

    static constexpr int key_dim    = gdn_k_heads * gdn_k_dim;
    static constexpr int value_dim  = gdn_v_heads * gdn_v_dim;
    static constexpr int conv_dim   = 2 * key_dim + value_dim;
    static constexpr int q_size     = n_q * head_dim;
    static constexpr int kv_size    = n_kv * head_dim;
    static constexpr int q_proj_out = 2 * q_size;

    static constexpr int mtp_layers          = 1;
    static constexpr int mtp_fc_in           = 2 * hidden;
    static constexpr int mtp_attn_in         = 2 * q_size + 2 * kv_size;
    static constexpr int mtp_attn_q_rows     = q_size;
    static constexpr int mtp_attn_k_rows     = kv_size;
    static constexpr int mtp_attn_gate_rows  = q_size;
    static constexpr int mtp_attn_v_rows     = kv_size;
    static constexpr int mtp_mlp_gateup_rows = 2 * intermediate;

    [[nodiscard]] static constexpr bool is_full(int layer) {
        return (layer + 1) % full_interval == 0;
    }

    [[nodiscard]] static constexpr int n_full() { return n_layers / full_interval; }

    [[nodiscard]] static constexpr int n_gdn() { return n_layers - n_full(); }

    [[nodiscard]] static constexpr int full_idx(int layer) {
        return (layer + 1) / full_interval - 1;
    }

    [[nodiscard]] static constexpr int gdn_idx(int layer) {
        return layer - (layer + 1) / full_interval;
    }
};

inline constexpr ModelConfig kCfg{};
inline constexpr float kAttnScale                     = 0.0625f;
inline constexpr float kGdnScale                      = 0.08838834764831845f;
inline constexpr std::uint32_t kPrefillChunkAlignment = 128;
inline constexpr std::uint32_t kDefaultPrefillChunk   = 1024;
inline constexpr int kMaxMtpDraftTokens               = 5;

} // namespace qus::model
