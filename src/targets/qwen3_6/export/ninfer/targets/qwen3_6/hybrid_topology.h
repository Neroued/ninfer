#pragma once

#include <cstdint>

namespace ninfer::targets::qwen3_6 {

inline constexpr std::int32_t kHybridAttentionInterval = 4;

[[nodiscard]] constexpr bool is_full_attention_layer(std::int32_t layer) noexcept {
    return (layer + 1) % kHybridAttentionInterval == 0;
}

[[nodiscard]] constexpr std::int32_t full_attention_layers(std::int32_t total_layers) noexcept {
    return total_layers / kHybridAttentionInterval;
}

[[nodiscard]] constexpr std::int32_t gdn_layers(std::int32_t total_layers) noexcept {
    return total_layers - full_attention_layers(total_layers);
}

[[nodiscard]] constexpr std::int32_t full_attention_index(std::int32_t layer) noexcept {
    return (layer + 1) / kHybridAttentionInterval - 1;
}

[[nodiscard]] constexpr std::int32_t gdn_index(std::int32_t layer) noexcept {
    return layer - (layer + 1) / kHybridAttentionInterval;
}

} // namespace ninfer::targets::qwen3_6
