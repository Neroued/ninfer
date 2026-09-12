#pragma once

#include <cstdint>

namespace ninfer::ops {

/** Logical head geometry for explicit-Q/K/V Softmax Attention. */
struct AttentionHeadGeometry {
    std::int32_t head_dim    = 0;
    std::int32_t query_heads = 0;
    std::int32_t kv_heads    = 0;
};

[[nodiscard]] constexpr bool valid_attention_head_geometry(AttentionHeadGeometry geometry) {
    return geometry.head_dim > 0 && geometry.query_heads > 0 && geometry.kv_heads > 0 &&
           geometry.query_heads % geometry.kv_heads == 0;
}

} // namespace ninfer::ops
