#pragma once

// Compile-time form of the public contiguous query-head to KV-head mapping.

namespace ninfer::ops {

template <int QHeadsValue, int KVHeadsValue>
struct AttentionHeadMapping {
    static_assert(QHeadsValue > 0 && KVHeadsValue > 0);
    static_assert(QHeadsValue % KVHeadsValue == 0);

    static constexpr int QHeads    = QHeadsValue;
    static constexpr int KVHeads   = KVHeadsValue;
    static constexpr int GroupSize = QHeads / KVHeads;
};

} // namespace ninfer::ops
