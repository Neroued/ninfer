#pragma once

namespace ninfer::ops::detail {

template <int QHeadsValue, int KVHeadsValue>
struct AttentionHeadMapping {
    static_assert(QHeadsValue > 0 && KVHeadsValue > 0);
    static_assert(QHeadsValue % KVHeadsValue == 0);

    static constexpr int QHeads    = QHeadsValue;
    static constexpr int KVHeads   = KVHeadsValue;
    static constexpr int GroupSize = QHeads / KVHeads;
};

inline constexpr int kCausalHeadDim = 256;

template <int HeadDim, int QueryHeads, int KVHeads>
struct CausalGeometry : AttentionHeadMapping<QueryHeads, KVHeads> {
    static_assert(HeadDim > 0 && HeadDim % 64 == 0);
    static constexpr int kHeadDim = HeadDim;
};

using CausalD256H24Kv4 = CausalGeometry<256, 24, 4>;
using CausalD256H16Kv2 = CausalGeometry<256, 16, 2>;
} // namespace ninfer::ops::detail
