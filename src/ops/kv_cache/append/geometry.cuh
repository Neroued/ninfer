#pragma once

namespace ninfer::ops {

inline constexpr int kKVCacheAppendFullHeadDim = 256;

template <int KVHeadsValue>
struct KVCacheAppendFullGeometry {
    static_assert(KVHeadsValue == 4 || KVHeadsValue == 2);
    static constexpr int KVHeads = KVHeadsValue;
};

using KVCacheAppendD256Kv4 = KVCacheAppendFullGeometry<4>;
using KVCacheAppendD256Kv2 = KVCacheAppendFullGeometry<2>;

} // namespace ninfer::ops
