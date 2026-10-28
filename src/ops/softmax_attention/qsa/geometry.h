#pragma once
#include "core/platform.h"
#include "core/paged_kv_cache.h"

namespace ninfer::ops::detail::qsa {
template <int KvHeads>
struct HeadGeometry {
    static_assert(KvHeads == 1 || KvHeads == 2 || KvHeads == 4);
    static constexpr int query_heads = KvHeads == 1 ? 12 : 24;
    static constexpr int ratio       = query_heads / KvHeads;
};

// Represented head geometries. Each Op admits only its implemented profiles.
inline int registered_query_heads(int kv, KvCacheStorage storage) {
    if (kv == 1) return HeadGeometry<1>::query_heads;
    if (kv == 2) return HeadGeometry<2>::query_heads;
#if NINFER_TARGET_RTX5090
    if (kv == 4 && (storage == KvCacheStorage::Fp8E4M3Row256 ||
                    storage == KvCacheStorage::BFloat16 || storage == KvCacheStorage::Int8Group64))
        return HeadGeometry<4>::query_heads;
#endif
    return 0;
}
} // namespace ninfer::ops::detail::qsa
