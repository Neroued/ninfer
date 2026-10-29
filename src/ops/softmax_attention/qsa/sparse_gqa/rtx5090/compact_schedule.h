#pragma once

#include "ninfer/observability/resources.h"

namespace ninfer::ops::detail::qsa {

// The launch bound constrains compilation. Modeled residency also accounts for
// the selected kernel's shared memory and measured register allocation.
constexpr int compact_launch_min_blocks(KvCacheStorage storage, int kv_heads, int rows, bool prefix,
                                        bool reused, bool delay_softmax) {
    const bool native = storage == KvCacheStorage::BFloat16;
    if (kv_heads == 2 && (rows == 16 || (!prefix && reused && (delay_softmax || !native))))
        return 5;
    if (kv_heads == 4 && !prefix) {
        if (native && rows == 32) return 5;
        if (!native && reused && rows == 16) return 10;
    }
    return 1;
}

constexpr int compact_mma_resident_blocks(KvCacheStorage storage, int kv_heads, int rows,
                                          bool reused, bool delay_softmax) {
    const int minimum =
        compact_launch_min_blocks(storage, kv_heads, rows, false, reused, delay_softmax);
    if (minimum > 1) return minimum;
    if (kv_heads == 2) {
        if (storage == KvCacheStorage::BFloat16) return delay_softmax ? 5 : 4;
        return storage == KvCacheStorage::Nvfp4Group16 ? 4 : 3;
    }
    if (rows == 16) return storage == KvCacheStorage::Nvfp4Group16 ? 8 : 7;
    return 2;
}

} // namespace ninfer::ops::detail::qsa
