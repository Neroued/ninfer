#pragma once

#include "ninfer/observability/resources.h"

namespace ninfer::ops::detail::qsa {
template <KvCacheStorage Storage, bool Key>
inline constexpr auto kCachePlaneFormat =
    Storage == KvCacheStorage::Fp8KeyNvfp4Value
        ? (Key ? KvCacheStorage::Fp8E4M3Row256 : KvCacheStorage::Nvfp4Group16)
        : Storage;
} // namespace ninfer::ops::detail::qsa
