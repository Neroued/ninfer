#pragma once

#include "ninfer/types.h"
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::test {

inline const char* kv_cache_storage_name(KvCacheStorage storage) {
    return kv_cache_profile_info(storage).name;
}

inline KvCacheStorage parse_kv_cache_storage(std::string_view name) {
    return kv_cache_storage_from_option(name);
}

} // namespace ninfer::test
