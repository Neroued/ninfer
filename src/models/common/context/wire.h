#pragma once

#include "distributed/wire.h"
#include "core/paged_kv_cache.h"
#include "models/common/context/contract.h"
#include "models/common/context/directory.h"

namespace ninfer::models::context {

void encode_actions(distributed::wire::Encoder&, const ContextActions&);
[[nodiscard]] ContextActions decode_actions(distributed::wire::Decoder&);

// Mapping ranges are counted separately from their encoded offsets.
inline std::size_t action_capacity(std::size_t records, std::size_t mappings, std::size_t offsets) {
    return 8 + 46 * records + 21 * mappings + 8 * offsets;
}

inline std::size_t action_batch_capacity() {
    // A peer batch contains one matched copy with at most the offset limit in ranges.
    return std::max(action_capacity(kContextActionBatchOperations, 0, kContextActionBatchOffsets),
                    8 + 25 + 17 * kContextActionBatchOffsets);
}

inline std::size_t write_action_capacity(std::size_t rows, std::size_t pools, std::size_t tokens) {
    const std::size_t page = kPagedKVPageSize;
    const auto pages       = tokens ? rows * pools * ((tokens + 2 * (page - 1)) / page) : 0;
    return action_capacity(pages + rows, rows * pools, pages);
}

void encode_directory(distributed::wire::Encoder&, const ContextDirectorySpec&);
[[nodiscard]] ContextDirectorySpec decode_directory(distributed::wire::Decoder&);

} // namespace ninfer::models::context
