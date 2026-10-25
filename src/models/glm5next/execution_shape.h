#pragma once

#include "ninfer/types.h"

namespace ninfer::models::glm5next {

struct ExecutionShape {
    std::uint32_t max_context, max_concurrency, prefill_chunk, draft_tokens;
    KvCacheStorage kv_storage;
    EnginePurpose purpose;
    bool context_cache, gpu_timing;
    bool operator==(const ExecutionShape&) const = default;
};

} // namespace ninfer::models::glm5next
