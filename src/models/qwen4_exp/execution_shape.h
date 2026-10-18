#pragma once

#include "ninfer/types.h"

namespace ninfer::models::qwen4_exp {

struct ExecutionShape {
    std::uint32_t max_context, max_concurrency, prefill_chunk, draft_tokens;
    KvCacheStorage kv_storage;
    EnginePurpose purpose;
    bool context_cache = false;
    bool cuda_graph    = false;
};

} // namespace ninfer::models::qwen4_exp
