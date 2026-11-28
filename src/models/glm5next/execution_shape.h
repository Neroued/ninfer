#pragma once

#include "ninfer/types.h"
#include "models/common/moe/step.h"

namespace ninfer::models::glm5next {

struct ExecutionShape {
    std::uint32_t max_context, max_concurrency, prefill_chunk, draft_tokens;
    KvCacheStorage kv_storage;
    KvCacheVectorFormat index_format;
    EnginePurpose purpose;
    bool context_cache, gpu_timing;
    SpeculativeBackend speculative               = SpeculativeBackend::None;
    bool narrow_dflash_targets                   = false;
    SpeculativeLengthPolicy length_policy        = SpeculativeLengthPolicy::Adaptive;
    LinearStateDType linear_state_dtype          = LinearStateDType::Float32;
    std::uint32_t pipeline_stages                = 1;
    moe::ExchangeGeometry exchange;
    bool operator==(const ExecutionShape&) const = default;
};

} // namespace ninfer::models::glm5next
