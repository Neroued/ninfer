#pragma once

#include "ninfer/types.h"

namespace ninfer::models::inkling2 {

struct ExecutionShape {
    std::uint32_t max_context                    = 0;
    std::uint32_t max_concurrency                = 0;
    std::uint32_t prefill_chunk                  = 0;
    std::uint32_t draft_tokens                   = 0;
    KvCacheStorage kv_storage                    = KvCacheStorage::BFloat16;
    KvCacheStorage swa_kv_storage                = KvCacheStorage::BFloat16;
    EnginePurpose purpose                        = EnginePurpose::Generation;
    bool context_cache                           = false;
    bool gpu_timing                              = false;
    SpeculativeBackend speculative               = SpeculativeBackend::None;
    SpeculativeLengthPolicy length_policy        = SpeculativeLengthPolicy::Adaptive;
    bool narrow_dspark_targets                   = false;
    bool operator==(const ExecutionShape&) const = default;
};

} // namespace ninfer::models::inkling2
