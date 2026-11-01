#pragma once

#include "models/common/context/directory/state.h"

#include <cstdint>

namespace ninfer::models::qwen3_5 {

struct ContinuationMetadata {
    std::uint32_t execution_frontier      = 0;
    std::int32_t rope_delta               = 0;
    std::uint32_t text_kv_valid           = 0;
    std::uint32_t backend_kv_valid        = 0;
    std::uint32_t dflash_context_frontier = 0;
    bool tail_hidden_valid                = false;
};

using Continuation = context::Continuation<ContinuationMetadata>;

} // namespace ninfer::models::qwen3_5
