#pragma once

#include "models/common/startup.h"
#include "models/qwen3_5/planning.h"
#include "ninfer/types.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::models::qwen3_5 {

// Default prefill chunks. This header is the only place that decides which chunk a placement runs
// with when the caller does not set one; the sequence planner accepts only a resolved value.
inline constexpr std::uint32_t kResidentPrefillChunk    = 2048;
inline constexpr std::uint32_t kHostExpertsPrefillChunk = 4096;

// Explicit EngineOptions::prefill_chunk wins over the placement default. The result is clamped to
// max_context and must be a multiple of kPrefillChunkAlignment.
[[nodiscard]] inline std::uint32_t resolve_prefill_chunk(const EngineOptions& options,
                                                         WeightPlacement placement) {
    const auto base  = options.prefill_chunk != 0                  ? options.prefill_chunk
                       : placement == WeightPlacement::HostExperts ? kHostExpertsPrefillChunk
                                                                   : kResidentPrefillChunk;
    const auto chunk = std::min(base, options.max_context);
    if (chunk == 0 || chunk % kPrefillChunkAlignment != 0)
        throw std::invalid_argument("prefill_chunk must be a nonzero multiple of " +
                                    std::to_string(kPrefillChunkAlignment));
    return chunk;
}

} // namespace ninfer::models::qwen3_5
