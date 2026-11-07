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

// One packed prefill execution carries at most this many chunks of tokens in total, summed over
// its sequence segments. The Session reports the resulting capacity through
// Session::prefill_pack_tokens(); the Engine derives its Throughput allowance from that and holds
// no copy of this value.
inline constexpr std::uint32_t kPrefillPackChunks = 2;

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

// Upper bound of the summed segment length of one packed prefill execution. It is the only
// authority for that bound: planning sizes the token-axis workspace and the execution frame from
// it, and the Session admits segments against it. The result is a multiple of
// kPrefillChunkAlignment and at least `prefill_chunk`, so a single window always fits. A pack
// never needs to exceed the largest context a request can occupy.
[[nodiscard]] inline std::uint32_t resolve_prefill_pack_tokens(std::uint32_t prefill_chunk,
                                                               std::uint32_t max_context) {
    if (prefill_chunk == 0 || prefill_chunk % kPrefillChunkAlignment != 0)
        throw std::invalid_argument("prefill_chunk must be a nonzero multiple of " +
                                    std::to_string(kPrefillChunkAlignment));
    const std::uint64_t aligned_context =
        (std::uint64_t{max_context} + kPrefillChunkAlignment - 1) / kPrefillChunkAlignment *
        kPrefillChunkAlignment;
    const std::uint64_t packed = std::uint64_t{prefill_chunk} * kPrefillPackChunks;
    return std::max(prefill_chunk, static_cast<std::uint32_t>(std::min(packed, aligned_context)));
}

} // namespace ninfer::models::qwen3_5
