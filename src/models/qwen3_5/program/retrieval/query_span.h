#pragma once

#include "models/qwen3_5/frontend/prepared_prompt.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {

struct KvMemQuerySpan {
    std::uint32_t begin = 0;
    std::uint32_t end   = 0;
    bool exact         = false;
};

inline KvMemQuerySpan kvmem_query_span(std::uint32_t prompt_tokens, std::uint32_t reuse_base,
                                      const std::optional<TokenSpan>& query,
                                      std::span<const VisionItem> media = {}) {
    if (reuse_base > prompt_tokens) { throw std::logic_error("query reuse base exceeds prompt"); }
    const bool exact = query && query->begin >= reuse_base && query->begin <= prompt_tokens &&
                       query->count > 0 && query->count <= prompt_tokens - query->begin;
    const auto end = exact ? static_cast<std::uint32_t>(query->begin + query->count) : prompt_tokens;
    const auto floor = exact ? static_cast<std::uint32_t>(query->begin) : reuse_base;
    auto begin = std::max(floor, end > 512U ? end - 512U : 0U);
    for (const auto& item : media) {
        if (item.token_spans.empty()) { continue; }
        const auto first = item.token_spans.front().begin;
        const auto& last = item.token_spans.back();
        const auto consumer_begin = first == 0 ? 0 : first - 1;
        if (consumer_begin < begin && begin < last.begin + last.count) {
            begin = std::max(reuse_base, static_cast<std::uint32_t>(consumer_begin));
            break;
        }
    }
    return {begin, end, exact};
}

// Replay returns to Scheduler after each chunk, including a chunk shortened by
// a rewrite frontier. These are real service units, although prompt progress is
// counted only once. Query-checkpoint splits inside the initial step add no unit.
inline std::uint64_t kvmem_replay_quanta(std::uint32_t prompt_tokens, std::uint32_t query_begin,
                                        std::uint32_t window_tokens, std::uint32_t prefill_chunk,
                                        std::span<const std::uint32_t> rewrite_frontiers) {
    if (window_tokens == 0 || prompt_tokens <= window_tokens || query_begin == 0 ||
        query_begin >= prompt_tokens) { return 0; }
    if (prefill_chunk == 0) { throw std::logic_error("query replay needs a nonzero chunk"); }
    std::uint64_t units = 0;
    std::uint32_t begin = query_begin;
    for (const auto frontier : rewrite_frontiers) {
        if (frontier <= begin) { continue; }
        if (frontier >= prompt_tokens) { break; }
        units += 1ULL + (frontier - begin - 1ULL) / prefill_chunk;
        begin = frontier;
    }
    return units + 1ULL + (prompt_tokens - begin - 1ULL) / prefill_chunk;
}

} // namespace ninfer::models::qwen3_5::detail
