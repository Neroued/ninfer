#pragma once

#include "ninfer/types.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "models/qwen3_5/program/retrieval/block_retrieval.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <span>
#include <stdexcept>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

struct MediaPageGroup {
    std::uint32_t begin = 0;
    std::uint32_t end = 0;
};

// Include the predecessor consumed by the MTP embedding bridge. Items sharing a
// physical page form one indivisible group, including intervening video timestamps.
inline std::vector<MediaPageGroup> media_page_groups(std::span<const VisionItem> items) {
    std::vector<MediaPageGroup> groups;
    for (const auto& item : items) {
        if (item.token_spans.empty()) { throw std::invalid_argument("media item has no token span"); }
        const auto first = item.token_spans.front().begin;
        const auto& last = item.token_spans.back();
        const MediaPageGroup group{
            static_cast<std::uint32_t>((first == 0 ? 0 : first - 1) / 64),
            static_cast<std::uint32_t>((last.begin + last.count + 63) / 64)};
        if (!groups.empty() && group.begin < groups.back().end) {
            groups.back().end = std::max(groups.back().end, group.end);
        } else {
            groups.push_back(group);
        }
    }
    return groups;
}

inline void validate_media_window(std::span<const MediaPageGroup> groups, std::uint32_t budget) {
    std::uint32_t sink_extent = 2;
    for (const auto& group : groups) {
        if (group.begin < 2) { sink_extent = std::max(sink_extent, group.end); }
    }
    for (const auto& group : groups) {
        // Two sink pages and two boundary/recent pages remain available. Reject at
        // admission rather than silently cutting a media group midway through prefill.
        const auto required = sink_extent + (group.begin < 2 ? 0 : group.end - group.begin) + 2;
        if (required > budget) {
            throw RequestError(RequestErrorKind::MediaBudgetExceeded,
                "KVMem window cannot hold a complete media group plus sink and tail pages");
        }
    }
}

// Latest visible media is mandatory; other media is selected as a whole or omitted.
// Only the materialized prefix of an in-progress item exists. No future KV is loaded.
// Text-only requests retain the existing selection policy exactly.
inline std::vector<std::uint32_t> media_window_page_set(
    std::uint32_t mapped, std::uint32_t budget, std::span<const std::uint32_t> preferred,
    std::span<const MediaPageGroup> groups, bool fill_recent = true) {
    if (groups.empty()) { return decode_window_page_set(mapped, budget, preferred); }
    std::vector<std::uint8_t> kept(mapped, 0);
    std::uint32_t used = 0;
    const auto keep = [&](std::uint32_t page) {
        if (page >= mapped) { return true; }
        if (kept[page]) { return true; }
        std::uint32_t begin = page, end = page + 1;
        const auto it = std::upper_bound(groups.begin(), groups.end(), page,
            [](std::uint32_t p, const MediaPageGroup& group) { return p < group.begin; });
        if (it != groups.begin() && page < std::prev(it)->end) {
            begin = std::prev(it)->begin;
            end = std::min(mapped, std::prev(it)->end);
        }
        std::uint32_t cost = 0;
        for (auto p = begin; p < end; ++p) { cost += !kept[p]; }
        if (cost > budget - used) { return false; }
        for (auto p = begin; p < end; ++p) { kept[p] = 1; }
        used += cost;
        return true;
    };
    for (std::uint32_t p = 0; p < std::min(mapped, 2U); ++p) {
        if (!keep(p)) { throw std::logic_error("KVMem media sink exceeds its admitted window"); }
    }
    for (auto it = groups.rbegin(); it != groups.rend(); ++it) {
        if (it->begin < mapped) {
            if (!keep(it->begin)) {
                throw std::logic_error("KVMem latest media exceeds its admitted window");
            }
            break;
        }
    }
    for (const auto p : preferred) { (void)keep(p); }
    if (fill_recent) {
        for (auto p = mapped; p > 0 && used < budget;) { (void)keep(--p); }
    }
    std::vector<std::uint32_t> pages;
    pages.reserve(used);
    for (std::uint32_t p = 0; p < mapped; ++p) { if (kept[p]) { pages.push_back(p); } }
    return pages;
}

} // namespace ninfer::models::qwen3_5::detail
