#pragma once

// Pure complexity core of the context-reclaim ranking. Exposed as a header template so the
// serving-thread hot loop can be exercised on synthetic checkpoints (tests) without a live
// ProgramImpl. Production callers pass a ProgramImpl& (which provides valid_checkpoint() and
// checkpoint()); tests pass a lightweight view with the same two members.
//
// The survivor-membership test below is O(1) via a hash set (see fix for
// Neroued/ninfer#402): the previous linear std::find over `ordered` made the whole loop
// O(n^2*m) (cubic when n ~= m ~= N retained checkpoints), which stalled the serving thread.

#include "models/qwen3_5/program/program.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <span>
#include <unordered_set>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

template <typename View, typename Prefix>
[[nodiscard]] std::uint64_t
recovery_loss(const View& view, std::span<const CheckpointHandle> removed,
              std::span<const CheckpointHandle> surviving, Prefix&& prefix) {
    struct HandleHash {
        std::size_t operator()(const CheckpointHandle& h) const noexcept {
            std::size_t seed = std::hash<std::uint32_t>{}(h.index);
            seed = seed * 1000003U + std::hash<std::uint64_t>{}(h.generation);
            seed = seed * 1000003U + std::hash<const void*>{}(h.owner);
            return seed;
        }
    };

    std::vector<CheckpointHandle> ordered;
    std::unordered_set<CheckpointHandle, HandleHash> member;
    ordered.reserve(removed.size());
    member.reserve(removed.size());
    for (const auto handle : removed) {
        // Dedup and O(1) survivor-membership in one pass (was an O(n) std::find per element).
        if (view.valid_checkpoint(handle) && member.insert(handle).second) {
            ordered.push_back(handle);
        }
    }
    std::sort(ordered.begin(), ordered.end(), [&](auto left, auto right) {
        const auto l = view.checkpoint(left).frontier(), r = view.checkpoint(right).frontier();
        return l != r ? l > r : left.index < right.index;
    });

    struct LineageLoss {
        CheckpointHandle deepest;
        std::uint32_t tokens;
    };

    std::vector<LineageLoss> lineages;
    for (const auto handle : ordered) {
        const auto& record     = view.checkpoint(handle);
        std::uint32_t fallback = 0;
        for (const auto retained : surviving) {
            if (!view.valid_checkpoint(retained) || member.count(retained) != 0u) {
                continue;
            }
            const auto& candidate = view.checkpoint(retained);
            if (candidate.frontier() > fallback && prefix(retained, handle)) {
                fallback = candidate.frontier();
            }
        }
        const auto lost    = record.frontier() - fallback;
        const auto lineage = std::find_if(lineages.begin(), lineages.end(), [&](const auto& item) {
            return prefix(handle, item.deepest);
        });
        if (lineage == lineages.end()) {
            lineages.push_back({handle, lost});
        } else {
            lineage->tokens = std::max(lineage->tokens, lost);
        }
    }
    std::uint64_t result = 0;
    for (const auto& lineage : lineages) { result += lineage.tokens; }
    return result;
}

} // namespace ninfer::models::qwen3_5::detail
