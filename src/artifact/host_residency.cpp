#include "artifact/host_residency.h"

#include "artifact/schema.h"

#include <algorithm>
#include <set>

namespace ninfer::artifact {
namespace {

// Lays out the placements whose `pageable` flag equals `which`, in plan order, each at the next
// multiple of its alignment; writes their offsets and returns the allocation size.
std::uint64_t layout(std::span<const HostMappedPlacement> placements,
                     const std::vector<std::uint8_t>& pageable, std::uint8_t which,
                     std::vector<std::uint64_t>* offset) {
    std::uint64_t end = 0;
    for (std::size_t i = 0; i < placements.size(); ++i) {
        if (pageable[i] != which) { continue; }
        const auto& p = placements[i];
        const auto at = align_up(end, p.alignment, "host residency offset");
        end           = checked_add(at, p.bytes, "host residency bytes");
        if (offset != nullptr) { (*offset)[i] = at; }
    }
    return end;
}

} // namespace

bool any_pageable_candidate(std::span<const HostMappedPlacement> placements) noexcept {
    return std::any_of(placements.begin(), placements.end(),
                       [](const HostMappedPlacement& p) { return p.pageable_rank > 0; });
}

HostResidency plan_host_residency(std::span<const HostMappedPlacement> placements,
                                  std::uint64_t total_bytes, std::uint64_t pinned_budget) {
    HostResidency out;
    out.pageable.assign(placements.size(), 0);
    out.offset.assign(placements.size(), 0);
    if (pinned_budget >= total_bytes) {
        // The plan's own layout, byte for byte: the path every load took before the fallback.
        for (std::size_t i = 0; i < placements.size(); ++i) {
            out.offset[i] = placements[i].offset;
        }
        out.pinned_bytes          = total_bytes;
        out.required_pinned_bytes = total_bytes;
        return out;
    }
    // Start from everything pageable except rank 0, then pin whole ranks while they fit.
    for (std::size_t i = 0; i < placements.size(); ++i) {
        out.pageable[i] = placements[i].pageable_rank > 0 ? 1 : 0;
    }
    out.required_pinned_bytes = layout(placements, out.pageable, 0, nullptr);
    std::set<std::uint32_t> ranks;
    for (const auto& p : placements) {
        if (p.pageable_rank > 0) { ranks.insert(p.pageable_rank); }
    }
    for (const std::uint32_t rank : ranks) {
        std::vector<std::uint8_t> trial = out.pageable;
        for (std::size_t i = 0; i < placements.size(); ++i) {
            if (placements[i].pageable_rank == rank) { trial[i] = 0; }
        }
        if (layout(placements, trial, 0, nullptr) <= pinned_budget) {
            out.pageable = std::move(trial);
        }
    }
    out.pinned_bytes   = layout(placements, out.pageable, 0, &out.offset);
    out.pageable_bytes = layout(placements, out.pageable, 1, &out.offset);
    out.pageable_objects =
        static_cast<std::size_t>(std::count(out.pageable.begin(), out.pageable.end(), 1));
    return out;
}

std::uint64_t cma_pinned_limit(std::uint64_t total_bytes, std::uint64_t available_bytes,
                               std::uint64_t cma_free_bytes, std::uint64_t host_mapped_bytes,
                               std::uint64_t reserved_bytes, std::uint64_t engine_bytes) noexcept {
    if (cma_free_bytes == 0 || available_bytes == 0) { return 0; }
    const std::uint64_t outside =
        available_bytes > cma_free_bytes ? available_bytes - cma_free_bytes : 0;
    const std::uint64_t keep = std::max<std::uint64_t>(2ULL << 30, total_bytes / 20) + engine_bytes;
    const std::uint64_t limit = outside > keep ? outside - keep : 1;
    return limit >= host_mapped_bytes + reserved_bytes ? 0 : limit;
}

} // namespace ninfer::artifact
