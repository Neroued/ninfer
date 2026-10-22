#pragma once

#include "models/common/context/directory/history.h"
#include "models/common/context/release_inventory.h"
#include "models/common/context/directory/state.h"
#include "models/common/context/work.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ninfer::models::context {

struct ReclaimPool {
    const HistoryDirectory* histories = nullptr;
    const PageDirectory* pages        = nullptr;
    std::size_t host_page_bytes       = 0;
};

// Shared store facts for both Native implementations. Model code supplies actual snapshot
// prefixes (including its backend frontier), pool layouts, and state-image size.
template <class Resources>
struct PhysicalReclaim {
    struct Page {
        PageId handle;
        std::uint32_t position = 0;
        bool releasable        = false;
    };

    struct Group {
        std::vector<CheckpointHandle> sources;
        std::vector<StateId> states;
        std::array<std::vector<Page>, Resources::pool_capacity> pages;
        Resources released;
        std::vector<HostExtent> host_extents;
        std::uint32_t host_pages = 0;
    };

    std::vector<Group> groups;
    std::shared_ptr<ReleaseInventory<Resources>> inventory;

    template <class Store>
    PhysicalReclaim(const Store& checkpoints, const StateDirectory& states, std::size_t state_bytes,
                    std::span<const ReclaimPool> pools, std::span<const CheckpointHandle> allowed,
                    std::span<const CheckpointHandle> excluded, Resources shortage)
        : inventory(std::make_shared<ReleaseInventory<Resources>>(checkpoints.size())) {
        std::vector<bool> eligible(checkpoints.size());
        for (const auto h : allowed)
            if (checkpoints.erasable(h)) eligible[h.index] = true;
        for (const auto h : excluded)
            if (checkpoints.valid(h)) eligible[h.index] = false;
        for (std::uint32_t i = 0; i < checkpoints.size(); ++i)
            if (checkpoints.valid(checkpoints.handle(i))) inventory->include(checkpoints.handle(i));

        std::map<std::vector<std::uint32_t>, Group> cohorts;
        const auto permitted = [&](const auto& holders) {
            return std::all_of(holders.begin(), holders.end(), [&](auto i) { return eligible[i]; });
        };
        for (std::uint32_t i = 0; i < eligible.size(); ++i)
            if (eligible[i]) cohorts.try_emplace(std::vector{i});

        if (shortage.state_slots || shortage.host_bytes) {
            struct Image {
                StateId handle;
                std::vector<std::uint32_t> holders;
            };

            std::vector<Image> images(states.capacity());
            for (std::uint32_t i = 0; i < checkpoints.size(); ++i) {
                const auto& slot = checkpoints[i];
                if (!slot.value) continue;
                const auto image = slot.value->snapshot.continuation().image;
                auto& entry      = images[states.descriptor_index(image)];
                entry.handle     = image;
                entry.holders.push_back(i);
            }
            for (const auto& image : images) {
                if (image.holders.empty() || !permitted(image.holders)) continue;
                auto& group = cohorts[image.holders];
                if (states.device_resident(image.handle)) group.states.push_back(image.handle);
                if (image.holders.size() != states.checkpoint_references(image.handle) ||
                    !states.can_release_after_checkpoint_references(image.handle,
                                                                    image.holders.size()))
                    continue;
                if (shortage.state_slots)
                    group.released.state_slots += states.device_resident(image.handle);
                if (shortage.host_bytes && states.host_resident(image.handle)) {
                    group.released.host_bytes += state_bytes;
                    group.host_extents.push_back(states.host_extent(image.handle));
                }
            }
        }

        constexpr auto none = std::numeric_limits<std::uint32_t>::max();
        for (std::uint16_t pool = 0; pool < pools.size(); ++pool) {
            const auto& storage = pools[pool];
            if (!storage.histories || (!shortage.host_bytes && !shortage.kv_page_groups[pool]))
                continue;
            const auto& histories = *storage.histories;
            const auto& pages     = *storage.pages;

            struct History {
                HistoryId handle;
                std::vector<std::uint32_t> holders;
            };

            std::vector<History> histories_by_id(histories.capacity());
            std::vector<const HistoryPrefix*> prefixes;
            for (std::uint32_t i = 0; i < checkpoints.size(); ++i) {
                const auto& slot = checkpoints[i];
                if (!slot.value || pool >= slot.value->snapshot.prefixes().size()) continue;
                const auto& prefix = slot.value->snapshot.prefix(pool);
                prefixes.push_back(&prefix);
                auto& entry  = histories_by_id[histories.descriptor_index(prefix.history())];
                entry.handle = prefix.history();
                entry.holders.push_back(i);
            }
            // Store proof includes owners outside the checkpoint catalog, active histories,
            // unfinished bindings, and source pins. They never become fictitious free bytes.
            std::vector<bool> releasable(pages.capacity());
            for (const auto page : histories.released_pages(prefixes))
                releasable[pages.descriptor_index(page)] = true;

            struct PageOwners {
                PageId handle;
                std::uint32_t holders = none, position = 0, last_owner = 0;
            };

            std::vector<PageOwners> owners(pages.capacity());
            std::vector<std::vector<std::uint32_t>> holder_sets;
            std::unordered_map<std::uint64_t, std::uint32_t> merged_sets;
            for (const auto& history : histories_by_id) {
                if (history.holders.empty()) continue;
                std::vector<std::uint32_t> boundaries{0};
                for (const auto i : history.holders)
                    boundaries.push_back(kv_pages_for_frontier(
                        checkpoints[i].value->snapshot.prefix(pool).frontier()));
                std::sort(boundaries.begin(), boundaries.end());
                boundaries.erase(std::unique(boundaries.begin(), boundaries.end()),
                                 boundaries.end());
                for (std::size_t interval = 1; interval < boundaries.size(); ++interval) {
                    const auto begin = boundaries[interval - 1], end = boundaries[interval];
                    std::vector<std::uint32_t> holders;
                    for (const auto i : history.holders)
                        if (kv_pages_for_frontier(
                                checkpoints[i].value->snapshot.prefix(pool).frontier()) > begin)
                            holders.push_back(i);
                    const auto last_owner = holders.back();
                    const auto set        = static_cast<std::uint32_t>(holder_sets.size());
                    holder_sets.push_back(std::move(holders));
                    for (auto position = begin; position < end; ++position) {
                        const auto page = histories.logical_page(history.handle, position);
                        if (!(shortage.host_bytes && pages.host_resident(page)) &&
                            !(shortage.kv_page_groups[pool] && pages.device_resident(page)))
                            continue;
                        auto& entry  = owners[pages.descriptor_index(page)];
                        entry.handle = page;
                        if (entry.holders == none)
                            entry.holders = set;
                        else {
                            const auto key       = (std::uint64_t(entry.holders) << 32) | set;
                            auto [merged, added] = merged_sets.try_emplace(key, holder_sets.size());
                            if (added) {
                                std::vector<std::uint32_t> joined;
                                const auto& left  = holder_sets[entry.holders];
                                const auto& right = holder_sets[set];
                                std::set_union(left.begin(), left.end(), right.begin(), right.end(),
                                               std::back_inserter(joined));
                                holder_sets.push_back(std::move(joined));
                            }
                            entry.holders = merged->second;
                        }
                        if (last_owner >= entry.last_owner) {
                            entry.last_owner = last_owner;
                            entry.position   = position;
                        }
                    }
                }
            }
            for (std::uint32_t i = 0; i < owners.size(); ++i) {
                const auto& entry = owners[i];
                if (entry.holders == none || !permitted(holder_sets[entry.holders])) continue;
                auto& group = cohorts[holder_sets[entry.holders]];
                group.pages[pool].push_back({entry.handle, entry.position, releasable[i]});
                if (releasable[i]) {
                    if (shortage.kv_page_groups[pool])
                        group.released.kv_page_groups[pool] += pages.device_resident(entry.handle);
                    if (shortage.host_bytes && pages.host_resident(entry.handle)) {
                        group.released.host_bytes += storage.host_page_bytes;
                        group.host_extents.push_back(pages.host_extent(entry.handle));
                        ++group.host_pages;
                    }
                }
            }
        }
        for (auto& [indices, group] : cohorts) {
            for (const auto i : indices) group.sources.push_back(checkpoints.handle(i));
            for (auto& pages : group.pages)
                std::sort(pages.begin(), pages.end(),
                          [](const auto& a, const auto& b) { return a.position > b.position; });
            inventory->add(group.sources, group.released, std::move(group.host_extents),
                           group.host_pages);
            groups.push_back(std::move(group));
        }
    }
};

} // namespace ninfer::models::context
