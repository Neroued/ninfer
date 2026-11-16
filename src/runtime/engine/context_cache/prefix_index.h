#pragma once

#include "runtime/contract/recovery.h"

#include <algorithm>
#include <list>
#include <unordered_map>
#include <vector>

namespace ninfer::runtime {

// Runtime indexes cache visibility. Native owns exact semantic prefix matching and lifetime.
template <class Model>
class PrefixIndex {
public:
    using Handle  = typename Model::CheckpointHandle;
    using Session = typename Model::Session;
    using Base    = typename Model::RequestBasePlan;

    PrefixIndex()                                  = default;
    PrefixIndex(const PrefixIndex&)                = delete;
    PrefixIndex& operator=(const PrefixIndex&)     = delete;
    PrefixIndex(PrefixIndex&&) noexcept            = default;
    PrefixIndex& operator=(PrefixIndex&&) noexcept = default;

    void insert(Session& session, Handle handle, CheckpointRole role) {
        if (entries_.contains(handle)) return;
        const auto prefix = session.checkpoint_prefix(handle);
        auto& group       = prefixes_[prefix];
        group.push_back({handle, role});
        entries_.emplace(handle, Location{prefix, std::prev(group.end())});
        ++counters(role).retained;
        ++counters(role).published;
    }

    void erase(Handle handle, bool reclaimed = false) {
        const auto found = entries_.find(handle);
        if (found == entries_.end()) return;
        auto& value = counters(found->second.entry->role);
        --value.retained;
        ++value.released;
        if (reclaimed) ++value.reclaimed;
        const auto group = prefixes_.find(found->second.prefix);
        group->second.erase(found->second.entry);
        if (group->second.empty()) prefixes_.erase(group);
        entries_.erase(found);
    }

    [[nodiscard]] const PrefixCacheStatistics& statistics() const noexcept { return statistics_; }

    void record_adoption(CheckpointRole role, std::uint64_t tokens) noexcept {
        auto& value = counters(role);
        ++value.adopted;
        value.reused_tokens += tokens;
    }

    [[nodiscard]] std::vector<Handle> matches(const Session& session, const Base& base) const {
        std::vector<Handle> result;
        for (const auto prefix : session.matching_prefixes(base)) {
            const auto found = prefixes_.find(prefix);
            if (found == prefixes_.end()) continue;
            for (auto entry = found->second.rbegin(); entry != found->second.rend(); ++entry)
                result.push_back(entry->handle);
        }
        return result;
    }

    void clear() noexcept {
        for (auto* value : {&statistics_.input_replay, &statistics_.long_anchor,
                            &statistics_.shared_prefix, &statistics_.continuation}) {
            value->released += value->retained;
            value->retained = 0;
        }
        entries_.clear();
        prefixes_.clear();
    }
private:
    struct Entry {
        Handle handle;
        CheckpointRole role;
    };

    struct Location {
        PrefixId prefix;
        typename std::list<Entry>::iterator entry;
    };

    struct HandleHash {
        std::size_t operator()(Handle h) const noexcept { return h.index; }
    };

    struct PrefixHash {
        std::size_t operator()(PrefixId h) const noexcept {
            return h.index ^ (h.generation * 0x9e3779b97f4a7c15ULL);
        }
    };

    PrefixCacheCounters& counters(CheckpointRole role) noexcept {
        switch (role) {
        case CheckpointRole::InputReplay:
            return statistics_.input_replay;
        case CheckpointRole::LongAnchor:
            return statistics_.long_anchor;
        case CheckpointRole::SharedPrefix:
            return statistics_.shared_prefix;
        case CheckpointRole::Continuation:
            return statistics_.continuation;
        }
        std::terminate();
    }

    std::unordered_map<Handle, Location, HandleHash> entries_;
    std::unordered_map<PrefixId, std::list<Entry>, PrefixHash> prefixes_;
    PrefixCacheStatistics statistics_;
};
} // namespace ninfer::runtime
