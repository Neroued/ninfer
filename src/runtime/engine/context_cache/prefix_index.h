#pragma once

#include "runtime/contract/placement.h"
#include "runtime/contract/recovery.h"

#include <algorithm>
#include <functional>
#include <list>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ninfer::runtime {

template <class Key = PrefixKey>
struct PrefixIndexDelta {
    enum class Kind : std::uint8_t { Insert, Erase, Clear };
    Kind kind              = Kind::Insert;
    Key key                = {};
    std::uint32_t frontier = 0;
    PlacementId placement  = {};
};

using PrefixIndexObserver = std::function<void(const PrefixIndexDelta<PrefixKey>&)>;

// Runtime indexes cache visibility. Native owns exact semantic prefix matching and lifetime.
template <class Model>
class PrefixIndex {
public:
    using Handle   = typename Model::CheckpointHandle;
    using Session  = typename Model::Session;
    using Base     = typename Model::RequestBasePlan;
    using Key      = std::decay_t<decltype(std::declval<const Session&>().checkpoint_key(
        std::declval<Handle>()))>;
    using Delta    = PrefixIndexDelta<Key>;
    using Observer = std::function<void(const Delta&)>;

    PrefixIndex()                                  = default;
    PrefixIndex(const PrefixIndex&)                = delete;
    PrefixIndex& operator=(const PrefixIndex&)     = delete;
    PrefixIndex(PrefixIndex&&) noexcept            = default;
    PrefixIndex& operator=(PrefixIndex&&) noexcept = default;

    void set_observer(Observer observer) { observer_ = std::move(observer); }

    [[nodiscard]] std::size_t size() const noexcept { return entries_.size(); }

    void insert(Session& session, Handle handle, CheckpointRole role) {
        if (entries_.contains(handle)) return;
        const auto prefix   = session.checkpoint_prefix(handle);
        const auto key      = session.checkpoint_key(handle);
        const auto metadata = session.checkpoint_metadata(handle);
        auto& group         = prefixes_[prefix];
        group.push_back({handle, role});
        entries_.emplace(handle,
                         Location{prefix, key, metadata.frontier, metadata.placement,
                                  std::prev(group.end())});
        ++counters(role).retained;
        ++counters(role).published;
        if (observer_) {
            observer_({Delta::Kind::Insert, key, metadata.frontier, metadata.placement});
        }
    }

    void erase(Handle handle, bool reclaimed = false) {
        const auto found = entries_.find(handle);
        if (found == entries_.end()) return;
        const auto key       = found->second.key;
        const auto frontier  = found->second.frontier;
        const auto placement = found->second.placement;
        auto& value          = counters(found->second.entry->role);
        --value.retained;
        ++value.released;
        if (reclaimed) ++value.reclaimed;
        const auto group = prefixes_.find(found->second.prefix);
        group->second.erase(found->second.entry);
        if (group->second.empty()) prefixes_.erase(group);
        entries_.erase(found);
        if (observer_) { observer_({Delta::Kind::Erase, key, frontier, placement}); }
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
        const bool had_entries = !entries_.empty();
        for (auto* value : {&statistics_.input_replay, &statistics_.long_anchor,
                            &statistics_.shared_prefix, &statistics_.continuation}) {
            value->released += value->retained;
            value->retained = 0;
        }
        entries_.clear();
        prefixes_.clear();
        if (had_entries && observer_) { observer_({Delta::Kind::Clear}); }
    }
private:
    struct Entry {
        Handle handle;
        CheckpointRole role;
    };

    struct Location {
        PrefixId prefix;
        Key key;
        std::uint32_t frontier = 0;
        PlacementId placement  = {};
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
    Observer observer_;
};
} // namespace ninfer::runtime
