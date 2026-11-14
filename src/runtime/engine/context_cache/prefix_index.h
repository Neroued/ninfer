#pragma once

#include "runtime/contract/resources.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <list>
#include <map>
#include <unordered_map>
#include <utility>
#include <optional>
#include <vector>

namespace ninfer::runtime {

// Compressed edges refer to a surviving native checkpoint's semantic digest path. The model
// verifies exact token/position/media identity before a returned checkpoint can be used.
template <class Model>
class PrefixIndex {
public:
    using Handle  = typename Model::CheckpointHandle;
    using Session = typename Model::Session;
    using Base    = typename Model::RequestBasePlan;

    using Key = typename decltype(std::declval<const Base&>().prefix_shortlist_key(0))::value_type;

    PrefixIndex()                                  = default;
    PrefixIndex(const PrefixIndex&)                = delete;
    PrefixIndex& operator=(const PrefixIndex&)     = delete;
    PrefixIndex(PrefixIndex&&) noexcept            = default;
    PrefixIndex& operator=(PrefixIndex&&) noexcept = default;

    void insert(Session& session, Handle handle, std::uint32_t frontier, CheckpointRole role) {
        if (entries_.contains(handle)) return;
        auto& node = insert_at(session, *root_, handle, frontier);
        node.checkpoints.push_back({handle, role});
        entries_.emplace(handle, Location{&node, std::prev(node.checkpoints.end())});
        ++counters(role).retained;
        ++counters(role).published;
    }

    void erase(Handle handle, bool reclaimed = false) {
        const auto found = entries_.find(handle);
        if (found == entries_.end()) return;
        auto* node  = found->second.node;
        auto& value = counters(found->second.entry->role);
        --value.retained;
        ++value.released;
        if (reclaimed) ++value.reclaimed;
        node->checkpoints.erase(found->second.entry);
        entries_.erase(found);
        while (node != root_.get()) {
            auto* parent    = node->parent;
            const auto edge = node->edge;
            if (node->checkpoints.empty() && node->children.empty()) {
                parent->children.erase(edge);
            } else if (node->checkpoints.empty() && node->children.size() == 1) {
                auto child                = std::move(node->children.begin()->second);
                child->parent             = parent;
                child->edge               = edge;
                parent->children.at(edge) = std::move(child);
            } else {
                node->representative = node->checkpoints.empty()
                                           ? node->children.begin()->second->representative
                                           : node->checkpoints.front().handle;
            }
            node = parent;
        }
    }

    [[nodiscard]] const PrefixCacheStatistics& statistics() const noexcept { return statistics_; }

    void record_adoption(CheckpointRole role, std::uint64_t tokens) noexcept {
        auto& value = counters(role);
        ++value.adopted;
        value.reused_tokens += tokens;
    }

    [[nodiscard]] std::vector<Handle> matches(const Session& session, const Base& base) const {
        std::vector<Handle> result;
        const Node* current = root_.get();
        while (current) {
            for (const auto& entry : current->checkpoints) result.push_back(entry.handle);
            if (current->frontier >= base.summary().prompt_tokens) break;
            const auto edge = base.prefix_shortlist_key(current->frontier + 1);
            if (!edge) break;
            const auto found = current->children.find(*edge);
            if (found == current->children.end()) break;
            const auto* child = found->second.get();
            if (child->frontier > base.summary().prompt_tokens) break;
            const auto key = base.prefix_shortlist_key(child->frontier);
            if (!key || *key != session.checkpoint_key(child->representative, child->frontier))
                break;
            current = child;
        }
        std::reverse(result.begin(), result.end());
        return result;
    }

    void clear() noexcept {
        for (auto* value : {&statistics_.input_replay, &statistics_.long_anchor,
                            &statistics_.shared_prefix, &statistics_.continuation}) {
            value->released += value->retained;
            value->retained = 0;
        }
        entries_.clear();
        *root_ = Node{};
    }

private:
    struct Entry {
        Handle handle;
        CheckpointRole role;
    };

    struct Node {
        std::uint32_t frontier = 0;
        Handle representative{};
        Node* parent = nullptr;
        Key edge;
        std::list<Entry> checkpoints;
        std::map<Key, std::unique_ptr<Node>> children;
    };

    struct Location {
        Node* node;
        typename std::list<Entry>::iterator entry;
    };

    struct HandleHash {
        std::size_t operator()(Handle h) const noexcept { return h.index; }
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

    static Node& insert_at(Session& session, Node& node, Handle handle, std::uint32_t frontier) {
        if (node.frontier == frontier) return node;
        const auto edge  = session.checkpoint_key(handle, node.frontier + 1U);
        const auto found = node.children.find(edge);
        if (found != node.children.end()) {
            auto& child       = found->second;
            std::uint32_t low = node.frontier + 1U, high = std::min(frontier, child->frontier);
            while (low < high) {
                const auto middle = low + (high - low + 1U) / 2U;
                if (session.checkpoint_key(child->representative, middle) ==
                    session.checkpoint_key(handle, middle))
                    low = middle;
                else
                    high = middle - 1U;
            }
            if (low < child->frontier) {
                auto branch            = std::make_unique<Node>();
                branch->frontier       = low;
                branch->representative = child->representative;
                branch->parent         = &node;
                branch->edge           = edge;
                child->parent          = branch.get();
                child->edge            = session.checkpoint_key(child->representative, low + 1U);
                const auto suffix      = child->edge;
                branch->children.emplace(suffix, std::move(child));
                child = std::move(branch);
            }
            return insert_at(session, *child, handle, frontier);
        }
        auto child            = std::make_unique<Node>();
        child->frontier       = frontier;
        child->representative = handle;
        child->parent         = &node;
        child->edge           = edge;
        return *node.children.emplace(edge, std::move(child)).first->second;
    }

    std::unique_ptr<Node> root_ = std::make_unique<Node>();
    std::unordered_map<Handle, Location, HandleHash> entries_;
    PrefixCacheStatistics statistics_;
};
} // namespace ninfer::runtime
