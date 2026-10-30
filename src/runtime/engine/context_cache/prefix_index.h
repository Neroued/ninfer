#pragma once

#include "runtime/contract/resources.h"

#include <algorithm>
#include <cstdint>
#include <memory>
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

    void insert(Session& session, Handle handle, std::uint32_t frontier, CheckpointRole role) {
        if (insert_at(session, root_, handle, frontier, role)) {
            ++counters(role).retained;
            ++counters(role).published;
        }
    }

    void erase(Handle handle, bool reclaimed = false) { erase_at(root_, handle, reclaimed, true); }

    [[nodiscard]] const PrefixCacheStatistics& statistics() const noexcept { return statistics_; }

    [[nodiscard]] std::vector<Handle> matches(const Session& session, const Base& base) const {
        std::vector<Handle> result;
        const Node* current = &root_;
        while (current) {
            for (const auto& entry : current->checkpoints) result.push_back(entry.handle);
            const Node* next = nullptr;
            for (const auto& child : current->children) {
                if (child->frontier > base.summary().prompt_tokens) { continue; }
                const auto key = base.prefix_shortlist_key(child->frontier);
                if (key && *key == session.checkpoint_key(child->representative, child->frontier)) {
                    next = child.get();
                    break;
                }
            }
            current = next;
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
        root_ = Node{};
    }

private:
    struct Entry {
        Handle handle;
        CheckpointRole role;
    };

    struct Node {
        std::uint32_t frontier = 0;
        Handle representative{};
        std::vector<Entry> checkpoints;
        std::vector<std::unique_ptr<Node>> children;
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

    static bool insert_at(Session& session, Node& node, Handle handle, std::uint32_t frontier,
                          CheckpointRole role) {
        if (node.frontier == frontier) {
            const bool added =
                std::none_of(node.checkpoints.begin(), node.checkpoints.end(),
                             [&](const auto& entry) { return entry.handle == handle; });
            if (added) node.checkpoints.push_back({handle, role});
            node.representative = handle;
            return added;
        }
        for (auto& child : node.children) {
            if (session.checkpoint_key(child->representative, node.frontier + 1U) !=
                session.checkpoint_key(handle, node.frontier + 1U)) {
                continue;
            }
            std::uint32_t low = node.frontier + 1U, high = std::min(frontier, child->frontier);
            while (low < high) {
                const auto middle = low + (high - low + 1U) / 2U;
                if (session.checkpoint_key(child->representative, middle) ==
                    session.checkpoint_key(handle, middle)) {
                    low = middle;
                } else {
                    high = middle - 1U;
                }
            }
            if (low < child->frontier) {
                auto branch            = std::make_unique<Node>();
                branch->frontier       = low;
                branch->representative = child->representative;
                branch->children.push_back(std::move(child));
                child = std::move(branch);
            }
            return insert_at(session, *child, handle, frontier, role);
        }
        auto child            = std::make_unique<Node>();
        child->frontier       = frontier;
        child->representative = handle;
        child->checkpoints.push_back({handle, role});
        node.children.push_back(std::move(child));
        return true;
    }

    bool erase_at(Node& node, Handle handle, bool reclaimed, bool root = false) {
        std::erase_if(node.checkpoints, [&](const auto& entry) {
            if (entry.handle != handle) return false;
            auto& value = counters(entry.role);
            --value.retained;
            ++value.released;
            if (reclaimed) ++value.reclaimed;
            return true;
        });
        for (auto it = node.children.begin(); it != node.children.end();) {
            if (erase_at(**it, handle, reclaimed)) {
                it = node.children.erase(it);
            } else {
                ++it;
            }
        }
        if (root) { return false; }
        if (node.checkpoints.empty() && node.children.empty()) { return true; }
        if (node.checkpoints.empty() && node.children.size() == 1) {
            auto child = std::move(node.children.front());
            node       = std::move(*child);
        }
        node.representative = !node.checkpoints.empty() ? node.checkpoints.front().handle
                                                        : node.children.front()->representative;
        return false;
    }

    Node root_;
    PrefixCacheStatistics statistics_;
};
} // namespace ninfer::runtime
