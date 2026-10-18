#pragma once

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

    void insert(Session& session, Handle handle, std::uint32_t frontier) {
        insert_at(session, root_, handle, frontier);
    }

    void erase(Handle handle) { erase_at(root_, handle, true); }

    [[nodiscard]] std::vector<Handle> matches(const Session& session, const Base& base) const {
        std::vector<Handle> result;
        const Node* current = &root_;
        while (current) {
            result.insert(result.end(), current->checkpoints.begin(), current->checkpoints.end());
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

    void clear() noexcept { root_ = Node{}; }

private:
    struct Node {
        std::uint32_t frontier = 0;
        Handle representative{};
        std::vector<Handle> checkpoints;
        std::vector<std::unique_ptr<Node>> children;
    };

    static void insert_at(Session& session, Node& node, Handle handle, std::uint32_t frontier) {
        if (node.frontier == frontier) {
            if (std::find(node.checkpoints.begin(), node.checkpoints.end(), handle) ==
                node.checkpoints.end()) {
                node.checkpoints.push_back(handle);
            }
            node.representative = handle;
            return;
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
            insert_at(session, *child, handle, frontier);
            return;
        }
        auto child            = std::make_unique<Node>();
        child->frontier       = frontier;
        child->representative = handle;
        child->checkpoints.push_back(handle);
        node.children.push_back(std::move(child));
    }

    static bool erase_at(Node& node, Handle handle, bool root = false) {
        std::erase(node.checkpoints, handle);
        for (auto it = node.children.begin(); it != node.children.end();) {
            if (erase_at(**it, handle)) {
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
        node.representative = !node.checkpoints.empty() ? node.checkpoints.front()
                                                        : node.children.front()->representative;
        return false;
    }

    Node root_;
};
} // namespace ninfer::runtime
