#pragma once

#include "models/common/context/checkpoint_store.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ninfer::models::context {

// A shortlist only. Index construction always obtains an exact prefix proof from its view.
struct RecoveryKey {
    std::array<std::uint64_t, 2> digests{};
    std::uint32_t domain                                           = 0;
    friend bool operator==(const RecoveryKey&, const RecoveryKey&) = default;
};

template <class Handle>
class RecoveryEvaluation;

// Immutable checkpoint identities form a forest; equal identities/frontiers share a node, not
// physical State/KV ownership. No node owns checkpoint content or pins a store replica.
template <class Handle = CheckpointHandle>
class RecoveryIndex {
public:
    static constexpr auto none = std::numeric_limits<std::uint32_t>::max();

    struct Work {
        std::uint64_t synchronizations  = 0;
        std::uint64_t shortlist_queries = 0;
        std::uint64_t exact_comparisons = 0;
    };

    // View supplies revision() by const reference, capacity(), handle(index), valid(handle),
    // frontier(handle), key(handle, frontier), and prefix(earlier, later). Identity is immutable
    // between publication and retirement; a changed generation denotes new content.
    template <class View>
    void synchronize(const View& view) {
        if (revision_ == &view.revision() && observed_revision_ == view.revision()) return;
        if (revision_ && revision_ != &view.revision())
            throw std::logic_error("recovery index changed checkpoint owner");
        revision_ = &view.revision();
        ++work_.synchronizations;
        try {
            slots_.resize(view.capacity());
            for (std::uint32_t i = 0; i < slots_.size(); ++i)
                if (slots_[i].node != none && !view.valid(slots_[i].handle)) erase(i);
            for (std::uint32_t i = 0; i < slots_.size(); ++i) {
                const auto handle = view.handle(i);
                if (slots_[i].node == none && view.valid(handle)) insert(handle, view);
            }
            number_nodes();
            observed_revision_ = view.revision();
            ++epoch_;
        } catch (...) {
            slots_.clear();
            nodes_.assign(1, {});
            free_.clear();
            depths_.clear();
            preorder_.clear();
            observed_revision_ = 0;
            ++epoch_;
            throw;
        }
    }

    [[nodiscard]] const Work& work() const noexcept { return work_; }

    [[nodiscard]] RecoveryEvaluation<Handle> evaluate(std::span<const Handle> advertised) const {
        return RecoveryEvaluation<Handle>(*this, advertised);
    }

private:
    struct Node {
        std::uint32_t frontier = 0, parent = 0, sibling = 0, component = 0;
        std::uint32_t begin = 0, end = 0;
        RecoveryKey key;
        std::vector<Handle> aliases;
        std::vector<std::uint32_t> children;
    };

    struct Slot {
        Handle handle{};
        std::uint32_t node = none, alias = 0;
    };

    struct KeyHash {
        std::size_t operator()(const RecoveryKey& key) const noexcept {
            auto hash = key.digests[0] ^ (key.digests[1] + 0x9e3779b97f4a7c15ULL +
                                          (key.digests[0] << 6) + (key.digests[0] >> 2));
            return hash ^ (std::uint64_t(key.domain) * 0x9e3779b97f4a7c15ULL);
        }
    };

    using Buckets = std::unordered_map<RecoveryKey, std::vector<std::uint32_t>, KeyHash>;

    void check() const {
        if (!revision_ || *revision_ != observed_revision_)
            throw std::logic_error("recovery evaluation outlived its checkpoint revision");
    }

    std::uint32_t node(Handle handle) const {
        if (handle.index >= slots_.size()) return none;
        const auto& slot = slots_[handle.index];
        return slot.node != none && slot.handle == handle ? slot.node : none;
    }

    void detach(std::uint32_t child) {
        auto& node             = nodes_[child];
        auto& siblings         = nodes_[node.parent].children;
        const auto moved       = siblings.back();
        siblings[node.sibling] = moved;
        nodes_[moved].sibling  = node.sibling;
        siblings.pop_back();
    }

    void attach(std::uint32_t child, std::uint32_t parent) {
        nodes_[child].parent  = parent;
        nodes_[child].sibling = nodes_[parent].children.size();
        nodes_[parent].children.push_back(child);
    }

    void erase(std::uint32_t slot_index) {
        auto& slot                = slots_[slot_index];
        const auto index          = slot.node;
        auto& node                = nodes_[index];
        const auto moved          = node.aliases.back();
        node.aliases[slot.alias]  = moved;
        slots_[moved.index].alias = slot.alias;
        node.aliases.pop_back();
        slot.node = none;
        if (!node.aliases.empty()) return;
        const auto parent = node.parent;
        detach(index);
        for (const auto child : node.children) attach(child, parent);
        auto depth  = depths_.find(node.frontier);
        auto bucket = depth->second.find(node.key);
        std::erase(bucket->second, index);
        if (bucket->second.empty()) depth->second.erase(bucket);
        if (depth->second.empty()) depths_.erase(depth);
        node = {};
        free_.push_back(index);
    }

    template <class View>
    RecoveryKey key(const View& view, Handle handle, std::uint32_t frontier) {
        ++work_.shortlist_queries;
        return view.key(handle, frontier);
    }

    template <class View>
    bool prefix(const View& view, Handle earlier, Handle later) {
        ++work_.exact_comparisons;
        return view.prefix(earlier, later);
    }

    template <class View>
    void insert(Handle handle, const View& view) {
        const auto frontier = view.frontier(handle);
        const auto identity = key(view, handle, frontier);
        if (const auto depth = depths_.find(frontier); depth != depths_.end()) {
            if (const auto found = depth->second.find(identity); found != depth->second.end())
                for (const auto index : found->second)
                    if (prefix(view, nodes_[index].aliases.back(), handle)) {
                        slots_[handle.index] = {
                            handle, index,
                            static_cast<std::uint32_t>(nodes_[index].aliases.size())};
                        nodes_[index].aliases.push_back(handle);
                        return;
                    }
        }
        std::uint32_t parent = 0;
        auto depth           = depths_.lower_bound(frontier);
        while (depth != depths_.begin() && !parent) {
            --depth;
            const auto found = depth->second.find(key(view, handle, depth->first));
            if (found == depth->second.end()) continue;
            for (const auto index : found->second)
                if (prefix(view, nodes_[index].aliases.back(), handle)) {
                    parent = index;
                    break;
                }
        }
        const auto index = free_.empty() ? static_cast<std::uint32_t>(nodes_.size()) : free_.back();
        if (free_.empty())
            nodes_.emplace_back();
        else
            free_.pop_back();
        nodes_[index].frontier = frontier;
        nodes_[index].key      = identity;
        nodes_[index].aliases.push_back(handle);
        slots_[handle.index] = {handle, index, 0};
        // Only direct children of the nearest ancestor can need reparenting. Exact checks on
        // shortlist matches keep digest collisions from inventing an ancestry relationship.
        auto children = nodes_[parent].children;
        for (const auto child : children) {
            if (nodes_[child].frontier <= frontier) continue;
            const auto representative = nodes_[child].aliases.back();
            if (key(view, representative, frontier) == identity &&
                prefix(view, handle, representative)) {
                detach(child);
                attach(child, index);
            }
        }
        attach(index, parent);
        depths_[frontier][identity].push_back(index);
    }

    void number_nodes() {
        preorder_.clear();
        std::uint32_t clock = 0;
        std::vector<std::pair<std::uint32_t, bool>> stack{{0, false}};
        while (!stack.empty()) {
            const auto [index, closing] = stack.back();
            stack.pop_back();
            auto& node = nodes_[index];
            if (closing) {
                node.end = clock;
                continue;
            }
            node.begin     = clock++;
            node.component = node.parent ? nodes_[node.parent].component : index;
            preorder_.push_back(index);
            stack.emplace_back(index, true);
            for (auto child = node.children.rbegin(); child != node.children.rend(); ++child)
                stack.emplace_back(*child, false);
        }
    }

    const std::uint64_t* revision_   = nullptr;
    std::uint64_t observed_revision_ = 0, epoch_ = 0;
    std::vector<Node> nodes_{1};
    std::vector<Slot> slots_;
    std::vector<std::uint32_t> free_, preorder_;
    std::map<std::uint32_t, Buckets> depths_;
    Work work_;
    friend class RecoveryEvaluation<Handle>;
};

// One synchronous, read-only view of advertised recovery points. Waiting-only physical holders
// may be removed, but do not become fallback unless explicitly advertised by Runtime.
template <class Handle = CheckpointHandle>
class RecoveryEvaluation {
public:
    using Index                = RecoveryIndex<Handle>;
    static constexpr auto none = Index::none;

    struct Work {
        std::uint64_t queries = 0, removed_handles = 0, fallback_steps = 0;
    };

    RecoveryEvaluation(const Index& index, std::span<const Handle> advertised)
        : index_(&index), epoch_(index.epoch_), advertised_(index.slots_.size()),
          available_(index.nodes_.size()), nearest_(index.nodes_.size()),
          removed_(index.nodes_.size()), memo_(index.nodes_.size()),
          handle_stamp_(index.slots_.size()), node_stamp_(index.nodes_.size()),
          memo_stamp_(index.nodes_.size()), score_stamp_(index.nodes_.size()) {
        check();
        for (const auto handle : advertised) {
            const auto node = index.node(handle);
            if (node != none && !advertised_[handle.index]) {
                advertised_[handle.index] = true;
                ++available_[node];
            }
        }
        for (const auto node : index.preorder_)
            nearest_[node] = available_[node] ? node : nearest_[index.nodes_[node].parent];
    }

    [[nodiscard]] bool contains(Handle handle) const {
        check();
        return index_->node(handle) != none;
    }

    [[nodiscard]] std::uint32_t frontier(Handle handle) const {
        check();
        return index_->nodes_[require_node(handle)].frontier;
    }

    [[nodiscard]] std::uint32_t component(Handle handle) const {
        check();
        return index_->nodes_[require_node(handle)].component;
    }

    [[nodiscard]] bool prefix(Handle earlier, Handle later) const {
        check();
        const auto a = index_->node(earlier), b = index_->node(later);
        if (a == none || b == none) return false;
        const auto& x = index_->nodes_[a];
        const auto& y = index_->nodes_[b];
        return x.begin <= y.begin && y.begin < x.end;
    }

    [[nodiscard]] std::uint64_t loss(std::span<const Handle> removed,
                                     std::span<const Handle> unavailable = {}) {
        prepare(removed, unavailable);
        std::sort(groups_.begin(), groups_.end(), [&](auto a, auto b) {
            return index_->nodes_[a].begin < index_->nodes_[b].begin;
        });

        // The stack is the virtual forest induced by removed nodes. Propagate the maximum
        // degradation along each path; count it only at that path's deepest removed node.
        struct Path {
            std::uint32_t node, worst;
            bool leaf;
        };

        std::vector<Path> stack;
        std::uint64_t total = 0;
        for (const auto node : groups_) {
            const auto& record = index_->nodes_[node];
            while (!stack.empty() && index_->nodes_[stack.back().node].end <= record.begin) {
                if (stack.back().leaf) total += stack.back().worst;
                stack.pop_back();
            }
            auto worst = record.frontier - fallback(node);
            if (!stack.empty()) {
                worst             = std::max(worst, stack.back().worst);
                stack.back().leaf = false;
            }
            stack.push_back({node, worst, true});
        }
        for (const auto& path : stack)
            if (path.leaf) total += path.worst;
        return total;
    }

    // Candidates are prepared once in decreasing frontier order by the admission's exact
    // identity query. No token comparison or full surviving-set reconstruction occurs here.
    [[nodiscard]] std::uint32_t best_remaining(std::span<const Handle> candidates,
                                               std::span<const Handle> removed) {
        prepare(removed);
        for (const auto handle : candidates)
            if (index_->node(handle) != none && advertised_[handle.index] &&
                handle_stamp_[handle.index] != stamp_)
                return frontier(handle);
        return 0;
    }

    [[nodiscard]] std::size_t capacity() const {
        check();
        return index_->slots_.size();
    }

    [[nodiscard]] std::size_t component_capacity() const {
        check();
        return index_->nodes_.size();
    }

    [[nodiscard]] const Work& work() const noexcept { return work_; }

private:
    void check() const {
        index_->check();
        if (index_->epoch_ != epoch_)
            throw std::logic_error("recovery index changed during evaluation");
    }

    std::uint32_t require_node(Handle handle) const {
        const auto node = index_->node(handle);
        if (node == none) throw std::logic_error("recovery query has a stale checkpoint");
        return node;
    }

    void prepare(std::span<const Handle> handles, std::span<const Handle> unavailable = {}) {
        check();
        ++work_.queries;
        if (++stamp_ == 0) {
            std::fill(handle_stamp_.begin(), handle_stamp_.end(), 0);
            std::fill(node_stamp_.begin(), node_stamp_.end(), 0);
            std::fill(memo_stamp_.begin(), memo_stamp_.end(), 0);
            std::fill(score_stamp_.begin(), score_stamp_.end(), 0);
            ++stamp_;
        }
        groups_.clear();
        const auto mark = [&](Handle handle, bool score) {
            const auto node = index_->node(handle);
            if (node == none) return;
            if (score && score_stamp_[node] != stamp_) {
                score_stamp_[node] = stamp_;
                groups_.push_back(node);
            }
            if (handle_stamp_[handle.index] == stamp_) return;
            ++work_.removed_handles;
            handle_stamp_[handle.index] = stamp_;
            if (node_stamp_[node] != stamp_) {
                node_stamp_[node] = stamp_;
                removed_[node]    = 0;
            }
            removed_[node] += advertised_[handle.index];
        };
        for (const auto handle : unavailable) mark(handle, false);
        for (const auto handle : handles) mark(handle, true);
    }

    std::uint32_t fallback(std::uint32_t node) {
        std::vector<std::uint32_t> path;
        node = nearest_[node];
        while (node && node_stamp_[node] == stamp_ && removed_[node] == available_[node] &&
               memo_stamp_[node] != stamp_) {
            ++work_.fallback_steps;
            path.push_back(node);
            node = nearest_[index_->nodes_[node].parent];
        }
        const auto result =
            memo_stamp_[node] == stamp_ ? memo_[node] : index_->nodes_[node].frontier;
        for (const auto skipped : path) {
            memo_stamp_[skipped] = stamp_;
            memo_[skipped]       = result;
        }
        return result;
    }

    const Index* index_;
    std::uint64_t epoch_, stamp_ = 0;
    std::vector<bool> advertised_;
    std::vector<std::uint32_t> available_, nearest_, removed_, memo_, groups_;
    std::vector<std::uint64_t> handle_stamp_, node_stamp_, memo_stamp_, score_stamp_;
    Work work_;
};

} // namespace ninfer::models::context
