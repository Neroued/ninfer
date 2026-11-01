#pragma once

#include "models/common/context/checkpoint_store.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
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

// Immutable checkpoint identities form a forest. Flat links and hash buckets keep all storage
// bounded by the prepared checkpoint extent; no node owns State/KV or allocates child containers.
template <class Handle = CheckpointHandle>
class RecoveryIndex : public MetadataCapacity {
public:
    static constexpr auto none = std::numeric_limits<std::uint32_t>::max();

    explicit RecoveryIndex(MetadataBudget& budget) : metadata_(budget), scratch_(budget) {}

    [[nodiscard]] static std::size_t metadata_bytes(std::size_t capacity) {
        capacity += 255; // Checkpoint descriptors grow by slabs of 256.
        return sizeof(RecoveryIndex) + capacity * sizeof(Slot) +
               (capacity + 1) * (sizeof(Node) + 2 * sizeof(std::uint32_t)) +
               capacity * sizeof(Depth) + 4 * (capacity + 1) * sizeof(std::uint32_t) +
               capacity * (sizeof(std::uint8_t) + sizeof(std::uint64_t)) +
               (capacity + 1) *
                   (3 * sizeof(std::uint64_t) + 6 * sizeof(std::uint32_t) + sizeof(LossPath)) +
               11 * 7;
    }

    [[nodiscard]] std::size_t metadata_requirement(std::size_t capacity) const override {
        std::size_t reallocation = 0;
        const auto growth        = [&](const auto& values, std::size_t count) {
            using Value = typename std::decay_t<decltype(values)>::value_type;
            if (count > values.capacity())
                reallocation = std::max(reallocation, values.capacity() * sizeof(Value));
            return count > values.capacity() ? (count - values.capacity()) * sizeof(Value) : 0;
        };
        const auto retained = growth(slots_, capacity) + growth(nodes_, capacity + 1) +
                              growth(free_, capacity + 1) + growth(preorder_, capacity + 1) +
                              growth(depths_, capacity) +
                              growth(buckets_, hash_capacity(capacity)) +
                              scratch_.growth_bytes(evaluation_bytes(capacity, capacity + 1));
        return retained + reallocation;
    }

    void reserve_metadata(std::size_t capacity) override {
        if (capacity < slots_.size()) return;
        if (capacity >= none) throw std::length_error("recovery index exceeds uint32");
        metadata_.reserve(slots_, capacity);
        slots_.resize(capacity);
        metadata_.reserve(nodes_, capacity + 1);
        if (nodes_.empty()) nodes_.emplace_back();
        metadata_.reserve(free_, capacity + 1);
        metadata_.reserve(preorder_, capacity + 1);
        metadata_.reserve(depths_, capacity);
        const auto hash_size = hash_capacity(capacity);
        if (buckets_.size() < hash_size) {
            metadata_.reserve(buckets_, hash_size);
            buckets_.assign(hash_size, none);
            for (std::uint32_t i = 1; i < nodes_.size(); ++i)
                if (nodes_[i].first_alias != none) link_bucket(i);
        }
        scratch_.reserve(evaluation_bytes(capacity, capacity + 1));
    }

    struct Work {
        std::uint64_t synchronizations = 0, shortlist_queries = 0, exact_comparisons = 0;
    };

    // View supplies immutable identities and an exact prefix proof. The producer reserves index
    // capacity before publication; synchronize only updates links within that prepared extent.
    template <class View>
    void synchronize(const View& view) {
        if (revision_ == &view.revision() && observed_revision_ == view.revision()) return;
        if (revision_ && revision_ != &view.revision())
            throw std::logic_error("recovery index changed checkpoint owner");
        revision_ = &view.revision();
        ++work_.synchronizations;
        reserve_metadata(view.capacity());
        try {
            for (std::uint32_t i = 0; i < slots_.size(); ++i)
                if (slots_[i].node != none && !view.valid(slots_[i].handle)) erase(i);
            for (std::uint32_t i = 0; i < view.capacity(); ++i) {
                const auto handle = view.handle(i);
                if (slots_[i].node == none && view.valid(handle)) insert(handle, view);
            }
            number_nodes();
            observed_revision_ = view.revision();
            ++epoch_;
        } catch (...) {
            std::fill(slots_.begin(), slots_.end(), Slot{});
            nodes_.assign(1, {});
            free_.clear();
            depths_.clear();
            preorder_.clear();
            std::fill(buckets_.begin(), buckets_.end(), none);
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
        std::uint32_t frontier = 0, parent = 0, first_child = none;
        std::uint32_t previous = none, next = none, component = 0, begin = 0, end = 0;
        RecoveryKey key;
        std::uint32_t first_alias = none, bucket_next = none;
    };

    struct Slot {
        Handle handle{};
        std::uint32_t node = none, previous = none, next = none;
    };

    struct Depth {
        std::uint32_t frontier = 0, count = 0;
    };

    struct LossPath {
        std::uint32_t node, worst;
        bool leaf;
    };

    static std::size_t hash_capacity(std::size_t capacity) {
        std::size_t size = 2;
        while (size < 2 * (capacity + 1)) size *= 2;
        return size;
    }

    static std::size_t evaluation_bytes(std::size_t slots, std::size_t nodes) {
        // Four stamp arrays, five counters/lists, traversal path and loss stack. Each array
        // starts on an eight-byte boundary, including the byte-per-slot advertised flags.
        const auto aligned = [](std::size_t n) { return (n + 7) / 8 * 8; };
        return aligned(slots) + aligned(slots * sizeof(std::uint64_t)) +
               3 * aligned(nodes * sizeof(std::uint64_t)) +
               6 * aligned(nodes * sizeof(std::uint32_t)) + aligned(nodes * sizeof(LossPath));
    }

    std::size_t bucket(std::uint32_t frontier, const RecoveryKey& key) const {
        auto h = key.digests[0] ^ (key.digests[1] + 0x9e3779b97f4a7c15ULL + (key.digests[0] << 6) +
                                   (key.digests[0] >> 2));
        h ^= (std::uint64_t(key.domain) << 32) | frontier;
        h ^= h >> 30;
        h *= 0xbf58476d1ce4e5b9ULL;
        return (h ^ (h >> 27)) & (buckets_.size() - 1);
    }

    void link_bucket(std::uint32_t index) {
        auto& head                = buckets_[bucket(nodes_[index].frontier, nodes_[index].key)];
        nodes_[index].bucket_next = head;
        head                      = index;
    }

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
        const auto& node = nodes_[child];
        if (node.previous != none)
            nodes_[node.previous].next = node.next;
        else
            nodes_[node.parent].first_child = node.next;
        if (node.next != none) nodes_[node.next].previous = node.previous;
    }

    void attach(std::uint32_t child, std::uint32_t parent) {
        auto& node    = nodes_[child];
        node.parent   = parent;
        node.previous = none;
        node.next     = nodes_[parent].first_child;
        if (node.next != none) nodes_[node.next].previous = child;
        nodes_[parent].first_child = child;
    }

    void add_alias(std::uint32_t index, Handle handle) {
        auto& first          = nodes_[index].first_alias;
        slots_[handle.index] = {handle, index, none, first};
        if (first != none) slots_[first].previous = handle.index;
        first = handle.index;
    }

    Handle representative(std::uint32_t index) const {
        return slots_[nodes_[index].first_alias].handle;
    }

    auto depth(std::uint32_t frontier) {
        return std::lower_bound(
            depths_.begin(), depths_.end(), frontier,
            [](const Depth& entry, auto value) { return entry.frontier < value; });
    }

    void erase(std::uint32_t index) {
        auto& slot    = slots_[index];
        const auto id = slot.node;
        auto& node    = nodes_[id];
        if (slot.previous != none)
            slots_[slot.previous].next = slot.next;
        else
            node.first_alias = slot.next;
        if (slot.next != none) slots_[slot.next].previous = slot.previous;
        slot = {};
        if (node.first_alias != none) return;
        detach(id);
        for (auto child = node.first_child; child != none;) {
            const auto next = nodes_[child].next;
            attach(child, node.parent);
            child = next;
        }
        auto* link = &buckets_[bucket(node.frontier, node.key)];
        while (*link != id) link = &nodes_[*link].bucket_next;
        *link  = node.bucket_next;
        auto d = depth(node.frontier);
        if (!--d->count) depths_.erase(d);
        node = {};
        free_.push_back(id);
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
    std::uint32_t find(const View& view, Handle handle, std::uint32_t frontier,
                       const RecoveryKey& key) {
        for (auto id = buckets_[bucket(frontier, key)]; id != none; id = nodes_[id].bucket_next)
            if (nodes_[id].frontier == frontier && nodes_[id].key == key &&
                prefix(view, representative(id), handle))
                return id;
        return none;
    }

    template <class View>
    void insert(Handle handle, const View& view) {
        const auto frontier = view.frontier(handle);
        const auto identity = key(view, handle, frontier);
        if (const auto equal = find(view, handle, frontier, identity); equal != none) {
            add_alias(equal, handle);
            return;
        }
        std::uint32_t parent = 0;
        auto d               = depth(frontier);
        while (d != depths_.begin() && !parent) {
            --d;
            const auto found = find(view, handle, d->frontier, key(view, handle, d->frontier));
            if (found != none) parent = found;
        }
        const auto index = free_.empty() ? static_cast<std::uint32_t>(nodes_.size()) : free_.back();
        if (free_.empty())
            nodes_.emplace_back();
        else
            free_.pop_back();
        nodes_[index].frontier = frontier;
        nodes_[index].key      = identity;
        add_alias(index, handle);
        for (auto child = nodes_[parent].first_child; child != none;) {
            const auto next = nodes_[child].next;
            if (nodes_[child].frontier > frontier) {
                const auto other = representative(child);
                if (key(view, other, frontier) == identity && prefix(view, handle, other)) {
                    detach(child);
                    attach(child, index);
                }
            }
            child = next;
        }
        attach(index, parent);
        link_bucket(index);
        d = depth(frontier);
        if (d == depths_.end() || d->frontier != frontier) d = depths_.insert(d, {frontier, 0});
        ++d->count;
    }

    void number_nodes() {
        preorder_.clear();
        std::uint32_t clock = 0, id = 0;
        for (;;) {
            auto& node     = nodes_[id];
            node.begin     = clock++;
            node.component = node.parent ? nodes_[node.parent].component : id;
            preorder_.push_back(id);
            if (node.first_child != none) {
                id = node.first_child;
                continue;
            }
            for (;;) {
                nodes_[id].end = clock;
                if (!id) return;
                if (nodes_[id].next != none) {
                    id = nodes_[id].next;
                    break;
                }
                id = nodes_[id].parent;
            }
        }
    }

    MetadataAccount metadata_;
    mutable MetadataScratch scratch_;
    const std::uint64_t* revision_   = nullptr;
    std::uint64_t observed_revision_ = 0, epoch_ = 0;
    std::vector<Node> nodes_;
    std::vector<Slot> slots_;
    std::vector<std::uint32_t> free_, preorder_, buckets_;
    std::vector<Depth> depths_;
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
        : storage_(index.scratch_.acquire(
              Index::evaluation_bytes(index.slots_.size(), index.nodes_.size()))),
          index_(&index), epoch_(index.epoch_) {
        auto* cursor    = storage_.data();
        const auto take = [&]<class T>(std::size_t count) {
            auto values = std::span<T>(reinterpret_cast<T*>(cursor), count);
            cursor += (count * sizeof(T) + 7) / 8 * 8;
            std::fill(values.begin(), values.end(), T{});
            return values;
        };
        const auto slots = index.slots_.size(), nodes = index.nodes_.size();
        advertised_     = take.template operator()<std::uint8_t>(slots);
        handle_stamp_   = take.template operator()<std::uint64_t>(slots);
        node_stamp_     = take.template operator()<std::uint64_t>(nodes);
        memo_stamp_     = take.template operator()<std::uint64_t>(nodes);
        score_stamp_    = take.template operator()<std::uint64_t>(nodes);
        available_      = take.template operator()<std::uint32_t>(nodes);
        nearest_        = take.template operator()<std::uint32_t>(nodes);
        removed_        = take.template operator()<std::uint32_t>(nodes);
        memo_           = take.template operator()<std::uint32_t>(nodes);
        groups_.storage = take.template operator()<std::uint32_t>(nodes);
        path_.storage   = take.template operator()<std::uint32_t>(nodes);
        stack_.storage  = take.template operator()<Path>(nodes);
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
        auto& stack = stack_;
        stack.clear();
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
        auto& path = path_;
        path.clear();
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

    template <class T>
    struct List {
        std::span<T> storage;
        std::size_t count = 0;

        auto begin() const { return storage.begin(); }

        auto end() const { return storage.begin() + count; }

        bool empty() const { return count == 0; }

        T& back() { return storage[count - 1]; }

        void clear() { count = 0; }

        void pop_back() { --count; }

        void push_back(T value) { storage[count++] = value; }
    };

    using Path = typename Index::LossPath;
    MetadataScratch::Lease storage_;
    const Index* index_;
    std::uint64_t epoch_, stamp_ = 0;
    std::span<std::uint8_t> advertised_;
    std::span<std::uint32_t> available_, nearest_, removed_, memo_;
    std::span<std::uint64_t> handle_stamp_, node_stamp_, memo_stamp_, score_stamp_;
    List<std::uint32_t> groups_, path_;
    List<Path> stack_;
    Work work_;
};

} // namespace ninfer::models::context
