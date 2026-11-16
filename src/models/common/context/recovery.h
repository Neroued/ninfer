#pragma once
#include "models/common/context/checkpoint_store.h"
#include "runtime/contract/recovery.h"
#include <algorithm>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace ninfer::models::context {
template <class Handle>
class RecoveryEvaluation;

template <class Handle = CheckpointHandle>
class RecoveryIndex : public MetadataCapacity {
public:
    static constexpr auto none = UINT32_MAX;

    explicit RecoveryIndex(MetadataBudget& budget, MetadataCapacity* auxiliary = nullptr)
        : metadata_(budget), scratch_(budget), auxiliary_(auxiliary) {}

    static std::size_t metadata_bytes(std::size_t capacity) {
        capacity += 255;
        return sizeof(RecoveryIndex) + capacity * sizeof(Slot) +
               (capacity + 1) * (sizeof(Node) + sizeof(std::uint32_t)) +
               evaluation_bytes(capacity, capacity + 1) +
               capacity * (sizeof(runtime::PrefixRelation) + sizeof(runtime::PrefixId) + 64);
    }

    std::size_t metadata_requirement(std::size_t capacity) const override {
        std::size_t overlap = 0;
        const auto growth   = [&](const auto& values, std::size_t n) {
            using T = typename std::decay_t<decltype(values)>::value_type;
            if (n <= values.capacity()) return std::size_t{0};
            overlap = std::max(overlap, values.capacity() * sizeof(T));
            return (n - values.capacity()) * sizeof(T);
        };
        const auto bytes = growth(slots_, capacity) + growth(nodes_, capacity + 1) +
                           growth(preorder_, capacity + 1) +
                           scratch_.growth_bytes(evaluation_bytes(capacity, capacity + 1));
        return bytes + overlap + (auxiliary_ ? auxiliary_->metadata_requirement(capacity) : 0);
    }

    void reserve_metadata(std::size_t capacity) override {
        if (auxiliary_) auxiliary_->reserve_metadata(capacity);
        if (capacity >= none) throw std::length_error("recovery index exceeds uint32");
        metadata_.reserve(slots_, capacity);
        metadata_.reserve(nodes_, capacity + 1);
        metadata_.reserve(preorder_, capacity + 1);
        scratch_.reserve(evaluation_bytes(capacity, capacity + 1));
    }

    void trim_metadata(std::size_t capacity) noexcept override {
        if (auxiliary_) auxiliary_->trim_metadata(capacity);
    }

    struct Work {
        std::uint64_t synchronizations = 0, relations = 0;
    };

    template <class View>
    void synchronize(const View& view) {
        if (revision_ == &view.revision() && observed_revision_ == view.revision()) return;
        if (revision_ && revision_ != &view.revision())
            throw std::logic_error("recovery index changed owner");
        revision_ = &view.revision();
        reserve_metadata(view.capacity());
        const auto& links = view.relations();
        if (links.size() != view.capacity()) throw std::logic_error("incomplete prefix relations");
        slots_.assign(view.capacity(), {});
        nodes_.assign(1, {});
        for (std::uint32_t i = 0; i < view.capacity(); ++i) {
            if (!view.valid(view.handle(i)) || links[i].representative != i) continue;
            slots_[i].node = nodes_.size();
            nodes_.push_back({.frontier = view.frontier(view.handle(i))});
        }
        for (std::uint32_t i = 0; i < view.capacity(); ++i) {
            const auto h = view.handle(i);
            if (!view.valid(h)) continue;
            const auto r = links[i].representative;
            if (r >= slots_.size() || slots_[r].node == none)
                throw std::logic_error("invalid equivalent prefix");
            const auto id    = slots_[r].node;
            auto& node       = nodes_[id];
            slots_[i]        = {h, id, node.first_alias};
            node.first_alias = i;
            if (r != i) continue;
            const auto parent = links[i].parent;
            node.parent = parent == none ? 0 : slots_.at(links.at(parent).representative).node;
            if (node.parent == none ||
                (node.parent && nodes_[node.parent].frontier >= node.frontier))
                throw std::logic_error("invalid strict prefix ancestor");
            node.next                       = nodes_[node.parent].first_child;
            nodes_[node.parent].first_child = id;
        }
        number_nodes();
        ++work_.synchronizations;
        work_.relations += view.capacity();
        observed_revision_ = view.revision();
        ++epoch_;
    }

    const Work& work() const noexcept { return work_; }

    RecoveryEvaluation<Handle> evaluate(std::span<const Handle> advertised) const {
        return RecoveryEvaluation<Handle>(*this, advertised);
    }
private:
    struct Node {
        std::uint32_t frontier = 0, parent = 0, first_child = none, next = none;
        std::uint32_t component = 0, begin = 0, end = 0, first_alias = none;
    };

    struct Slot {
        Handle handle{};
        std::uint32_t node = none, next = none;
    };

    struct LossPath {
        std::uint32_t node;
        std::uint64_t worst;
        bool leaf;
    };

    static std::size_t evaluation_bytes(std::size_t slots, std::size_t nodes) {
        // Four stamp arrays, five counters/lists, traversal path and loss stack. Each array
        // starts on an eight-byte boundary, including the byte-per-slot advertised flags.
        const auto aligned = [](std::size_t n) { return (n + 7) / 8 * 8; };
        return aligned(slots) + aligned(slots * sizeof(std::uint64_t)) +
               4 * aligned(nodes * sizeof(std::uint64_t)) +
               6 * aligned(nodes * sizeof(std::uint32_t)) + aligned(nodes * sizeof(LossPath));
    }

    void check() const {
        if (!revision_ || observed_revision_ != *revision_)
            throw std::logic_error("recovery index is stale");
    }

    std::uint32_t node(Handle h) const {
        return h.index < slots_.size() && slots_[h.index].handle == h ? slots_[h.index].node : none;
    }

    std::uint32_t require_node(Handle h) const {
        const auto id = node(h);
        if (id == none) throw std::logic_error("stale recovery handle");
        return id;
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
    MetadataCapacity* auxiliary_     = nullptr;
    const std::uint64_t* revision_   = nullptr;
    std::uint64_t observed_revision_ = 0, epoch_ = 0;
    std::vector<Node> nodes_;
    std::vector<Slot> slots_;
    std::vector<std::uint32_t> preorder_;
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
        values_         = take.template operator()<std::uint64_t>(nodes);
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

    template <class Visit>
    void visit_prefixes(Handle target, Visit visit) const {
        check();
        for (auto node = require_node(target); node; node = index_->nodes_[node].parent)
            for (auto slot = index_->nodes_[node].first_alias; slot != none;
                 slot      = index_->slots_[slot].next)
                if (!visit(index_->slots_[slot].handle)) return;
    }

    // Aggregate externally priced degradation using the same exact semantic forest.
    // Equivalent aliases contribute the maximum capability loss, never their count.
    template <class Value>
    std::uint64_t aggregate(std::span<const Handle> targets, Value value) {
        prepare({}, {});
        for (auto target : targets) {
            const auto loss = value(target);
            if (!loss) continue;
            const auto node = require_node(target);
            if (score_stamp_[node] != stamp_) {
                score_stamp_[node] = stamp_;
                values_[node]      = loss;
                groups_.push_back(node);
            } else
                values_[node] = std::max(values_[node], loss);
        }
        std::sort(groups_.begin(), groups_.end(), [&](auto a, auto b) {
            return index_->nodes_[a].begin < index_->nodes_[b].begin;
        });
        stack_.clear();
        std::uint64_t total = 0;
        const auto add      = [&](std::uint64_t value) {
            total = value > UINT64_MAX - total ? UINT64_MAX : total + value;
        };
        for (auto node : groups_) {
            while (!stack_.empty() &&
                   index_->nodes_[stack_.back().node].end <= index_->nodes_[node].begin) {
                if (stack_.back().leaf) add(stack_.back().worst);
                stack_.pop_back();
            }
            auto worst = values_[node];
            if (!stack_.empty()) {
                worst              = std::max(worst, stack_.back().worst);
                stack_.back().leaf = false;
            }
            stack_.push_back({node, worst, true});
        }
        for (const auto& path : stack_)
            if (path.leaf) add(path.worst);
        return total;
    }

    // Removing checkpoints on disjoint branches has additive recovery loss. This set tracks
    // ancestor overlap without comparing every trial with every previously removed point.
    class OverlapSet {
    public:
        explicit OverlapSet(const RecoveryEvaluation& evaluation) : evaluation_(&evaluation) {}

        [[nodiscard]] bool intersects(std::span<const Handle> handles) const {
            for (const auto handle : handles) {
                const auto region = region_of(handle);
                const auto it     = first_overlap(region.first);
                if (it != regions_.end() && it->first < region.second) return true;
            }
            return false;
        }

        void include(std::span<const Handle> handles) {
            for (const auto handle : handles) {
                auto region = region_of(handle);
                auto first = first_overlap(region.first), last = first;
                while (last != regions_.end() && last->first < region.second) {
                    region.first  = std::min(region.first, last->first);
                    region.second = std::max(region.second, last->second);
                    ++last;
                }
                regions_.insert(regions_.erase(first, last), region);
            }
        }

    private:
        std::pair<std::uint32_t, std::uint32_t> region_of(Handle handle) const {
            evaluation_->check();
            const auto& node = evaluation_->index_->nodes_[evaluation_->require_node(handle)];
            return {node.begin, node.end};
        }

        auto first_overlap(std::uint32_t begin) const {
            return std::lower_bound(
                regions_.begin(), regions_.end(), begin,
                [](const auto& region, auto value) { return region.second <= value; });
        }

        const RecoveryEvaluation* evaluation_;
        std::vector<std::pair<std::uint32_t, std::uint32_t>> regions_;
    };

    // A previously removed ancestor can share loss between effective branches. Include its
    // subtree even when redundant removed aliases obscure the effective forest's leaves.
    [[nodiscard]] OverlapSet dependencies(std::span<const Handle> removed,
                                          std::span<const Handle> additions) const {
        OverlapSet result(*this);
        result.include(additions);
        for (const auto added : additions)
            for (const auto h : removed)
                if (prefix(h, added)) result.include(std::span<const Handle>(&h, 1));
        return result;
    }

    [[nodiscard]] std::uint64_t loss(std::span<const Handle> removed,
                                     std::span<const Handle> unavailable = {}) {
        prepare(removed, unavailable);
        std::sort(groups_.begin(), groups_.end(), [&](auto a, auto b) {
            return index_->nodes_[a].begin < index_->nodes_[b].begin;
        });

        // Only effective degradation creates a loss branch. Removing a redundant alias must
        // not split an ancestor's loss while the same recovery capability survives.
        auto& stack = stack_;
        stack.clear();
        std::uint64_t total = 0;
        for (const auto node : groups_) {
            const auto& record = index_->nodes_[node];
            while (!stack.empty() && index_->nodes_[stack.back().node].end <= record.begin) {
                if (stack.back().leaf) total += stack.back().worst;
                stack.pop_back();
            }
            std::uint64_t worst = record.frontier - fallback(node);
            if (!worst) continue;
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
    std::span<std::uint64_t> handle_stamp_, node_stamp_, memo_stamp_, score_stamp_, values_;
    List<std::uint32_t> groups_, path_;
    List<Path> stack_;
    Work work_;
};

} // namespace ninfer::models::context
