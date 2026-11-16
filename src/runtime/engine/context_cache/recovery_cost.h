#pragma once

#include "runtime/contract/recovery.h"
#include "runtime/engine/context_cache/context_cost.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::runtime {

inline std::uint64_t add_recovery_cost(std::uint64_t a, std::uint64_t b) {
    return b > UINT64_MAX - a ? UINT64_MAX : a + b;
}

// UINT64_MAX denotes an unpriced/saturated ordering value, not an amount that can be
// subtracted to manufacture a zero loss or positive admission budget.
inline std::uint64_t recovery_cost_increase(std::uint64_t before, std::uint64_t after) {
    if (before == UINT64_MAX || after == UINT64_MAX) return UINT64_MAX;
    return after > before ? after - before : 0;
}

// Keep the finite sum and unpriced terms separate so component replacement never
// subtracts a saturated sentinel or loses a finite remainder.
struct RecoveryCostSum {
    unsigned __int128 finite = 0;
    std::size_t unpriced     = 0;

    void add(std::uint64_t value) {
        if (value == UINT64_MAX)
            ++unpriced;
        else
            finite += value;
    }

    void replace(std::uint64_t before, std::uint64_t after) {
        if (before == UINT64_MAX)
            --unpriced;
        else
            finite -= before;
        add(after);
    }

    std::uint64_t value() const {
        return unpriced || finite >= UINT64_MAX ? UINT64_MAX : static_cast<std::uint64_t>(finite);
    }
};

inline std::uint64_t recovery_restore_cost(const ContextMachineCostModel& costs,
                                           const RecoveryRestore& restore, bool replay = false) {
    const auto unpriced_peer = [&](const auto& work) {
        return !costs.prefill_measured && std::any_of(work.begin(), work.end(), [](const auto& r) {
            return r.peer.has_value();
        });
    };
    if (unpriced_peer(restore.transfers) || (replay && unpriced_peer(restore.replay_transfers)))
        return UINT64_MAX;
    if (!replay || restore.replay_transfers.empty())
        return price_context_transfer_requirements(costs, restore.transfers);
    auto work = restore.transfers;
    work.insert(work.end(), restore.replay_transfers.begin(), restore.replay_transfers.end());
    return price_context_transfer_requirements(costs, work);
}

// The static model is fixed for the policy lifetime. Decisions retain its compact prices,
// not repeated per-rank transfer inventories; immutable identity is shared by Native.
template <class Session>
struct RecoveryValue {
    typename Session::RecoveryAccount memory;
    std::shared_ptr<const RecoveryIdentity> identity;
    PlacementId placement;
    std::vector<RecoveryRestoreCost> restores;

    RecoveryValue(Session& session, const RecoveryDescription& native,
                  const ContextMachineCostModel& costs)
        : memory(session.recovery_budget()), identity(native.identity),
          placement(native.placement) {
        static_assert(sizeof(RecoveryValue) <= kRecoveryPricedDescriptorLimit);
        memory.acquire(sizeof(*this));
        memory.reserve(restores, native.restores.size());
        for (const auto& restore : native.restores)
            restores.push_back({restore.placement, recovery_restore_cost(costs, restore),
                                recovery_restore_cost(costs, restore, true)});
    }
};

template <class Session, class Handle>
struct PricedRecoveryPoint {
    Handle handle;
    std::shared_ptr<const RecoveryValue<Session>> description;
};

template <class Session, class Handle>
struct RecoveryLedger {
    using Point = PricedRecoveryPoint<Session, Handle>;
    typename Session::RecoveryAccount memory;
    std::vector<Point> basis;
    std::vector<Handle> advertised;

    struct Effect {
        Handle handle;
        bool retired = false;
    };

    std::vector<Effect> effects;
    std::uint64_t loss_after = 0, paid_actions = 0, revision = 0;
    bool bounded_actions = false;

    RecoveryLedger(Session& session, std::span<const Handle> visible,
                   const ContextMachineCostModel& costs)
        : memory(session.recovery_budget()) {
        const auto native = session.recovery_points();
        memory.acquire(native.capacity() * sizeof(RecoveryPoint<Handle>));
        memory.reserve(basis, native.size());
        for (const auto& point : native)
            basis.push_back({point.handle, std::make_shared<RecoveryValue<Session>>(
                                               session, *point.description, costs)});
        std::sort(basis.begin(), basis.end(),
                  [](const auto& a, const auto& b) { return a.handle.index < b.handle.index; });
        memory.reserve(advertised, visible.size());
        advertised.assign(visible.begin(), visible.end());
        memory.reserve(effects, basis.size());
        memory.release(native.capacity() * sizeof(RecoveryPoint<Handle>));
    }

    bool contains(Handle h) const {
        auto it =
            std::lower_bound(basis.begin(), basis.end(), h.index,
                             [](const auto& p, auto index) { return p.handle.index < index; });
        for (; it != basis.end() && it->handle.index == h.index; ++it)
            if (it->handle == h) return true;
        return false;
    }

    const Effect* effect(Handle h) const {
        auto it = std::lower_bound(
            effects.begin(), effects.end(), h.index,
            [](const auto& value, auto index) { return value.handle.index < index; });
        for (; it != effects.end() && it->handle.index == h.index; ++it)
            if (it->handle == h) return &*it;
        return nullptr;
    }

    bool affected(Handle h) const { return effect(h) != nullptr; }

    bool retired(Handle h) const {
        const auto* e = effect(h);
        return e && e->retired;
    }

    // A required recovery may revoke a snapshot created by pausing another request after
    // the decision began. Its owner explicitly adds that recovery obligation before mutation.
    // Optional admission never invokes this; new cache publications remain ineligible victims.
    void include_required(Session& session, std::span<const RecoveryPoint<Handle>> points,
                          std::span<const Handle> visible, const ContextMachineCostModel& costs) {
        memory.reserve(basis, basis.size() + points.size());
        memory.reserve(advertised, advertised.size() + points.size());
        bool changed = false;
        for (const auto& point : points) {
            if (contains(point.handle)) continue;
            basis.push_back({point.handle, std::make_shared<RecoveryValue<Session>>(
                                               session, *point.description, costs)});
            std::stable_sort(basis.begin(), basis.end(), [](const auto& a, const auto& b) {
                return a.handle.index < b.handle.index;
            });
            if (std::find(visible.begin(), visible.end(), point.handle) != visible.end())
                advertised.push_back(point.handle);
            changed = true;
        }
        memory.reserve(effects, basis.size());
        if (changed) ++revision;
    }

    void accept(std::span<const Handle> changed, std::span<const Handle> retired,
                std::uint64_t loss, std::uint64_t action, bool bounded = false) {
        for (auto h : changed)
            if (!contains(h))
                throw std::logic_error("reclaim action is outside its decision basis");
        for (auto h : changed) {
            const bool erased = std::find(retired.begin(), retired.end(), h) != retired.end();
            if (auto* prior = const_cast<Effect*>(effect(h))) {
                prior->retired |= erased;
            } else {
                const auto position = std::upper_bound(
                    effects.begin(), effects.end(), h.index,
                    [](auto index, const auto& value) { return index < value.handle.index; });
                effects.insert(position, {h, erased});
            }
        }
        loss_after   = loss;
        paid_actions = add_recovery_cost(paid_actions, action);
        bounded_actions |= bounded;
        ++revision;
    }
};

// A synchronous counterfactual evaluation. The reference view undoes only this decision's
// accepted effects; the actual view never advertises deleted records. New live fallbacks
// enter both views, while newly published targets cannot enlarge an existing budget.
template <class Session, class Handle>
class RecoveryCostEvaluation {
    using Point    = RecoveryPoint<Handle>;
    using Value    = RecoveryValue<Session>;
    using Ledger   = RecoveryLedger<Session, Handle>;
    using Topology = typename Session::RecoveryTopology;
    using Core = decltype(std::declval<Topology&>().evaluate(std::span<const RecoveryHandle>{}));

    struct Entry {
        Handle handle;
        std::shared_ptr<const Value> reference, current, projected;
        bool reference_visible = false, current_visible = false, target = false;
        bool restored_reference = false, externally_removed = false;
        std::uint32_t next_slot = UINT32_MAX;
        std::optional<std::uint64_t> reference_cost, current_cost;
    };

    using NativeSlot = std::uint32_t;

    struct Component {
        std::vector<RecoveryHandle> targets;
        std::uint64_t baseline = 0;
        bool changed           = false;
        std::size_t projected  = 0;
    };

    static_assert(sizeof(Entry) + sizeof(Component) + sizeof(NativeSlot) + 2 * sizeof(std::size_t) +
                      sizeof(RecoveryHandle) <=
                  kRecoveryEvaluationEntryLimit);

    struct View {
        const RecoveryCostEvaluation& owner;

        const std::uint64_t& revision() const { return owner.revision_; }

        std::size_t capacity() const { return owner.entries_.size(); }

        RecoveryHandle handle(std::uint32_t i) const { return {i}; }

        bool valid(RecoveryHandle h) const { return h.index < capacity(); }

        std::uint32_t frontier(RecoveryHandle h) const { return owner.identity(h).frontier; }

        typename Topology::Key key(RecoveryHandle h, std::uint32_t f) const {
            const auto& value = owner.identity(h);
            return {value.digest(f), value.domain};
        }

        bool prefix(RecoveryHandle a, RecoveryHandle b) const {
            return owner.identity(a).prefix_of(owner.identity(b));
        }
    };
public:
    RecoveryCostEvaluation(Session& session, std::span<const Handle> advertised, Ledger* ledger,
                           const ContextMachineCostModel& costs,
                           std::span<const Point> projected = {})
        : memory_(session.recovery_budget()), session_(session), ledger_(ledger), costs_(costs),
          view_{*this}, topology_(session.recovery_budget()) {
        auto current = session.recovery_points();
        memory_.acquire(current.capacity() * sizeof(Point));
        const auto by_slot = [](const auto& a, const auto& b) {
            return a.handle.index < b.handle.index;
        };
        std::sort(current.begin(), current.end(), by_slot);
        memory_.reserve(entries_, current.size() + (ledger ? ledger->basis.size() : 0));
        for (auto& point : current) {
            auto value = std::make_shared<Value>(session, *point.description, costs);
            Entry entry{point.handle, value, value, value, false, false, !ledger};
            if (ledger) {
                auto old =
                    std::lower_bound(ledger->basis.begin(), ledger->basis.end(), point, by_slot);
                while (old != ledger->basis.end() && old->handle.index == point.handle.index &&
                       old->handle != point.handle)
                    ++old;
                entry.target = old != ledger->basis.end() && old->handle == point.handle;
                if (entry.target && ledger->affected(point.handle)) {
                    entry.reference          = old->description;
                    entry.restored_reference = true;
                }
            }
            entries_.push_back(std::move(entry));
        }
        if (ledger)
            for (const auto& old : ledger->basis) {
                const auto live = std::lower_bound(current.begin(), current.end(), old, by_slot);
                if (live != current.end() && live->handle == old.handle) continue;
                // External deletion is absent from both source views. Its identity remains
                // a scoring target; only our own deletion restores a reference source.
                const bool ours = ledger->retired(old.handle);
                entries_.push_back(
                    {old.handle, old.description, {}, {}, false, false, true, ours, !ours});
            }
        for (const auto& entry : entries_)
            native_capacity_ = std::max<std::size_t>(native_capacity_, entry.handle.index + 1);
        memory_.reserve(native_slots_, native_capacity_);
        native_slots_.resize(native_capacity_, UINT32_MAX);
        for (std::uint32_t i = 0; i < entries_.size(); ++i) {
            auto& slot            = native_slots_[entries_[i].handle.index];
            entries_[i].next_slot = slot;
            slot                  = i;
        }
        for (auto h : advertised)
            if (const auto id = find(h)) {
                auto& entry           = entries_[id->index];
                entry.current_visible = true;
                if (!entry.restored_reference && !entry.externally_removed)
                    entry.reference_visible = true;
            }
        if (ledger)
            for (auto h : ledger->advertised)
                if (const auto id = find(h); id && entries_[id->index].restored_reference)
                    entries_[id->index].reference_visible = true;
        for (const auto& point : projected)
            if (const auto id = find(point.handle); id && entries_[id->index].current) {
                entries_[id->index].projected =
                    std::make_shared<Value>(session, *point.description, costs);
                has_projection_ = true;
            }
        topology_.synchronize(view_);
        core_.emplace(topology_.evaluate({}));
        memory_.reserve(components_, core_->component_capacity());
        components_.resize(core_->component_capacity());
        for (std::uint32_t i = 0; i < entries_.size(); ++i) {
            const auto& entry = entries_[i];
            auto& group       = components_[core_->component({i})];
            group.changed |= entry.current != entry.reference && !entry.externally_removed;
            group.projected += entry.current != entry.projected;
        }
        std::vector<std::size_t> target_counts(components_.size());
        memory_.acquire(target_counts.capacity() * sizeof(std::size_t));
        for (std::uint32_t i = 0; i < entries_.size(); ++i)
            if (entries_[i].target) ++target_counts[core_->component({i})];
        for (std::size_t c = 0; c < components_.size(); ++c)
            memory_.reserve(components_[c].targets, target_counts[c]);
        for (std::uint32_t i = 0; i < entries_.size(); ++i) {
            auto& entry = entries_[i];
            if (!entry.target) continue;
            auto& targets = components_[core_->component({i})].targets;
            targets.push_back({i});
            const auto placement = description({i}).placement;
            if (!contains(placements_, placement)) placements_.push_back(placement);
        }
        memory_.acquire(placements_.capacity() * sizeof(PlacementId));
        memory_.reserve(component_stamps_, components_.size());
        component_stamps_.resize(components_.size());
        memory_.reserve(dirty_components_, components_.size());
        memory_.reserve(projected_components_, components_.size());
        for (std::uint32_t c = 0; c < components_.size(); ++c) {
            auto& component = components_[c];
            if (component.projected) projected_components_.push_back(c);
            component.baseline = component_loss(component, {}, {}, Mode::Current);
            baseline_sum_.add(component.baseline);
        }
        baseline_ = baseline_sum_.value();
        if (ledger_) ledger_->loss_after = baseline_;
        memory_.release(current.capacity() * sizeof(Point) +
                        target_counts.capacity() * sizeof(std::size_t));
    }

    bool contains(Handle h) const { return find(h).has_value(); }

    std::uint32_t frontier(Handle h) const { return identity(require(h)).frontier; }

    std::uint32_t component(Handle h) const { return core_->component(require(h)); }

    std::size_t capacity() const { return native_capacity_; }

    std::size_t component_capacity() const { return components_.size(); }

    bool prefix(Handle a, Handle b) const { return core_->prefix(require(a), require(b)); }

    class Projection {
    public:
        Projection(RecoveryCostEvaluation& owner, std::span<const Point> points) : owner_(&owner) {
            if (owner.has_projection_) throw std::logic_error("nested recovery projection");
            owner.memory_.reserve(prior_, points.size());
            for (const auto& point : points) {
                const auto id = owner.find(point.handle);
                if (!id) continue;
                auto& entry = owner.entries_[id->index];
                prior_.emplace_back(id->index, entry.projected);
                entry.projected =
                    std::make_shared<Value>(owner.session_, *point.description, owner.costs_);
                const auto c = owner.core_->component(*id);
                if (!owner.components_[c].projected++) owner.projected_components_.push_back(c);
            }
            owner.has_projection_ = true;
        }

        Projection(const Projection&) = delete;

        ~Projection() {
            for (auto& [i, value] : prior_) {
                owner_->entries_[i].projected = std::move(value);
                --owner_->components_[owner_->core_->component({i})].projected;
            }
            owner_->has_projection_ = false;
            owner_->projected_components_.clear();
            owner_->memory_.release(prior_.capacity() *
                                    sizeof(typename decltype(prior_)::value_type));
        }
    private:
        RecoveryCostEvaluation* owner_;
        std::vector<std::pair<std::uint32_t, std::shared_ptr<const Value>>> prior_;
    };

    Projection project(std::span<const Point> points) { return Projection(*this, points); }

    class OverlapSet {
    public:
        OverlapSet(const RecoveryCostEvaluation& owner, typename Core::OverlapSet value)
            : owner_(&owner), value_(std::move(value)) {}

        bool intersects(std::span<const Handle> values) const {
            for (auto h : values) {
                const auto id = owner_->require(h);
                if (value_.intersects(std::span<const RecoveryHandle>(&id, 1))) return true;
            }
            return false;
        }
    private:
        const RecoveryCostEvaluation* owner_;
        typename Core::OverlapSet value_;
    };

    OverlapSet dependencies(std::span<const Handle> removed,
                            std::span<const Handle> additions) const {
        std::vector<RecoveryHandle> before, next;
        for (auto h : removed) before.push_back(require(h));
        for (auto h : additions) next.push_back(require(h));
        if (ledger_)
            for (const auto& effect : ledger_->effects)
                if (find(effect.handle)) before.push_back(require(effect.handle));
        return {*this, core_->dependencies(before, next)};
    }

    std::uint64_t total_loss(std::span<const Handle> removed     = {},
                             std::span<const Handle> unavailable = {}) {
        auto total = baseline_sum_;
        affected_components(removed, unavailable, true);
        for (auto c : dirty_components_)
            total.replace(components_[c].baseline,
                          component_loss(components_[c], removed, unavailable, Mode::Projected));
        return total.value();
    }

    std::uint64_t loss(std::span<const Handle> removed, std::span<const Handle> unavailable = {}) {
        auto before = baseline_sum_;
        if (!unavailable.empty()) {
            affected_components({}, unavailable, false);
            for (auto c : dirty_components_)
                before.replace(components_[c].baseline,
                               component_loss(components_[c], {}, unavailable, Mode::Current));
        }
        return recovery_cost_increase(before.value(), total_loss(removed, unavailable));
    }

    std::uint64_t gain(const RecoveryDescription& target, std::span<const Handle> removed = {},
                       bool reference = false) const {
        auto before = costs_.prefill_ns(target.identity->rebuild(0));
        for (const auto& entry : entries_) {
            const auto& source = reference ? entry.reference : entry.current;
            const bool visible = reference ? entry.reference_visible : entry.current_visible;
            if (!source || !visible || contains(removed, entry.handle) ||
                (reference && entry.externally_removed) ||
                !source->identity->prefix_of(*target.identity))
                continue;
            before = std::min(before, source_cost(*source, target));
        }
        const auto after = restore_cost(target, target.placement);
        if (before == UINT64_MAX || after == UINT64_MAX) return 0;
        return before > after ? before - after : 0;
    }


private:
    enum class Mode { Reference, Current, Projected };

    void affected_components(std::span<const Handle> removed, std::span<const Handle> unavailable,
                             bool projected) {
        dirty_components_.clear();
        if (!++component_epoch_) {
            std::fill(component_stamps_.begin(), component_stamps_.end(), 0);
            ++component_epoch_;
        }
        const auto include = [&](std::uint32_t c) {
            if (component_stamps_[c] == component_epoch_) return;
            component_stamps_[c] = component_epoch_;
            dirty_components_.push_back(c);
        };
        for (auto h : removed) include(component(h));
        for (auto h : unavailable) include(component(h));
        if (projected)
            for (auto c : projected_components_) include(c);
    }

    template <class T>
    static bool contains(std::span<const T> values, T value) {
        return std::find(values.begin(), values.end(), value) != values.end();
    }

    template <class T>
    static bool contains(const std::vector<T>& values, T value) {
        return contains(std::span<const T>(values), value);
    }

    std::optional<RecoveryHandle> find(Handle h) const {
        if (h.index >= native_slots_.size()) return {};
        for (auto i = native_slots_[h.index]; i != UINT32_MAX; i = entries_[i].next_slot)
            if (entries_[i].handle == h) return RecoveryHandle{i};
        return {};
    }

    RecoveryHandle require(Handle h) const {
        const auto value = find(h);
        if (!value) throw std::logic_error("recovery quote has a foreign checkpoint");
        return *value;
    }

    const Value& description(RecoveryHandle h) const {
        const auto& entry = entries_[h.index];
        return *(entry.reference ? entry.reference : entry.current);
    }

    const RecoveryIdentity& identity(RecoveryHandle h) const { return *description(h).identity; }

    std::uint64_t restore_cost(const Value& source, PlacementId target, bool replay = false) const {
        for (const auto& restore : source.restores)
            if (restore.placement == target)
                return replay ? restore.replay_nanoseconds : restore.nanoseconds;
        return UINT64_MAX;
    }

    std::uint64_t restore_cost(const RecoveryDescription& source, PlacementId target) const {
        for (const auto& restore : source.restores)
            if (restore.placement == target) return recovery_restore_cost(costs_, restore);
        return UINT64_MAX;
    }

    template <class Target>
    std::uint64_t source_cost(const Value& source, const Target& target) const {
        return add_recovery_cost(
            restore_cost(source, target.placement,
                         source.identity->frontier < target.identity->frontier),
            costs_.prefill_ns(target.identity->rebuild(source.identity->frontier)));
    }

    std::uint64_t cost_to(RecoveryHandle target, std::span<const Handle> removed,
                          std::span<const Handle> unavailable, Mode mode) const {
        const auto& goal     = description(target);
        const auto& own      = entries_[target.index];
        const bool reference = mode == Mode::Reference;
        const auto& local    = reference               ? own.reference
                               : mode == Mode::Current ? own.current
                                                       : own.projected;
        if (local && !contains(unavailable, own.handle) && !(reference && own.externally_removed) &&
            !(!reference && contains(removed, own.handle)) &&
            restore_cost(*local, goal.placement) == 0)
            return 0;
        auto result = costs_.prefill_ns(goal.identity->rebuild(0));
        if (local && !contains(unavailable, own.handle) && !(reference && own.externally_removed) &&
            !(!reference && contains(removed, own.handle)))
            result = std::min(result, restore_cost(*local, goal.placement));
        core_->visit_prefixes(target, [&](RecoveryHandle id) {
            const auto& entry = entries_[id.index];
            const auto begin  = identity(id).frontier;
            // Prefixes are visited from deepest to shallowest. Text work is a monotone
            // lower bound even when media windows alter the exact chunk segmentation.
            auto minimum = make_prefill_work(begin, goal.identity->frontier - begin, 0, 0, 0);
            if (costs_.prefill_ns(minimum) >= result) return false;
            const auto& source = reference               ? entry.reference
                                 : mode == Mode::Current ? entry.current
                                                         : entry.projected;
            if (!source || contains(unavailable, entry.handle) ||
                (reference && entry.externally_removed) ||
                (!reference && contains(removed, entry.handle)))
                return true;
            const bool visible = reference ? entry.reference_visible : entry.current_visible;
            if (!visible && entry.handle != own.handle) return true;
            result = std::min(result, source_cost(*source, goal));
            return true;
        });
        return result;
    }

    std::uint64_t component_loss(const Component& group, std::span<const Handle> removed,
                                 std::span<const Handle> unavailable, Mode mode) {
        const auto id = static_cast<std::uint32_t>(&group - components_.data());
        // Identical source sets prove zero degradation even if their absolute price is
        // unavailable. This proof is different from subtracting two saturated estimates.
        if (!group.changed && (mode != Mode::Projected || !group.projected) &&
            std::none_of(removed.begin(), removed.end(),
                         [&](auto h) { return component(h) == id; }))
            return 0;
        std::uint64_t total = 0;
        for (auto placement : placements_)
            total = add_recovery_cost(total, core_->aggregate(group.targets, [&](RecoveryHandle h) {
                if (description(h).placement != placement) return std::uint64_t{0};
                auto& entry       = entries_[h.index];
                const auto cached = [&](std::optional<std::uint64_t>& value, Mode requested) {
                    if (!value) value = cost_to(h, {}, {}, requested);
                    return *value;
                };
                const auto before = unavailable.empty()
                                        ? cached(entry.reference_cost, Mode::Reference)
                                        : cost_to(h, {}, unavailable, Mode::Reference);
                const auto after  = mode == Mode::Current && removed.empty() && unavailable.empty()
                                        ? cached(entry.current_cost, Mode::Current)
                                        : cost_to(h, removed, unavailable, mode);
                return recovery_cost_increase(before, after);
            }));
        return total;
    }

    typename Session::RecoveryAccount memory_;
    Session& session_;
    Ledger* ledger_;
    const ContextMachineCostModel& costs_;
    std::vector<Entry> entries_;
    std::vector<NativeSlot> native_slots_;
    std::vector<Component> components_;
    std::vector<PlacementId> placements_;
    std::vector<std::uint64_t> component_stamps_;
    std::vector<std::uint32_t> dirty_components_, projected_components_;
    std::uint64_t component_epoch_ = 0;
    RecoveryCostSum baseline_sum_;
    std::uint64_t revision_ = 1, baseline_ = 0;
    std::size_t native_capacity_ = 0;
    bool has_projection_         = false;
    View view_;
    Topology topology_;
    std::optional<Core> core_;
};

} // namespace ninfer::runtime
