#pragma once

#include "runtime/contract/resources.h"
#include "runtime/engine/context_cache/types.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace ninfer::runtime {

// One synchronous selection over immutable physical and recovery inventories. Trials update
// additive measures; only chosen actions materialize the release set used for placement.
template <class Policy>
class ReclaimSelection {
    using Session    = typename Policy::Session;
    using Handle     = typename Policy::Handle;
    using Assessment = typename Policy::ContextAssessment;
    using Cursor     = typename Policy::ReclaimCursor;
    using Evaluation = typename Policy::Evaluation;
    using Recovery   = typename Policy::Recovery;
    using Admission  = typename Policy::Admission;
    using Facts      = typename Policy::ActionFacts;
    using Rank       = typename Policy::ActionRank;
    using Plan       = typename Policy::HostReleasePlan;
    using Release    = typename decltype(std::declval<Plan>().releases)::value_type;
    using Inventory =
        std::remove_const_t<typename decltype(std::declval<Plan>().inventory)::element_type>;

    struct Group {
        std::vector<Handle> removed;
        std::uint64_t loss = 0, version = 0;
        bool private_proof = false;
    };

    struct Part {
        std::uint32_t group;
        std::vector<Handle> additions;
        std::uint64_t version = UINT64_MAX;
        Facts facts;
        std::uint64_t prior_loss     = 0;
        std::size_t observed_removed = 0;
        std::optional<typename Recovery::OverlapSet> dependencies;
    };

    struct Candidate {
        std::vector<Part> parts;
        std::uint64_t waiting      = 0;
        std::size_t snapshot_order = 0;
        bool allowed = true, chosen = false;
    };

    struct Gain {
        const Admission* admission;
        std::vector<Handle> matching, sources;
        std::vector<std::uint32_t> groups;
        std::size_t first = 0;
    };

public:
    struct Choice {
        std::size_t index;
        Rank rank;
        std::size_t snapshot_order;
    };

    ReclaimSelection(const Policy& policy, Session& session, const Assessment& assessment,
                     const Cursor& cursor, Evaluation& evaluation, const Inventory& inventory,
                     std::span<const Release> releases,
                     std::span<const Admission* const> admissions,
                     std::span<const Handle> excluded        = {},
                     std::span<const Handle> later_snapshots = {})
        : policy_(policy), session_(session), cursor_(cursor), evaluation_(evaluation),
          index_(policy.recovery(session, evaluation)), inventory_(inventory), releases_(releases),
          limits_(session.release_limits(assessment)),
          measures_(measure(session, assessment, inventory)),
          measured_(inventory.select(std::span<const ContextReleaseMeasure>(measures_))),
          committed_(index_.capacity()) {
        std::vector<std::uint32_t> parents(index_.component_capacity());
        std::iota(parents.begin(), parents.end(), 0);
        const auto root = [&](std::uint32_t node) {
            while (parents[node] != node) {
                parents[node] = parents[parents[node]];
                node          = parents[node];
            }
            return node;
        };
        for (const auto& proof : cursor.owners) {
            std::optional<std::uint32_t> first;
            const auto join = [&](Handle h) {
                if (!index_.contains(h)) return;
                const auto node = root(index_.component(h));
                if (first)
                    parents[node] = root(*first);
                else
                    first = node;
            };
            for (const auto h : proof.points) join(h);
            if (const auto* owner = policy.find_owner(proof.owner))
                for (const auto& point : owner->points) join(point.handle);
        }
        const auto component = [&](Handle h) { return root(index_.component(h)); };
        groups_.resize(parents.size());
        for (const auto& proof : cursor.owners) {
            if (!proof.priority.reused()) continue;
            for (const auto h : proof.points)
                if (index_.contains(h)) groups_[component(h)].private_proof = true;
        }
        candidates_.reserve(releases.size());
        for (const auto& release : releases) {
            Candidate candidate;
            candidate.allowed = policy.may_revoke(release.sources, cursor.rights) &&
                                std::none_of(release.sources.begin(), release.sources.end(),
                                             [&](auto h) { return Policy::contains(excluded, h); });
            if (candidate.allowed) {
                candidate.waiting = policy.waiting_ticket(release.sources);
                auto ordered      = release.sources;
                std::sort(ordered.begin(), ordered.end(), [&](auto a, auto b) {
                    return std::pair{component(a), a.index} < std::pair{component(b), b.index};
                });
                ordered.erase(std::unique(ordered.begin(), ordered.end()), ordered.end());
                for (const auto h : ordered) {
                    const auto group = component(h);
                    if (candidate.parts.empty() || candidate.parts.back().group != group)
                        candidate.parts.push_back({group, {}});
                    candidate.parts.back().additions.push_back(h);
                }
                for (std::size_t i = 0; i < later_snapshots.size(); ++i)
                    if (Policy::contains(release.sources, later_snapshots[i]))
                        candidate.snapshot_order = i + 1;
            }
            candidates_.push_back(std::move(candidate));
        }
        for (const auto* admission : admissions) {
            if (!admission || std::any_of(gains_.begin(), gains_.end(), [&](const auto& gain) {
                    return gain.admission == admission;
                }))
                continue;
            Gain gain{admission};
            if (admission->base) {
                gain.matching = policy.incoming_candidates(session, *admission, evaluation);
            } else if (admission->checkpoint && index_.contains(*admission->checkpoint)) {
                gain.sources = admission->sources;
                if (gain.sources.empty()) gain.sources.push_back(*admission->checkpoint);
                for (const auto h : gain.sources)
                    if (index_.contains(h)) gain.groups.push_back(component(h));
                std::sort(gain.groups.begin(), gain.groups.end());
                gain.groups.erase(std::unique(gain.groups.begin(), gain.groups.end()),
                                  gain.groups.end());
            }
            gains_.push_back(std::move(gain));
        }
    }

    [[nodiscard]] std::optional<Choice> best() {
        std::optional<Choice> best;
        const auto before_bytes    = measured_.released().total_bytes();
        const auto before_metadata = measured_.released().metadata_objects;
        for (std::size_t i = 0; i < candidates_.size(); ++i) {
            auto& candidate = candidates_[i];
            if (candidate.chosen || !candidate.allowed) continue;
            auto probe        = measured_.probe(releases_[i].sources);
            const auto& value = measured_.released();
            if (value.total_bytes() <= before_bytes && value.metadata_objects <= before_metadata)
                continue;
            // Removing more checkpoints cannot lower recovery loss. Cached component losses
            // therefore bound this trial even after their component has changed.
            auto bound = selected_;
            bool known = true;
            for (const auto& part : candidate.parts) {
                if (!part.facts.loss) {
                    known = false;
                    break;
                }
                const auto prior = groups_[part.group].loss;
                bound.loss += std::max(prior, *part.facts.loss) - prior;
                include_priority(bound.priority, part.facts.priority);
                bound.order         = std::max(bound.order, part.facts.order);
                bound.explicit_hint = bound.explicit_hint || part.facts.explicit_hint;
            }
            bound.units         = std::max<std::uint64_t>(1, value.useful_bytes(limits_));
            bound.waiting       = include_waiting(selected_.waiting, candidate.waiting);
            const auto snapshot = std::max(selected_snapshot_, candidate.snapshot_order);
            if (known && best &&
                (snapshot > best->snapshot_order ||
                 (snapshot == best->snapshot_order && !Policy::less_rank(bound, best->rank))))
                continue;
            auto rank = selected_;
            for (auto& part : candidate.parts) {
                const auto& group = groups_[part.group];
                if (part.version != group.version) {
                    const auto changes =
                        std::span<const Handle>(group.removed).subspan(part.observed_removed);
                    if (!group.private_proof && part.dependencies &&
                        !part.dependencies->intersects(changes)) {
                        part.facts.loss = group.loss + (*part.facts.loss - part.prior_loss);
                    } else {
                        auto removed = group.removed;
                        for (const auto h : part.additions)
                            if (!committed_[h.index]) removed.push_back(h);
                        part.facts = policy_.action_facts(session_, removed, cursor_, evaluation_);
                        part.facts.loss = index_.loss(removed);
                        if (!group.private_proof)
                            part.dependencies = index_.dependencies(group.removed, part.additions);
                    }
                    part.prior_loss       = group.loss;
                    part.observed_removed = group.removed.size();
                    part.version          = group.version;
                }
                rank.loss -= group.loss;
                rank.loss += *part.facts.loss;
                include_priority(rank.priority, part.facts.priority);
                rank.order         = std::max(rank.order, part.facts.order);
                rank.explicit_hint = rank.explicit_hint || part.facts.explicit_hint;
            }
            bool accepted = true;
            for (auto& gain : gains_) {
                if (!Policy::admits_limits(rank.priority, rank.loss, *gain.admission, cursor_) ||
                    (!gain.admission->demand.reused() &&
                     gain_after(gain, candidate) <= rank.loss)) {
                    accepted = false;
                    break;
                }
            }
            if (!accepted) continue;
            rank.units   = std::max<std::uint64_t>(1, value.useful_bytes(limits_));
            rank.waiting = include_waiting(selected_.waiting, candidate.waiting);
            if (!best || snapshot < best->snapshot_order ||
                (snapshot == best->snapshot_order && Policy::less_rank(rank, best->rank)))
                best = Choice{i, rank, snapshot};
        }
        return best;
    }

    void commit(const Choice& choice) {
        auto& candidate = candidates_[choice.index];
        auto chosen     = measured_.probe(releases_[choice.index].sources);
        chosen.commit();
        for (const auto object : measured_.freed().subspan(materialized_objects_))
            released_.merge(inventory_.objects()[object].released);
        materialized_objects_ = measured_.freed().size();
        for (const auto& part : candidate.parts) {
            auto& group = groups_[part.group];
            for (const auto h : part.additions)
                if (!committed_[h.index]) {
                    committed_[h.index] = true;
                    group.removed.push_back(h);
                }
            group.loss = *part.facts.loss;
            ++group.version;
        }
        candidate.chosen   = true;
        selected_          = choice.rank;
        selected_snapshot_ = choice.snapshot_order;
    }

    [[nodiscard]] std::span<const Handle> handles() const { return measured_.handles(); }

    [[nodiscard]] const auto& released() const { return released_; }

    [[nodiscard]] std::uint64_t loss() const { return selected_.loss; }

private:
    static std::vector<ContextReleaseMeasure>
    measure(Session& session, const Assessment& assessment, const Inventory& inventory) {
        std::vector<ContextReleaseMeasure> values;
        values.reserve(inventory.objects().size());
        for (const auto& object : inventory.objects())
            values.push_back(session.release_measure(object.released, assessment));
        return values;
    }

    std::uint64_t gain_after(Gain& gain, const Candidate& candidate) {
        if (gain.admission->base) {
            while (gain.first < gain.matching.size() && committed_[gain.matching[gain.first].index])
                ++gain.first;
            auto i = gain.first;
            while (i < gain.matching.size() && measured_.contains(gain.matching[i])) ++i;
            const auto fallback = i == gain.matching.size() ? 0 : index_.frontier(gain.matching[i]);
            return gain.admission->frontier - fallback;
        }
        std::vector<Handle> unavailable;
        for (const auto group : gain.groups) {
            const auto& removed = groups_[group].removed;
            unavailable.insert(unavailable.end(), removed.begin(), removed.end());
            for (const auto& part : candidate.parts)
                if (part.group == group)
                    for (const auto h : part.additions)
                        if (!committed_[h.index]) unavailable.push_back(h);
        }
        return index_.loss(gain.sources, unavailable);
    }

    static void include_priority(CacheRetentionPriority& current, CacheRetentionPriority next) {
        if (next.reused() && (!current.reused() || next.last_demand > current.last_demand))
            current = next;
    }

    static std::uint64_t include_waiting(std::uint64_t current, std::uint64_t next) {
        return !current ? next : !next ? current : std::min(current, next);
    }

    const Policy& policy_;
    Session& session_;
    const Cursor& cursor_;
    Evaluation& evaluation_;
    Recovery& index_;
    const Inventory& inventory_;
    std::span<const Release> releases_;
    std::vector<std::uint64_t> limits_;
    std::vector<ContextReleaseMeasure> measures_;
    typename Inventory::template Selection<ContextReleaseMeasure> measured_;
    typename Policy::ContextReleaseSet released_;
    std::size_t materialized_objects_ = 0;
    std::vector<bool> committed_;
    std::vector<Group> groups_;
    std::vector<Candidate> candidates_;
    std::vector<Gain> gains_;
    Rank selected_;
    std::size_t selected_snapshot_ = 0;
};

} // namespace ninfer::runtime
