#pragma once

#include "runtime/contract/resources.h"
#include "runtime/contract/placement.h"
#include "runtime/engine/context_cache/context_cost.h"
#include "runtime/engine/context_cache/prefix_index.h"
#include "runtime/engine/context_cache/types.h"
#include "runtime/engine/context_cache/reclaim_selection.h"
#include "runtime/engine/context_cache/recovery_cost.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <list>
#include <map>
#include <optional>
#include <span>
#include <stdexcept>
#include <unordered_set>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ninfer::runtime {

// Policy owns bounded logical continuations. Native owns checkpoint contents and every byte.
template <class Model>
class ResourceManager {
    friend class ReclaimSelection<ResourceManager>;
public:
    using Session           = typename Model::Session;
    using ResumeState       = typename Model::ResumeState;
    using ContextDemand     = typename Session::ContextDemand;
    using ContextAssessment = typename Session::ContextAssessment;
    using ContextReleaseSet = typename Session::ContextReleaseSet;
    using Base              = typename Model::RequestBasePlan;
    using Handle            = typename Model::CheckpointHandle;
    using HostRequirement   = typename Session::HostRequirement;
    using Source            = typename Model::SourceCandidate;
    using SessionKey        = typename Model::CacheSessionKey;
    using OwnerToken        = ContinuationOwnerToken;
    using Ledger            = RecoveryLedger<Session, Handle>;

    using DemandKey =
        typename decltype(std::declval<const Base&>().prefix_shortlist_key(0))::value_type;

    // Optional writes carry only their own demand evidence. Necessary execution has no
    // admission restriction. The request owns base for the lifetime of the reclaim decision.
    struct Admission {
        CacheRetentionPriority priority;
        CacheRetentionPriority demand;
        CheckpointRole purpose = CheckpointRole::SharedPrefix;
        OwnerToken owner       = 0;
        const Base* base       = nullptr;
        std::uint32_t frontier = 0;
        std::optional<Handle> checkpoint;
        std::optional<Handle> replacement;
        std::vector<Handle> sources;
        PlacementId placement;
        std::shared_ptr<const RecoveryDescription> target;
        std::uint64_t pending_action = 0;
    };

    struct ReclaimCursor {
        struct Victim {
            Handle handle;
            CacheRetentionPriority priority;
            CacheRetentionPriority demand;
            SharedCandidateEvidence evidence = SharedCandidateEvidence::None;
        };

        struct OwnerProtection {
            OwnerToken owner;
            CacheRetentionPriority priority;
            std::uint32_t proven_frontier;
            std::vector<Handle> points;
        };

        // Fixed membership, order and writeback rights for one decision. Native residency
        // supplies progress; a history may still own records needed by a later demand.
        std::vector<Victim> victims;
        std::vector<OwnerProtection> owners;
        std::optional<Admission> admission;
        std::optional<std::uint64_t> gain_limit;
        std::uint64_t id = 0;
        mutable std::unique_ptr<Ledger> ledger;
        ReclaimRights rights;
        std::vector<Handle> protected_points;
    };

    struct ReclaimActionQuote {
        std::uint64_t decision = 0, revision = 0;
        std::vector<Handle> retired;
        std::uint64_t loss_after = 0, action_cost = 0;
        bool budgeted = false;
        // Reserved in admission/ranking, paid only if that later maintenance starts.
        std::uint64_t followup_cost = 0;
        std::vector<Handle> changed;
    };

    struct SourceChoice {
        Source source;
        std::optional<CheckpointRole> source_role;
        OwnerToken owner           = 0;
        std::uint64_t shared_entry = 0;
        bool take_over             = false;
        bool private_snapshot      = false;
        std::optional<OwnerToken> resume_owner;
        bool session_hint               = false;
        std::uint64_t publication_order = 0;
        std::uint64_t ordinal           = 0;
    };

    explicit ResourceManager(bool enabled, ContextMachineCostModel costs)
        : enabled_(enabled), costs_(std::move(costs)) {}

    [[nodiscard]] const PrefixCacheStatistics& statistics() const noexcept {
        return index_.statistics();
    }

    [[nodiscard]] std::vector<SourceChoice>
    candidates(Session& session, const Base& base, std::uint32_t maximum_frontier = UINT32_MAX,
               std::optional<OwnerToken> own_resume = std::nullopt, std::uint64_t request = 0,
               std::uint64_t publication_order = UINT64_MAX, const ResumeState* resume = nullptr) {
        prune(session);
        std::vector<SourceChoice> result;
        if (enabled_) {
            for (const auto handle : index_.matches(session, base)) {
                if (session.checkpoint_metadata(handle).frontier > maximum_frontier) { continue; }
                for (const auto& owner : owners_) {
                    const auto point =
                        std::find_if(owner.points.begin(), owner.points.end(),
                                     [&](const auto& p) { return p.handle == handle; });
                    if (point == owner.points.end()) { continue; }
                    const auto& advice = base.context_cache();
                    const bool take =
                        own_resume
                            ? *own_resume == owner.token
                            : !owner.active && owner.publication_order <= publication_order &&
                                  (!owner.session || (advice.update_session_index &&
                                                      advice.session_key == owner.session));
                    result.push_back(
                        {.source            = Source{.checkpoint = handle},
                         .owner             = owner.token,
                         .take_over         = take,
                         .resume_owner      = own_resume,
                         .session_hint      = owner.session && owner.session == advice.session_key,
                         .publication_order = owner.publication_order,
                         .ordinal           = owner.token});
                }
                for (const auto& shared : shared_) {
                    if (shared.handle != handle) { continue; }
                    result.push_back({.source       = Source{.checkpoint = handle},
                                      .shared_entry = shared.ordinal,
                                      .resume_owner = own_resume,
                                      .ordinal      = shared.ordinal});
                }
            }
        }
        if (own_resume) {
            for (const auto handle : points(*own_resume)) {
                if (!session.valid_checkpoint(handle) ||
                    session.checkpoint_metadata(handle).frontier > maximum_frontier) {
                    continue;
                }
                result.push_back({.source            = Source{.checkpoint = handle},
                                  .owner             = *own_resume,
                                  .take_over         = true,
                                  .resume_owner      = own_resume,
                                  .publication_order = find_owner(*own_resume)->publication_order,
                                  .ordinal           = *own_resume});
            }
        }
        if (resume && resume->snapshot_handle() &&
            session.valid_checkpoint(*resume->snapshot_handle()) &&
            session.checkpoint_metadata(*resume->snapshot_handle()).frontier <= maximum_frontier) {
            result.push_back({.source       = Source{.checkpoint = resume->snapshot_handle()},
                              .resume_owner = own_resume});
        }
        result.push_back({.source = Source{}, .resume_owner = own_resume});
        if (const auto* waiting = find_waiting(request);
            waiting && waiting->choice.source.checkpoint &&
            contains(waiting->held, *waiting->choice.source.checkpoint)) {
            result.push_back(waiting->choice);
        }
        std::vector<SourceChoice> expanded;
        const auto placements = session.placements(base);
        for (const auto& choice : result) {
            for (const auto placement : placements) {
                auto candidate             = choice;
                candidate.source.placement = placement;
                if (prepare_source(session, base, candidate, request, std::nullopt, resume)) {
                    if (candidate.source.checkpoint)
                        candidate.source_role =
                            session.checkpoint_metadata(*candidate.source.checkpoint).role;
                    expanded.push_back(std::move(candidate));
                }
            }
        }
        result              = std::move(expanded);
        const auto raw_cost = [&](const Source& source) {
            const auto transfer = try_price_context_transfer_requirements(costs_, source.transfers);
            const auto prefill  = costs_.prefill_ns(source.remaining_work);
            return !transfer || *transfer > UINT64_MAX - prefill ? UINT64_MAX : *transfer + prefill;
        };
        const auto remote = [](const Source& source) {
            return std::any_of(source.transfers.begin(), source.transfers.end(),
                               [](const auto& transfer) { return transfer.peer.has_value(); });
        };
        std::vector<std::pair<PlacementId, std::uint64_t>> local_costs;
        for (const auto placement : placements) {
            std::uint64_t best = UINT64_MAX;
            for (const auto& choice : result) {
                if (choice.source.placement == placement && !remote(choice.source)) {
                    best = std::min(best, raw_cost(choice.source));
                }
            }
            local_costs.emplace_back(placement, best);
        }
        const auto cost = [&](const Source& source) {
            const auto price = raw_cost(source);
            if (!remote(source)) { return price; }
            if (!costs_.prefill_measured || price == UINT64_MAX) { return UINT64_MAX; }
            const auto local =
                std::find_if(local_costs.begin(), local_costs.end(),
                             [&](const auto& entry) { return entry.first == source.placement; });
            if (local == local_costs.end() || local->second == UINT64_MAX) { return price; }
            // Only measured immediate savings justify an optional peer copy. A remote source
            // remains available when local plans cannot reserve the required context.
            return price < local->second ? price : UINT64_MAX;
        };
        const auto bytes = [](const Source& source) {
            std::uint64_t total = 0;
            for (const auto& transfer : source.transfers) {
                total = transfer.work.payload_bytes > UINT64_MAX - total
                            ? UINT64_MAX
                            : total + transfer.work.payload_bytes;
            }
            return total;
        };
        std::stable_sort(result.begin(), result.end(), [&](const auto& a, const auto& b) {
            const auto left = cost(a.source), right = cost(b.source);
            if (left != right) { return left < right; }
            if (bytes(a.source) != bytes(b.source)) { return bytes(a.source) < bytes(b.source); }
            if (a.source.reused_tokens != b.source.reused_tokens) {
                return a.source.reused_tokens > b.source.reused_tokens;
            }
            if (a.session_hint != b.session_hint) { return a.session_hint; }
            if (bool(a.owner) != bool(b.owner)) { return bool(a.owner); }
            if (a.publication_order != b.publication_order) {
                return a.publication_order > b.publication_order;
            }
            return a.ordinal < b.ordinal;
        });
        // Equal handles are the same Native contents. Independently computed states with equal
        // semantic keys remain separate candidates; Native must not splice their KV and state.
        std::vector<SourceChoice> unique;
        for (auto& choice : result) {
            const auto duplicate =
                std::any_of(unique.begin(), unique.end(), [&](const auto& prior) {
                    return prior.source.checkpoint == choice.source.checkpoint &&
                           prior.source.placement == choice.source.placement &&
                           prior.source.consume_source == choice.source.consume_source &&
                           prior.source.take_private == choice.source.take_private &&
                           prior.source.private_points == choice.source.private_points &&
                           prior.source.retired_points == choice.source.retired_points;
                });
            if (!duplicate) { unique.push_back(std::move(choice)); }
        }
        return unique;
    }

    // Evaluation grants consumption rights but does not retire anything. Native checks the
    // complete binding before applying these grants in its ownership handoff.
    [[nodiscard]] bool prepare_source(Session& session, const Base& base, SourceChoice& choice,
                                      std::uint64_t request                   = 0,
                                      std::optional<ReclaimRights> revocation = std::nullopt,
                                      const ResumeState* resume               = nullptr) {
        const auto selected = choice.source.checkpoint;
        if (selected && !session.valid_checkpoint(*selected)) { return false; }
        auto* owner = find_owner(choice.owner);
        if (choice.take_over && (!owner || owner->publication_order != choice.publication_order ||
                                 (owner->active && choice.resume_owner != choice.owner))) {
            choice.take_over = false;
        }
        std::vector<Handle> retired;
        if (choice.take_over && selected && owner) {
            const auto frontier = session.checkpoint_metadata(*selected).frontier;
            for (const auto& point : owner->points) {
                if (point.handle == *selected || !session.valid_checkpoint(point.handle)) {
                    continue;
                }
                const auto summary = session.checkpoint_metadata(point.handle);
                if (summary.frontier > frontier && !summary.leased &&
                    consumable(point.handle, owner->token, request, revocation)) {
                    retired.push_back(point.handle);
                }
            }
        }
        auto carried = points(choice.resume_owner.value_or(choice.owner));
        if (const auto* waiting = find_waiting(request);
            waiting && waiting->choice.source.checkpoint == selected &&
            waiting->choice.source.placement == choice.source.placement) {
            carried = waiting->choice.source.private_points;
            std::erase_if(carried, [&](auto point) { return !contains(waiting->held, point); });
        }
        const auto private_owner =
            choice.resume_owner.value_or(choice.take_over ? choice.owner : 0);
        const bool private_snapshot = resume && selected && resume->snapshot_handle() == selected;
        const bool consume = selected && (private_snapshot || consumable(*selected, private_owner,
                                                                         request, revocation));
        auto inspected     = session.inspect_source(base, selected, consume, carried, retired,
                                                    choice.source.placement, resume);
        if (!inspected) { return false; }
        inspected->take_private =
            std::all_of(inspected->private_points.begin(), inspected->private_points.end(),
                        [&](auto point) { return consumable(point, private_owner, request); });
        choice.source           = std::move(*inspected);
        choice.private_snapshot = private_snapshot;
        return true;
    }

    void retain_source(Session& session, std::uint64_t request, const SourceChoice& choice) {
        std::vector<Handle> held;
        if (choice.source.checkpoint && !choice.private_snapshot) {
            held.push_back(*choice.source.checkpoint);
        }
        for (const auto point : choice.source.private_points) {
            if (!contains(held, point)) { held.push_back(point); }
        }
        auto* waiting = find_waiting(request);
        if (!waiting) {
            if (held.empty()) { return; }
            waiting_.push_back({.request = request, .choice = choice, .held = std::move(held)});
            return;
        }
        auto old        = std::move(waiting->held);
        waiting->choice = choice;
        waiting->held   = std::move(held);
        for (const auto point : old) { release_unreferenced(session, point); }
    }

    [[nodiscard]] std::optional<SourceChoice> retained_source(std::uint64_t request) const {
        const auto* waiting = find_waiting(request);
        if (!waiting || !waiting->choice.source.checkpoint ||
            !contains(waiting->held, *waiting->choice.source.checkpoint)) {
            return std::nullopt;
        }
        return waiting->choice;
    }

    [[nodiscard]] bool has_source_record(std::uint64_t request) const {
        return find_waiting(request) != nullptr;
    }

    [[nodiscard]] bool can_transfer_private(OwnerToken owner, std::uint64_t request) const {
        const auto carried = points(owner);
        return std::all_of(carried.begin(), carried.end(),
                           [&](auto point) { return consumable(point, owner, request); });
    }

    [[nodiscard]] std::uint32_t source_revocations(std::uint64_t request) const {
        const auto* waiting = find_waiting(request);
        return waiting ? waiting->revocations : 0;
    }

    void binding_started(std::uint64_t request, std::span<const Handle> retired,
                         std::optional<Handle> consumed) {
        if (auto* waiting = find_waiting(request)) { waiting->binding = true; }
        for (const auto handle : retired) { forget(handle); }
        if (consumed) {
            for (auto& waiting : waiting_) {
                if (waiting.request != request && std::erase(waiting.held, *consumed)) {
                    ++waiting.revocations;
                }
            }
        }
    }

    void release_source(Session& session, std::uint64_t request) {
        const auto found = std::find_if(waiting_.begin(), waiting_.end(), [=](const auto& entry) {
            return entry.request == request;
        });
        if (found == waiting_.end()) { return; }
        auto held = std::move(found->held);
        waiting_.erase(found);
        for (const auto handle : held) { release_unreferenced(session, handle); }
    }

    // Called exactly once after Native's irreversible bind commit. The choice preserves logical
    // ownership even when Native consumed its original checkpoint handle during that commit.
    [[nodiscard]] OwnerToken adopt(Session& session, const SourceChoice& choice, const Base& base,
                                   std::uint64_t publication_order, bool resuming,
                                   std::span<const Handle> carried = {},
                                   std::span<const Handle> retired = {}) {
        if (!resuming) {
            ++request_epoch_;
            if (choice.source_role && choice.source.reused_tokens)
                index_.record_adoption(*choice.source_role, choice.source.reused_tokens);
        }
        for (const auto handle : retired) { forget(handle); }
        if (!enabled_ || !base.summary().publish_continuation) {
            for (const auto handle : carried) { release_unreferenced(session, handle); }
            return 0;
        }
        OwnerToken token = 0;
        CacheRetentionPriority priority;
        std::uint32_t proven_frontier = 0;
        if (choice.resume_owner) {
            token = *choice.resume_owner;
            if (!find_owner(token)) { throw std::logic_error("resume lost continuation owner"); }
        } else {
            if (!resuming && choice.source.reused_tokens) {
                const auto used_at = request_epoch_;
                if (const auto key = base.prefix_shortlist_key(choice.source.reused_tokens)) {
                    remember_demand(*key, choice.source.reused_tokens, 0,
                                    CacheRetentionPriority{.last_demand = used_at});
                }
                if (auto* owner = find_owner(choice.owner)) {
                    priority        = {.last_demand = used_at};
                    proven_frontier = choice.source.reused_tokens;
                    // Consuming a branch replaces its proof. A shallower independent
                    // Fork has its own new evidence and leaves its donor's pair intact.
                    if (choice.take_over || proven_frontier >= owner->proven_frontier) {
                        owner->priority        = priority;
                        owner->proven_frontier = proven_frontier;
                    }
                    if (choice.take_over) { token = owner->token; }
                } else if (auto* shared = find_shared(choice.shared_entry)) {
                    shared->priority = {.last_demand = used_at};
                }
            }
            if (!token) {
                token = ++ordinal_;
                owners_.push_back({.token           = token,
                                   .priority        = priority,
                                   .proven_frontier = proven_frontier,
                                   .retained_at     = ++retention_clock_});
            }
        }
        auto* owner   = find_owner(token);
        owner->active = true;
        if (!resuming) {
            owner->recomputed_frontier = 0;
            owner->recomputed_priority = {};
        }
        owner->publication_order = publication_order;
        replace_points(session, *owner, carried);
        prune(session);
        return token;
    }

    void publish(Session& session, OwnerToken token, Handle handle) {
        if (!session.valid_checkpoint(handle)) { return; }
        const auto summary = session.checkpoint_metadata(handle);
        if (!enabled_ || !token) {
            release_unreferenced(session, handle);
            return;
        }
        if (summary.role == CheckpointRole::SharedPrefix) {
            for (auto& entry : shared_) {
                if (entry.handle == handle) return;
                const auto prior = session.checkpoint_metadata(entry.handle);
                if (prior.placement != summary.placement || prior.frontier != summary.frontier ||
                    session.checkpoint_key(entry.handle, prior.frontier) !=
                        session.checkpoint_key(handle, summary.frontier))
                    continue;
                auto identities = session.evaluate_recovery({});
                if (!identities.prefix(entry.handle, handle)) continue;
                entry.evidence |= summary.evidence;
                const auto cost = [&](Handle h) {
                    return restore_cost(session, h, summary.placement);
                };
                if (prior.leased || cost(entry.handle) <= cost(handle)) {
                    release_unreferenced(session, handle);
                    return;
                }
                const auto old = entry.handle;
                entry.handle   = handle;
                index_.insert(session, handle, summary.frontier, summary.role);
                release_unreferenced(session, old);
                return;
            }
            shared_.push_back(
                {.handle      = handle,
                 .ordinal     = ++ordinal_,
                 .priority    = demand_priority(session.checkpoint_key(handle, summary.frontier)),
                 .retained_at = ++retention_clock_,
                 .evidence    = summary.evidence});
            index_.insert(session, handle, summary.frontier, summary.role);
        } else {
            auto* owner = find_owner(token);
            if (!owner) {
                throw std::logic_error("checkpoint publication lost continuation owner");
            }
            const auto observed = demand_priority(session.checkpoint_key(handle, summary.frontier));
            if (summary.role == CheckpointRole::InputReplay && observed.reused() &&
                (!owner->priority.reused() || observed.last_demand > owner->priority.last_demand)) {
                owner->priority        = observed;
                owner->proven_frontier = summary.frontier;
            }
            if (std::none_of(owner->points.begin(), owner->points.end(),
                             [&](const auto& p) { return p.handle == handle; })) {
                std::optional<Handle> replace;
                std::size_t anchors = 0;
                std::optional<Handle> oldest_anchor;
                for (const auto& point : owner->points) {
                    if (point.role != summary.role) { continue; }
                    if (summary.role != CheckpointRole::LongAnchor) {
                        replace = point.handle;
                        break;
                    }
                    ++anchors;
                    if (!oldest_anchor) { oldest_anchor = point.handle; }
                    if (session.checkpoint_metadata(point.handle).frontier == summary.frontier) {
                        replace = point.handle;
                    }
                }
                if (!replace && anchors >= 4) { replace = oldest_anchor; }
                if (replace && !remove_point(session, *owner, *replace)) {
                    // Optional replacement never turns a legitimate request into a failure.
                    release_unreferenced(session, handle);
                    return;
                }
                owner->points.push_back({handle, summary.role});
                index_.insert(session, handle, summary.frontier, summary.role);
            }
        }
    }

    [[nodiscard]] bool
    retain_shared_checkpoint(Session& session, const Base& base, std::uint32_t frontier,
                             PlacementId placement             = {},
                             SharedCandidateEvidence evidence  = SharedCandidateEvidence::None,
                             const CaptureCostQuote* candidate = nullptr) {
        prune(session);
        const auto key = base.prefix_shortlist_key(frontier);
        if (!key) return false;
        for (auto& entry : shared_) {
            if (session.checkpoint_metadata(entry.handle).placement == placement &&
                session.checkpoint_metadata(entry.handle).frontier == frontier &&
                session.checkpoint_key(entry.handle, frontier) == *key &&
                session.checkpoint_recovery_frontier(entry.handle, base, frontier) == frontier) {
                if (candidate && capture_improves(session, entry.handle, *candidate)) continue;
                entry.evidence |= evidence;
                return true;
            }
        }
        return false;
    }

    [[nodiscard]] std::optional<Handle> capture_source(Session& session, const Base& base,
                                                       std::uint32_t frontier,
                                                       PlacementId placement,
                                                       const CaptureCostQuote* quote = nullptr) {
        prune(session);
        std::optional<Handle> best;
        std::uint64_t cost = UINT64_MAX;
        for (const auto handle : index_.matches(session, base)) {
            const auto metadata = session.checkpoint_metadata(handle);
            if (metadata.frontier != frontier || metadata.placement != placement) continue;
            if (session.checkpoint_recovery_frontier(handle, base, frontier) != frontier) continue;
            if (quote && capture_improves(session, handle, *quote)) continue;
            const auto candidate = restore_cost(session, handle, placement);
            if (!best || candidate < cost) {
                best = handle;
                cost = candidate;
            }
            if (!cost) break;
        }
        return best;
    }

    void finish(Session& session, OwnerToken token, const Base& base,
                std::uint64_t publication_order) {
        auto* owner = find_owner(token);
        if (!owner) { return; }
        owner->active            = false;
        owner->publication_order = publication_order;
        const auto& advice       = base.context_cache();
        if (advice.update_session_index && advice.session_key) {
            const auto newer = std::any_of(owners_.begin(), owners_.end(), [&](const auto& other) {
                return other.token != token && other.session == advice.session_key &&
                       other.publication_order > publication_order;
            });
            if (!newer) {
                for (auto& other : owners_) {
                    if (other.session == advice.session_key) { other.session.reset(); }
                }
                owner->session = advice.session_key;
            }
        }
        prune(session);
    }

    void abandon(Session& session, OwnerToken token) {
        auto* owner = find_owner(token);
        if (!owner) { return; }
        const auto old = points(token);
        std::erase_if(owners_, [&](const auto& entry) { return entry.token == token; });
        for (const auto handle : old) { release_unreferenced(session, handle); }
    }

    [[nodiscard]] std::vector<Handle> points(OwnerToken token) const {
        std::vector<Handle> out;
        if (const auto* owner = find_owner(token)) {
            for (const auto& point : owner->points) { out.push_back(point.handle); }
        }
        return out;
    }

    [[nodiscard]] CacheRetentionPriority retention(OwnerToken token) const {
        const auto* owner = find_owner(token);
        return owner ? owner->priority.at(request_epoch_) : CacheRetentionPriority{};
    }

    [[nodiscard]] std::optional<Handle> input_checkpoint(Session& session, OwnerToken token) const {
        if (const auto* owner = find_owner(token))
            for (const auto& point : owner->points)
                if (point.role == CheckpointRole::InputReplay &&
                    session.valid_checkpoint(point.handle))
                    return point.handle;
        return {};
    }

    // The caller supplies each request's strictly advancing, non-overlapping committed
    // prefill ranges. Replay and preparation never call this; one candidate is crossed at
    // most once by a request. Candidate frontiers are sorted by token position.
    void observe_committed(const Base& base, std::uint64_t request_id, std::uint32_t begin,
                           std::uint32_t end, std::span<const std::uint32_t> frontiers,
                           OwnerToken owner_token = 0) {
        if (!enabled_ || begin >= end) { return; }
        std::vector<std::uint32_t> crossed;
        for (auto it = demand_frontiers_.upper_bound(begin);
             it != demand_frontiers_.end() && it->first <= end; ++it)
            crossed.push_back(it->first);
        for (const auto frontier : frontiers)
            if (frontier > begin && frontier <= end) crossed.push_back(frontier);
        std::sort(crossed.begin(), crossed.end());
        crossed.erase(std::unique(crossed.begin(), crossed.end()), crossed.end());
        auto* owner = find_owner(owner_token);
        for (const auto frontier : crossed) {
            const auto key = base.prefix_shortlist_key(frontier);
            if (!key) { continue; }
            // A depth belonging only to another identity is not a new candidate.
            if (!demand_lookup_.contains(*key) &&
                !std::binary_search(frontiers.begin(), frontiers.end(), frontier))
                continue;
            remember_demand(*key, frontier, request_id);
            const auto repeated = demand_priority(key);
            if (owner && repeated.reused() && frontier >= owner->recomputed_frontier) {
                owner->recomputed_frontier = frontier;
                owner->recomputed_priority = repeated;
                owner->priority            = repeated;
                owner->proven_frontier     = std::max(owner->proven_frontier, frontier);
            }
        }
    }

    [[nodiscard]] Admission capture_admission(Session& session, OwnerToken owner, const Base& base,
                                              std::uint32_t frontier, CheckpointRole purpose,
                                              PlacementId placement         = {},
                                              const CaptureCostQuote* quote = nullptr) const {
        auto demand = demand_priority(base.prefix_shortlist_key(frontier));
        if (!demand.reused() && purpose == CheckpointRole::InputReplay) {
            if (const auto* entry = find_owner(owner);
                entry && entry->recomputed_frontier && entry->recomputed_frontier <= frontier)
                demand = entry->recomputed_priority.at(request_epoch_);
        }
        // A recurrent continuation's heat can rank its owner, but a new optional point
        // has not itself demonstrated repeat demand. Its added coverage remains bounded.
        auto priority = demand;
        if (!demand.reused() && purpose == CheckpointRole::InputReplay) priority = retention(owner);
        auto replacement = purpose == CheckpointRole::InputReplay ? input_checkpoint(session, owner)
                                                                  : std::nullopt;
        if (replacement && !session.checkpoint_recovery_frontier(*replacement, base, frontier))
            replacement.reset();
        return {.priority       = priority,
                .demand         = demand,
                .purpose        = purpose,
                .owner          = owner,
                .base           = &base,
                .frontier       = frontier,
                .replacement    = replacement,
                .placement      = placement,
                .target         = quote ? quote->retained : nullptr,
                .pending_action = quote ? action_cost(quote->work) : 0};
    }

    [[nodiscard]] ReclaimCursor begin_reclaim(Session& session,
                                              std::optional<Admission> admission = std::nullopt,
                                              ReclaimRights rights               = {}) {
        prune(session);
        auto cursor      = reclaim_order(session);
        cursor.id        = ++decision_clock_;
        cursor.admission = admission;
        cursor.rights    = admission ? ReclaimRights{ReclaimPurpose::OptionalWrite} : rights;
        if (admission) {
            for (const auto& owner : owners_)
                if (owner.active && owner.token != admission->owner)
                    for (const auto& point : owner.points)
                        if (point.role == CheckpointRole::InputReplay)
                            cursor.protected_points.push_back(point.handle);
        }
        if (admission && !admission->demand.reused()) {
            Evaluation evaluation{.cursor = &cursor};
            cursor.gain_limit = recovery_gain(session, *admission, {}, evaluation);
        }
        return cursor;
    }

    [[nodiscard]] std::optional<std::vector<Handle>>
    capture_victims(Session& session, const ContextAssessment& assessment,
                    const ReclaimCursor& cursor) const {
        if (assessment.ready()) return std::vector<Handle>{};
        if (assessment.impossible() || assessment.waiting()) return {};
        const auto referenced = referenced_handles(session);
        std::vector<Handle> allowed;
        for (const auto& victim : cursor.victims)
            if (session.valid_checkpoint(victim.handle) && referenced[victim.handle.index])
                allowed.push_back(victim.handle);
        auto plan = session.plan_releases(allowed, {}, assessment.demand);
        Evaluation evaluation{.cursor = &cursor};
        const std::array<const Admission*, 1> admissions{cursor.admission ? &*cursor.admission
                                                                          : nullptr};
        ReclaimSelection<ResourceManager> selection(*this, session, assessment, cursor, evaluation,
                                                    *plan.inventory, plan.releases, admissions);
        while (!session.assess(assessment.demand, selection.released()).ready()) {
            const auto next = selection.best();
            if (!next) return {};
            selection.commit(*next);
        }
        return std::vector<Handle>(selection.handles().begin(), selection.handles().end());
    }

    [[nodiscard]] ReclaimProgress preserve_capture(Session& session,
                                                   const ContextAssessment& assessment,
                                                   const ReclaimCursor& cursor) const {
        if (session.has_context_operation()) return ReclaimProgress::Transferring;
        const auto referenced = referenced_handles(session);
        std::vector<Handle> allowed;
        for (const auto& victim : cursor.victims)
            if (session.valid_checkpoint(victim.handle) && referenced[victim.handle.index])
                allowed.push_back(victim.handle);
        auto plans = session.plan_reclaim(allowed, {}, assessment.demand);
        Evaluation evaluation{.cursor = &cursor};
        using Demotion = typename decltype(plans.demotions)::value_type;
        std::optional<Demotion> combined;
        std::vector<bool> selected(plans.demotions.size());
        // Every trial is independent until selected. Prefer an affordable complete plan;
        // otherwise grow the cheapest useful partial plan and reprice the whole union.
        for (;;) {
            struct Choice {
                std::size_t index;
                Demotion plan;
                ReclaimActionQuote quote;
                ActionRank rank;
                bool complete;
            };

            std::optional<Choice> best;
            for (std::size_t i = 0; i < plans.demotions.size(); ++i) {
                if (selected[i]) continue;
                const auto& candidate = plans.demotions[i];
                if (!session.useful_release_bytes(candidate.released, assessment)) continue;
                auto trial = candidate;
                if (combined) {
                    const std::array parts{*combined, candidate};
                    trial = session.combine_demotions(parts);
                }
                if (!session.can_place_host(session.demotion_host_requirements(trial))) continue;
                const auto useful = session.useful_release_bytes(trial.released, assessment);
                if (combined &&
                    useful <= session.useful_release_bytes(combined->released, assessment) &&
                    !session.assess(assessment.demand, trial.released).feasible())
                    continue;
                auto facts = action_facts(session, trial.sources, cursor, evaluation);
                if (!admits(session, trial.sources, facts, cursor.admission, cursor, evaluation,
                            true))
                    continue;
                auto quote = quote_demotion(session, trial, {}, cursor, evaluation, assessment);
                if (!admits_action(session, quote, cursor, evaluation)) continue;
                auto rank = rank_action(session, trial.sources, facts, useful, evaluation, true);
                rank.loss = recovery_cost_increase(cursor.ledger->loss_after, quote.loss_after);
                rank.action_cost = add_recovery_cost(quote.action_cost, quote.followup_cost);
                if (cursor.admission &&
                    !admits_limits(rank.priority, add_recovery_cost(rank.loss, rank.action_cost),
                                   *cursor.admission, cursor))
                    continue;
                const bool complete = session.assess(assessment.demand, trial.released).feasible();
                if (!best || (complete != best->complete ? complete : less_rank(rank, best->rank)))
                    best.emplace(Choice{i, std::move(trial), std::move(quote), rank, complete});
            }
            if (!best) break;
            if (!best->complete) {
                selected[best->index] = true;
                combined              = std::move(best->plan);
                continue;
            }
            const auto started = session.start_demote(best->plan, {});
            if (started == ContextStart::Started) {
                cursor.ledger->accept(best->plan.sources, {}, best->quote.loss_after,
                                      best->quote.action_cost, best->quote.budgeted);
                return session.has_context_operation() ? ReclaimProgress::Transferring
                                                       : ReclaimProgress::Changed;
            }
            return started == ContextStart::Deferred ? ReclaimProgress::Deferred
                                                     : ReclaimProgress::Retry;
        }
        return ReclaimProgress::Blocked;
    }

    [[nodiscard]] std::optional<std::vector<Handle>>
    host_victims(Session& session, const HostRequirement& write, std::optional<Handle> incoming,
                 std::span<const Handle> later_snapshots = {},
                 std::span<const Handle> excluded        = {},
                 std::optional<Admission> admission      = std::nullopt,
                 const ReclaimCursor* decision = nullptr, ReclaimRights rights = {}) {
        prune(session);
        Evaluation evaluation{.cursor = decision};
        if (decision) {
            return host_victims_in_order(session, write, incoming, later_snapshots, excluded,
                                         decision->admission, *decision, evaluation);
        }
        auto cursor       = begin_reclaim(session, admission, rights);
        evaluation.cursor = &cursor;
        return host_victims_in_order(session, write, incoming, later_snapshots, excluded, admission,
                                     cursor, evaluation);
    }

    [[nodiscard]] ReclaimActionQuote quote_release(Session& session,
                                                   std::span<const Handle> handles,
                                                   const ReclaimCursor& cursor) const {
        include_required_snapshots(session, cursor, handles);
        Evaluation evaluation{.cursor = &cursor};
        auto& values = recovery(session, evaluation);
        return {cursor.id,
                cursor.ledger->revision,
                {handles.begin(), handles.end()},
                values.total_loss(handles),
                0,
                needs_budget(cursor, handles)};
    }

    // Read-only complete destination plan. Native prepares the selected pause before any
    // victim is consumed, and owns subsequent maintenance and relocation as one operation.
    [[nodiscard]] std::optional<std::vector<Handle>>
    pause_victims(Session& session, const ContextDemand& demand, const ReclaimCursor& cursor,
                  std::span<const Handle> excluded        = {},
                  std::span<const Handle> later_snapshots = {}) const {
        const auto assessment = session.assess(demand);
        if (assessment.feasible()) return std::vector<Handle>{};
        if (assessment.impossible() || assessment.waiting()) return {};
        include_required_snapshots(session, cursor, later_snapshots);
        Evaluation evaluation{.cursor = &cursor};
        (void)recovery(session, evaluation);
        const auto referenced = referenced_handles(session);
        std::vector<Handle> allowed;
        for (const auto& victim : cursor.victims)
            if (session.valid_checkpoint(victim.handle) && referenced[victim.handle.index])
                allowed.push_back(victim.handle);
        for (auto h : later_snapshots)
            if (session.valid_checkpoint(h) && !contains(allowed, h)) allowed.push_back(h);
        auto plan = session.plan_releases(allowed, {}, demand);
        const std::array<const Admission*, 1> admissions{nullptr};
        ReclaimSelection<ResourceManager> selection(*this, session, assessment, cursor, evaluation,
                                                    *plan.inventory, plan.releases, admissions,
                                                    excluded, later_snapshots);
        while (!session.assess(demand, selection.released()).feasible()) {
            const auto next = selection.best();
            if (!next) return {};
            selection.commit(*next);
        }
        return std::vector<Handle>(selection.handles().begin(), selection.handles().end());
    }

    [[nodiscard]] std::vector<Handle> pause_sources(Session& session) const {
        return advertised(session);
    }

    template <class PauseOption>
    [[nodiscard]] std::uint64_t
    pause_cost(Session& session, const PauseCostQuote& quote, const PauseOption& option,
               std::span<const Handle> victims, const ReclaimCursor* decision = nullptr) const {
        if (quote.unpriced_work) return UINT64_MAX;
        const auto price = [&](std::span<const ContextTransferRequirement> transfers) {
            if (!costs_.prefill_measured &&
                std::any_of(transfers.begin(), transfers.end(),
                            [](const auto& t) { return t.peer.has_value(); }))
                return UINT64_MAX;
            return action_cost(transfers);
        };
        auto future = UINT64_MAX;
        for (const auto& route : quote.recovery)
            future = std::min(future, add_recovery_cost(price(route.transfers),
                                                        costs_.prefill_ns(route.rebuild)));
        auto cost = add_recovery_cost(price(quote.work), future);
        if (victims.empty() && !option.host_write) return cost;
        if (decision) {
            const auto action = quote_pause_action(session, quote, option, victims, *decision);
            return add_recovery_cost(
                cost, recovery_cost_increase(decision->ledger->loss_after, action.loss_after));
        }
        const auto projected = pause_projection(session, option);
        RecoveryCostEvaluation<Session, Handle> values(session, advertised(session), nullptr,
                                                       costs_, projected);
        return add_recovery_cost(cost, values.total_loss(victims));
    }

    template <class PauseOption>
    ReclaimActionQuote
    quote_pause_action(Session& session, const PauseCostQuote& quote, const PauseOption& option,
                       std::span<const Handle> victims, const ReclaimCursor& cursor) const {
        const auto projected = pause_projection(session, option);
        auto changed         = std::vector<Handle>(victims.begin(), victims.end());
        for (const auto& point : projected)
            if (!contains(changed, point.handle)) changed.push_back(point.handle);
        include_required_snapshots(session, cursor, changed);
        RecoveryCostEvaluation<Session, Handle> values(session, advertised(session),
                                                       cursor.ledger.get(), costs_, projected);
        ReclaimActionQuote result{cursor.id,
                                  cursor.ledger->revision,
                                  {victims.begin(), victims.end()},
                                  values.total_loss(victims),
                                  action_cost(quote.work)};
        result.changed = std::move(changed);
        return result;
    }

    void accept_pause(Session& session, ReclaimCursor& cursor, const ReclaimActionQuote& quote) {
        if (!cursor.ledger || quote.decision != cursor.id ||
            quote.revision != cursor.ledger->revision)
            throw std::logic_error("pause differs from its decision quote");
        for (auto h : quote.retired) {
            if (session.valid_checkpoint(h))
                throw std::logic_error("pause did not retire its victims");
            forget(h, true);
        }
        cursor.ledger->accept(quote.changed, quote.retired, quote.loss_after, quote.action_cost);
        prune(session);
    }

    [[nodiscard]] std::uint64_t
    action_cost(std::span<const ContextTransferRequirement> work) const {
        return price_context_transfer_requirements(costs_, work);
    }

    [[nodiscard]] std::optional<ReclaimActionQuote>
    quote_capture(Session& session, const CaptureCostQuote& capture, const ReclaimCursor& cursor,
                  std::span<const Handle> removed = {}) const {
        Evaluation evaluation{.cursor = &cursor};
        auto& values       = recovery(session, evaluation);
        const auto loss    = values.total_loss(removed);
        const auto action  = action_cost(capture.work);
        const auto gain    = values.gain(*capture.retained, {}, true);
        const bool bounded = needs_budget(cursor, removed) || cursor.ledger->bounded_actions;
        if (!affords_action(cursor, loss, action, gain, bounded)) return {};
        return ReclaimActionQuote{
            cursor.id, cursor.ledger->revision, {removed.begin(), removed.end()}, loss, action,
            bounded};
    }

    [[nodiscard]] ReclaimProgress maintain(Session& session, const ContextAssessment& assessment,
                                           const ReclaimCursor& cursor) const {
        Evaluation evaluation{.cursor = &cursor};
        auto& values    = recovery(session, evaluation);
        const auto cost = action_cost(session.maintenance_work(assessment));
        if (cursor.admission) {
            const auto& a = *cursor.admission;
            const auto target =
                a.target ? a.target
                         : (a.base ? session.recovery_target(*a.base, a.frontier, a.placement)
                                   : nullptr);
            const auto gain =
                target ? values.gain(*target, {}, true) : recovery_gain(session, a, {}, evaluation);
            if (!affords_action(cursor, values.total_loss(),
                                add_recovery_cost(cost, a.pending_action), gain,
                                cursor.ledger->bounded_actions))
                return ReclaimProgress::Blocked;
        }
        const auto started = session.start_maintenance(assessment);
        if (started == ContextStart::Started) {
            cursor.ledger->accept({}, {}, values.total_loss(), cost);
            return session.has_context_operation() ? ReclaimProgress::Transferring
                                                   : ReclaimProgress::Changed;
        }
        return started == ContextStart::Deferred ? ReclaimProgress::Deferred
                                                 : ReclaimProgress::Retry;
    }

    void accept_action(const ReclaimCursor& cursor, const ReclaimActionQuote& quote) const {
        if (!cursor.ledger || quote.decision != cursor.id ||
            quote.revision != cursor.ledger->revision || !quote.retired.empty())
            throw std::logic_error("context action differs from its decision quote");
        cursor.ledger->accept({}, {}, quote.loss_after, quote.action_cost, quote.budgeted);
    }

    void adopt_release(Session& session, std::span<const Handle> handles, ReclaimCursor& cursor,
                       ReclaimActionQuote quote) {
        if (!cursor.ledger || quote.decision != cursor.id ||
            quote.revision != cursor.ledger->revision ||
            !std::equal(handles.begin(), handles.end(), quote.retired.begin(), quote.retired.end()))
            throw std::logic_error("cache release differs from its decision quote");
        for (const auto handle : handles) {
            if (session.valid_checkpoint(handle))
                throw std::logic_error("Session did not consume its quoted cache victim");
            forget(handle, true);
        }
        cursor.ledger->accept(handles, handles, quote.loss_after, quote.action_cost,
                              quote.budgeted);
        prune(session);
    }

    bool erase(Session& session, Handle handle, ReclaimRights rights = {}) {
        if (!references(handle)) { return false; }
        if (!may_revoke({&handle, 1}, rights)) { return false; }
        if (session.valid_checkpoint(handle) && !session.release_checkpoint(handle)) {
            return false;
        }
        forget(handle, true);
        prune(session);
        return true;
    }

    [[nodiscard]] ReclaimProgress reclaim(Session& session, const ContextAssessment& assessment,
                                          std::span<const Handle> excluded = {}) {
        if (session.has_context_operation()) return ReclaimProgress::Transferring;
        auto cursor = begin_reclaim(session);
        return reclaim(session, assessment, excluded, cursor);
    }

    [[nodiscard]] ReclaimProgress reclaim(Session& session, const ContextAssessment& assessment,
                                          std::span<const Handle> excluded, ReclaimCursor& cursor) {
        if (session.has_context_operation()) return ReclaimProgress::Transferring;
        if (assessment.waiting()) return ReclaimProgress::Deferred;
        const auto current = session.assess(assessment.demand);
        if (current.ready()) return ReclaimProgress::Retry;
        if (current.requires_maintenance()) return maintain(session, current, cursor);
        if (current.waiting()) return ReclaimProgress::Deferred;
        if (current.impossible()) return ReclaimProgress::Blocked;
        prune(session);
        std::vector<Handle> allowed;
        const auto referenced = referenced_handles(session);
        for (const auto& victim : cursor.victims) {
            if (session.valid_checkpoint(victim.handle) && referenced[victim.handle.index] &&
                !contains(excluded, victim.handle))
                allowed.push_back(victim.handle);
        }
        auto plans = session.plan_reclaim(allowed, excluded, current.demand);
        Evaluation evaluation{.cursor = &cursor};
        using Demotion = typename decltype(plans.demotions)::value_type;

        struct Action {
            Demotion demotion;
            std::vector<Handle> host_victims;
            Admission preservation;
            ActionRank rank;
        };

        const bool needs_host_cleanup =
            std::any_of(plans.demotions.begin(), plans.demotions.end(), [&](const auto& plan) {
                return !session.can_place_host(session.demotion_host_requirements(plan));
            });
        const bool existing_host_destination =
            std::any_of(plans.demotions.begin(), plans.demotions.end(),
                        [](const auto& plan) { return plan.host_bytes == 0; });
        if (needs_host_cleanup && !existing_host_destination &&
            session.release_redundant_host(excluded))
            return ReclaimProgress::Changed;

        std::vector<Action> actions;
        actions.reserve(plans.demotions.size());
        for (auto& plan : plans.demotions) {
            const auto bytes = session.released_bytes(plan.released, current);
            if (!bytes) continue;
            auto facts = action_facts(session, plan.sources, cursor, evaluation);
            if (!admits(session, plan.sources, facts, cursor.admission, cursor, evaluation, true))
                continue;
            auto protected_sources = std::vector<Handle>(excluded.begin(), excluded.end());
            protected_sources.insert(protected_sources.end(), plan.sources.begin(),
                                     plan.sources.end());
            auto preservation =
                preservation_admission(session, plan.sources, cursor, facts.priority);
            auto host = session.demotion_host_requirements(plan);
            auto victims =
                host_victims_in_order(session, host, preservation.checkpoint, {}, protected_sources,
                                      preservation, cursor, evaluation);
            if (!victims) continue;
            const auto useful = session.useful_release_bytes(plan.released, current);
            auto rank         = rank_action(session, plan.sources, facts, useful, evaluation, true);
            auto host_facts   = action_facts(session, *victims, cursor, evaluation);
            const auto host_rank = rank_action(session, *victims, host_facts, useful, evaluation);
            const auto quote = quote_demotion(session, plan, *victims, cursor, evaluation, current);
            if (!admits_action(session, quote, cursor, evaluation)) continue;
            rank.loss        = recovery_cost_increase(cursor.ledger->loss_after, quote.loss_after);
            rank.action_cost = add_recovery_cost(quote.action_cost, quote.followup_cost);
            rank.explicit_hint = rank.explicit_hint || host_rank.explicit_hint;
            rank.waiting       = host_rank.waiting;
            rank.order         = std::max(rank.order, host_rank.order);
            if (host_rank.priority.reused() &&
                (!rank.priority.reused() ||
                 host_rank.priority.last_demand > rank.priority.last_demand))
                rank.priority = host_rank.priority;
            if (cursor.admission &&
                (rank.action_cost == UINT64_MAX ||
                 !admits_limits(rank.priority, add_recovery_cost(rank.loss, rank.action_cost),
                                *cursor.admission, cursor)))
                continue;
            actions.push_back({.demotion     = std::move(plan),
                               .host_victims = std::move(*victims),
                               .preservation = std::move(preservation),
                               .rank         = rank});
        }
        std::stable_sort(actions.begin(), actions.end(),
                         [](const auto& a, const auto& b) { return less_rank(a.rank, b.rank); });
        const std::array<const Admission*, 1> admissions{cursor.admission ? &*cursor.admission
                                                                          : nullptr};
        ReclaimSelection<ResourceManager> selection(*this, session, current, cursor, evaluation,
                                                    *plans.inventory, plans.releases, admissions);
        while (!session.assess(current.demand, selection.released()).feasible()) {
            const auto next_release = selection.best();
            if (!actions.empty() &&
                (!next_release || !less_rank(next_release->rank, actions.front().rank))) {
                const auto& action = actions.front();
                // Earlier tentative deletions remain uncommitted. A preservation action
                // uses the current directory, including all of those surviving sources.
                auto plan = action.demotion;
                // Consecutive preservation actions with free Host destinations can share one
                // native transaction. Stop as soon as the complete demand fits; Host eviction
                // remains an individually ranked decision.
                if (action.host_victims.empty()) {
                    for (std::size_t next = 1;
                         next < actions.size() &&
                         !session.assess(current.demand, plan.released).feasible();
                         ++next) {
                        if (!actions[next].host_victims.empty() ||
                            (next_release && less_rank(next_release->rank, actions[next].rank)))
                            break;
                        const std::array parts{plan, actions[next].demotion};
                        auto combined = session.combine_demotions(parts);
                        if (!session.can_place_host(session.demotion_host_requirements(combined)))
                            break;
                        if (cursor.admission) {
                            const auto quote =
                                quote_demotion(session, combined, {}, cursor, evaluation, current);
                            if (!admits_action(session, quote, cursor, evaluation)) break;
                            const auto loss =
                                recovery_cost_increase(cursor.ledger->loss_after, quote.loss_after);
                            if (quote.action_cost == UINT64_MAX ||
                                !admits_limits(
                                    action.rank.priority,
                                    add_recovery_cost(loss, add_recovery_cost(quote.action_cost,
                                                                              quote.followup_cost)),
                                    *cursor.admission, cursor))
                                break;
                        }
                        plan = std::move(combined);
                    }
                }
                const auto host        = session.demotion_host_requirements(plan);
                auto protected_sources = std::vector<Handle>(excluded.begin(), excluded.end());
                protected_sources.insert(protected_sources.end(), plan.sources.begin(),
                                         plan.sources.end());
                const auto victims = host_victims_in_order(
                    session, host, action.preservation.checkpoint, {}, protected_sources,
                    action.preservation, cursor, evaluation);
                if (!victims || *victims != action.host_victims) {
                    actions.erase(actions.begin());
                    continue;
                }
                if (!may_revoke(action.host_victims, cursor.rights))
                    throw std::logic_error("demotion lacks Host victim revocation rights");
                const auto host_assessment = session.assess(session.host_demand(host));
                if (host_assessment.requires_maintenance())
                    return maintain(session, host_assessment, cursor);
                const auto quote =
                    quote_demotion(session, plan, action.host_victims, cursor, evaluation, current);
                const auto started = session.start_demote(plan, action.host_victims);
                if (started == ContextStart::Started) {
                    for (const auto victim : action.host_victims) {
                        if (session.valid_checkpoint(victim))
                            throw std::logic_error("demotion did not release its Host victims");
                        forget(victim, true);
                    }
                    auto changed = plan.sources;
                    changed.insert(changed.end(), action.host_victims.begin(),
                                   action.host_victims.end());
                    cursor.ledger->accept(changed, action.host_victims, quote.loss_after,
                                          quote.action_cost, quote.budgeted);
                    prune(session);
                    return session.has_context_operation() ? ReclaimProgress::Transferring
                                                           : ReclaimProgress::Changed;
                }
                // A failed preservation does not grant permission to delete its source.
                return started == ContextStart::Deferred ? ReclaimProgress::Deferred
                                                         : ReclaimProgress::Retry;
            }
            if (!next_release) return ReclaimProgress::Blocked;
            selection.commit(*next_release);
        }
        commit_release(session, selection.handles(), cursor, selection.loss());
        return ReclaimProgress::Changed;
    }

    void release_all(Session& session) noexcept {
        const auto ranked = ranked_victims(session);
        index_.clear();
        owners_.clear();
        shared_.clear();
        waiting_.clear();
        demand_lookup_.clear();
        demand_frontiers_.clear();
        demand_.clear();
        for (const auto& entry : ranked) {
            if (session.valid_checkpoint(entry.handle))
                (void)session.release_checkpoint(entry.handle);
        }
    }

private:
    struct WaitingSource {
        std::uint64_t request;
        SourceChoice choice;
        std::vector<Handle> held;
        std::uint32_t revocations = 0;
        bool binding              = false;
    };

    WaitingSource* find_waiting(std::uint64_t request) {
        const auto found = std::find_if(waiting_.begin(), waiting_.end(),
                                        [=](const auto& item) { return item.request == request; });
        return found == waiting_.end() ? nullptr : &*found;
    }

    const WaitingSource* find_waiting(std::uint64_t request) const {
        const auto found = std::find_if(waiting_.begin(), waiting_.end(),
                                        [=](const auto& item) { return item.request == request; });
        return found == waiting_.end() ? nullptr : &*found;
    }

    bool consumable(Handle handle, OwnerToken owner, std::uint64_t request,
                    std::optional<ReclaimRights> revocation = std::nullopt) const {
        std::size_t allowed = 0;
        if (const auto* entry = find_owner(owner)) {
            allowed = std::count_if(entry->points.begin(), entry->points.end(),
                                    [&](const auto& point) { return point.handle == handle; });
        }
        if (catalog_reference_count(handle) != allowed) { return false; }
        for (const auto& waiting : waiting_) {
            if (!contains(waiting.held, handle)) { continue; }
            if (waiting.binding) { return false; }
            if (waiting.request != request &&
                (!revocation || revocation->purpose == ReclaimPurpose::OptionalWrite ||
                 (revocation->purpose == ReclaimPurpose::FreshAdmission &&
                  waiting.request <= revocation->request))) {
                return false;
            }
            ++allowed;
        }
        return allowed != 0;
    }

    [[nodiscard]] std::uint64_t waiting_ticket(std::span<const Handle> handles) const {
        std::uint64_t oldest = 0;
        for (const auto& waiting : waiting_) {
            if (std::any_of(handles.begin(), handles.end(),
                            [&](auto handle) { return contains(waiting.held, handle); })) {
                oldest = oldest ? std::min(oldest, waiting.request) : waiting.request;
            }
        }
        return oldest;
    }

    bool may_revoke(std::span<const Handle> handles, ReclaimRights rights) const {
        for (const auto& waiting : waiting_) {
            if (!std::any_of(handles.begin(), handles.end(),
                             [&](auto handle) { return contains(waiting.held, handle); })) {
                continue;
            }
            if (waiting.binding || rights.purpose == ReclaimPurpose::OptionalWrite ||
                (rights.purpose == ReclaimPurpose::FreshAdmission &&
                 waiting.request <= rights.request)) {
                return false;
            }
        }
        return true;
    }

    struct Demand {
        DemandKey key;
        std::uint32_t frontier;
        std::uint64_t request_id;
        CacheRetentionPriority priority;
        std::uint64_t observed_at;
    };

    void remember_demand(const DemandKey& key, std::uint32_t frontier, std::uint64_t request_id,
                         std::optional<CacheRetentionPriority> adopted = std::nullopt) {
        const auto found = demand_lookup_.find(key);
        if (found != demand_lookup_.end()) {
            const auto entry = found->second;
            if (!adopted && entry->request_id == request_id) return;
            const bool recent =
                request_epoch_ - entry->observed_at < CacheRetentionPriority::request_window;
            entry->request_id  = request_id;
            entry->priority    = adopted.value_or(recent ? CacheRetentionPriority{request_epoch_}
                                                         : CacheRetentionPriority{});
            entry->observed_at = request_epoch_;
            demand_.splice(demand_.end(), demand_, entry);
            return;
        }
        if (demand_.size() >= demand_capacity_) {
            auto depth = demand_frontiers_.find(demand_.front().frontier);
            if (!--depth->second) demand_frontiers_.erase(depth);
            demand_lookup_.erase(demand_.front().key);
            demand_.pop_front();
        }
        demand_.push_back({key, frontier, request_id, adopted.value_or(CacheRetentionPriority{}),
                           request_epoch_});
        demand_lookup_.emplace(key, std::prev(demand_.end()));
        ++demand_frontiers_[frontier];
    }

    [[nodiscard]] CacheRetentionPriority demand_priority(std::optional<DemandKey> key) const {
        if (!key) { return {}; }
        const auto found = demand_lookup_.find(*key);
        return found == demand_lookup_.end() ? CacheRetentionPriority{}
                                             : found->second->priority.at(request_epoch_);
    }

    struct Point {
        Handle handle;
        CheckpointRole role;
    };

    struct Owner {
        OwnerToken token;
        CacheRetentionPriority priority;
        std::uint32_t proven_frontier     = 0;
        std::uint32_t recomputed_frontier = 0;
        CacheRetentionPriority recomputed_priority;
        std::uint64_t retained_at       = 0;
        std::uint64_t publication_order = 0;
        bool active                     = false;
        std::optional<SessionKey> session;
        std::vector<Point> points;
    };

    struct Shared {
        Handle handle;
        std::uint64_t ordinal;
        CacheRetentionPriority priority;
        std::uint64_t retained_at;
        SharedCandidateEvidence evidence = SharedCandidateEvidence::None;
    };

    static bool contains(std::span<const Handle> handles, Handle handle) {
        return std::find(handles.begin(), handles.end(), handle) != handles.end();
    }

    Owner* find_owner(OwnerToken token) {
        const auto it = std::find_if(owners_.begin(), owners_.end(),
                                     [&](const auto& owner) { return owner.token == token; });
        return it == owners_.end() ? nullptr : &*it;
    }

    const Owner* find_owner(OwnerToken token) const {
        const auto it = std::find_if(owners_.begin(), owners_.end(),
                                     [&](const auto& owner) { return owner.token == token; });
        return it == owners_.end() ? nullptr : &*it;
    }

    Shared* find_shared(std::uint64_t ordinal) {
        const auto it = std::find_if(shared_.begin(), shared_.end(),
                                     [&](const auto& entry) { return entry.ordinal == ordinal; });
        return it == shared_.end() ? nullptr : &*it;
    }

    std::size_t catalog_reference_count(Handle handle) const {
        std::size_t count = 0;
        for (const auto& owner : owners_) {
            count += std::count_if(owner.points.begin(), owner.points.end(),
                                   [&](const auto& point) { return point.handle == handle; });
        }
        count += std::count_if(shared_.begin(), shared_.end(),
                               [&](const auto& entry) { return entry.handle == handle; });
        return count;
    }

    std::size_t reference_count(Handle handle) const {
        auto count = catalog_reference_count(handle);
        for (const auto& waiting : waiting_) { count += contains(waiting.held, handle); }
        return count;
    }

    bool references(Handle handle) const { return reference_count(handle) != 0; }

    bool remove_point(Session& session, Owner& owner, Handle handle) {
        if (reference_count(handle) == 1 && session.valid_checkpoint(handle) &&
            !session.release_checkpoint(handle)) {
            return false;
        }
        std::erase_if(owner.points, [&](const auto& point) { return point.handle == handle; });
        if (!catalog_reference_count(handle)) { index_.erase(handle); }
        return true;
    }

    void forget(Handle handle, bool reclaimed = false) {
        index_.erase(handle, reclaimed);
        for (auto& owner : owners_) {
            std::erase_if(owner.points, [&](const auto& point) { return point.handle == handle; });
        }
        std::erase_if(shared_, [&](const auto& entry) { return entry.handle == handle; });
        for (auto& waiting : waiting_) {
            if (std::erase(waiting.held, handle) && !waiting.binding) { ++waiting.revocations; }
        }
    }

    void release_unreferenced(Session& session, Handle handle) {
        if (!catalog_reference_count(handle)) { index_.erase(handle); }
        if (references(handle)) { return; }
        if (session.valid_checkpoint(handle) && !session.release_checkpoint(handle)) {
            throw std::logic_error("Native could not release an unreferenced checkpoint owner");
        }
    }

    void replace_points(Session& session, Owner& owner, std::span<const Handle> carried) {
        std::vector<Handle> old;
        for (const auto& point : owner.points) { old.push_back(point.handle); }
        owner.points.clear();
        for (const auto handle : carried) {
            if (!session.valid_checkpoint(handle)) { continue; }
            const auto summary = session.checkpoint_metadata(handle);
            if (summary.role == CheckpointRole::SharedPrefix) {
                throw std::logic_error("Native carried a public point as a private continuation");
            }
            owner.points.push_back({handle, summary.role});
            index_.insert(session, handle, summary.frontier, summary.role);
        }
        for (const auto handle : old) { release_unreferenced(session, handle); }
    }

    void prune(Session& session) {
        demand_capacity_ = std::max<std::size_t>(256, session.checkpoint_extent() * 2);
        std::vector<Handle> stale;
        const auto inspect = [&](Handle handle) {
            if (!session.valid_checkpoint(handle) && !contains(stale, handle)) {
                stale.push_back(handle);
            }
        };
        for (const auto& owner : owners_) {
            for (const auto& point : owner.points) { inspect(point.handle); }
        }
        for (const auto& shared : shared_) { inspect(shared.handle); }
        for (const auto& waiting : waiting_) {
            for (const auto handle : waiting.held) { inspect(handle); }
        }
        for (const auto handle : stale) { forget(handle); }
        std::erase_if(owners_,
                      [](const auto& owner) { return !owner.active && owner.points.empty(); });
    }

    struct RankedVictim {
        Handle handle;
        CacheRetentionPriority priority;
        std::uint64_t age;
        SharedCandidateEvidence evidence = SharedCandidateEvidence::None;
    };

    std::vector<RankedVictim> ranked_victims(Session& session) const {
        std::vector<RankedVictim> ranked;
        std::vector<std::size_t> positions(session.checkpoint_extent(), SIZE_MAX);
        const auto add = [&](Handle handle, CacheRetentionPriority priority, std::uint64_t age,
                             SharedCandidateEvidence evidence = SharedCandidateEvidence::None) {
            if (!session.valid_checkpoint(handle)) return;
            evidence |= session.checkpoint_metadata(handle).evidence;
            auto& position = positions[handle.index];
            if (position == SIZE_MAX) {
                position = ranked.size();
                ranked.push_back({handle, priority, age, evidence});
                return;
            }
            auto& prior = ranked[position];
            prior.evidence |= evidence;
            if ((priority.reused() && !prior.priority.reused()) ||
                (priority.reused() == prior.priority.reused() &&
                 (priority.reused() ? priority.last_demand > prior.priority.last_demand
                                    : age > prior.age)))
                prior = {handle, priority, age, prior.evidence};
        };
        for (const auto& owner : owners_)
            for (const auto& point : owner.points)
                add(point.handle, owner.priority.at(request_epoch_), owner.retained_at);
        for (const auto& shared : shared_)
            add(shared.handle, shared.priority.at(request_epoch_), shared.retained_at,
                shared.evidence);
        for (const auto& waiting : waiting_)
            for (const auto handle : waiting.held) add(handle, {}, waiting.request);
        std::stable_sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
            if (a.priority.reused() != b.priority.reused()) return !a.priority.reused();
            return (a.priority.reused() ? a.priority.last_demand : a.age) <
                   (b.priority.reused() ? b.priority.last_demand : b.age);
        });
        return ranked;
    }

    std::vector<bool> referenced_handles(Session& session) const {
        std::vector<bool> result(session.checkpoint_extent());
        const auto add = [&](Handle h) {
            if (session.valid_checkpoint(h)) result[h.index] = true;
        };
        for (const auto& owner : owners_)
            for (const auto& point : owner.points) add(point.handle);
        for (const auto& shared : shared_) add(shared.handle);
        for (const auto& waiting : waiting_)
            for (const auto h : waiting.held) add(h);
        return result;
    }

    [[nodiscard]] ReclaimCursor reclaim_order(Session& session) const {
        ReclaimCursor cursor;
        for (const auto& entry : ranked_victims(session)) {
            const auto frontier = session.checkpoint_metadata(entry.handle).frontier;
            cursor.victims.push_back(
                {entry.handle, entry.priority,
                 demand_priority(session.checkpoint_key(entry.handle, frontier)), entry.evidence});
        }
        for (const auto& owner : owners_)
            cursor.owners.push_back({owner.token, owner.priority.at(request_epoch_),
                                     owner.proven_frontier, points(owner.token)});
        return cursor;
    }

    [[nodiscard]] std::vector<Handle> advertised(Session& session) const {
        // Only advertised entries can replace a future cache lookup. Waiting-only
        // checkpoints remain physical holders, but are not a caller's lookup fallback.
        std::vector<Handle> kept;
        std::unordered_set<std::uint32_t> seen;
        const auto include = [&](Handle handle) {
            if (session.valid_checkpoint(handle) && seen.insert(handle.index).second) {
                kept.push_back(handle);
            }
        };
        for (const auto& owner : owners_) {
            for (const auto& point : owner.points) { include(point.handle); }
        }
        for (const auto& shared : shared_) { include(shared.handle); }
        return kept;
    }

    using Recovery        = RecoveryCostEvaluation<Session, Handle>;
    using HostReleasePlan = decltype(std::declval<Session&>().plan_releases(
        std::span<const Handle>{}, std::span<const Handle>{}, ContextDemand{}));

    // These facts live only until this synchronous evaluation mutates resources or yields.
    // The fixed victim cursor may outlive them; newly published points still count as fallback.
    struct Evaluation {
        const ReclaimCursor* cursor = nullptr;
        std::optional<std::vector<Handle>> retained;
        std::optional<Recovery> recovery;

        struct Host {
            std::vector<Handle> allowed;
            HostReleasePlan plan;
            std::vector<Handle> selected;
            ContextReleaseSet released;
        };

        std::optional<Host> host;

        struct FactsIndex {
            struct Entry {
                std::optional<Handle> handle, removed;
                CacheRetentionPriority priority;
                std::size_t order  = 0;
                bool private_owner = false, explicit_hint = false;
                std::vector<std::uint32_t> proofs;
            };

            std::vector<Entry> entries;
            std::vector<const Owner*> owners;
            std::vector<std::uint64_t> removed_stamps, proof_stamps;
            std::uint64_t stamp = 0;
        };

        std::optional<FactsIndex> facts;
    };

    struct ActionFacts {
        CacheRetentionPriority priority;
        std::size_t order = 0;
        std::optional<std::uint64_t> loss;
        bool explicit_hint = false;
    };

    [[nodiscard]] std::span<const Handle> retained(Session& session, Evaluation& evaluation) const {
        if (!evaluation.retained) { evaluation.retained = advertised(session); }
        return *evaluation.retained;
    }

    template <class PauseOption>
    std::vector<RecoveryPoint<Handle>> pause_projection(Session& session,
                                                        const PauseOption& option) const {
        if (!option.host_write) return {};
        auto projected = session.recovery_points(option.released);
        auto current   = session.recovery_points();
        std::sort(current.begin(), current.end(),
                  [](const auto& a, const auto& b) { return a.handle.index < b.handle.index; });
        std::erase_if(projected, [&](const auto& point) {
            const auto old =
                std::lower_bound(current.begin(), current.end(), point.handle.index,
                                 [](const auto& p, auto index) { return p.handle.index < index; });
            if (old == current.end() || old->handle != point.handle) return true;
            const auto& a = *old->description;
            const auto& b = *point.description;
            return a.placement == b.placement && a.restores.size() == b.restores.size() &&
                   std::equal(a.restores.begin(), a.restores.end(), b.restores.begin(),
                              [](const auto& x, const auto& y) {
                                  return x.placement == y.placement && x.transfers == y.transfers &&
                                         x.replay_transfers == y.replay_transfers;
                              });
        });
        return projected;
    }

    void prepare_ledger(Session& session, const ReclaimCursor& cursor) const {
        if (!cursor.ledger)
            cursor.ledger = std::make_unique<Ledger>(session, advertised(session), costs_);
    }

    void include_required_snapshots(Session& session, const ReclaimCursor& cursor,
                                    std::span<const Handle> handles) const {
        prepare_ledger(session, cursor);
        if (cursor.admission || cursor.rights.purpose != ReclaimPurpose::Execution) return;
        std::vector<Handle> added;
        for (auto h : handles)
            if (!cursor.ledger->contains(h) && session.valid_checkpoint(h)) added.push_back(h);
        if (added.empty()) return;
        const auto points = session.recovery_points({}, added);
        cursor.ledger->include_required(session, points, advertised(session), costs_);
    }

    [[nodiscard]] Recovery& recovery(Session& session, Evaluation& evaluation) const {
        if (evaluation.cursor) prepare_ledger(session, *evaluation.cursor);
        if (!evaluation.recovery)
            evaluation.recovery.emplace(
                session, retained(session, evaluation),
                evaluation.cursor ? evaluation.cursor->ledger.get() : nullptr, costs_);
        return *evaluation.recovery;
    }

    [[nodiscard]] std::uint64_t action_loss(Session& session, std::span<const Handle> sources,
                                            ActionFacts& facts, Evaluation& evaluation) const {
        if (!facts.loss) {
            facts.loss = sources.empty() ? 0 : recovery(session, evaluation).loss(sources);
        }
        return *facts.loss;
    }

    [[nodiscard]] ActionFacts action_facts(Session& session, std::span<const Handle> sources,
                                           const ReclaimCursor& cursor,
                                           Evaluation& evaluation) const {
        auto& recovery_index = recovery(session, evaluation);
        if (!evaluation.facts) {
            auto& data = evaluation.facts.emplace();
            data.entries.resize(recovery_index.capacity());
            data.removed_stamps.resize(data.entries.size());
            data.proof_stamps.resize(cursor.owners.size());
            for (std::size_t i = 0; i < cursor.victims.size(); ++i) {
                const auto& victim = cursor.victims[i];
                if (!recovery_index.contains(victim.handle)) continue;
                auto& entry         = data.entries[victim.handle.index];
                entry.handle        = victim.handle;
                entry.priority      = victim.priority;
                entry.explicit_hint = has_shared_candidate_evidence(
                    victim.evidence, SharedCandidateEvidence::ExplicitBoundary);
                entry.order = i;
            }
            std::unordered_map<OwnerToken, const Owner*> owners;
            for (const auto& owner : owners_) owners.emplace(owner.token, &owner);
            for (std::uint32_t i = 0; i < cursor.owners.size(); ++i) {
                const auto& proof = cursor.owners[i];
                const auto found  = owners.find(proof.owner);
                data.owners.push_back(found == owners.end() ? nullptr : found->second);
                for (const auto h : proof.points) {
                    if (h.index >= data.entries.size()) continue;
                    auto& entry = data.entries[h.index];
                    if (entry.handle != h) continue;
                    entry.private_owner = true;
                    if (proof.priority.reused()) entry.proofs.push_back(i);
                }
            }
        }
        auto& data = *evaluation.facts;
        ++data.stamp;
        for (const auto h : sources) {
            data.entries[h.index].removed = h;
            data.removed_stamps[h.index]  = data.stamp;
        }
        const auto removed = [&](Handle h) {
            return h.index < data.entries.size() && data.removed_stamps[h.index] == data.stamp &&
                   data.entries[h.index].removed == h;
        };
        ActionFacts facts;
        const auto include = [&](CacheRetentionPriority value) {
            if (value.reused() &&
                (!facts.priority.reused() || value.last_demand > facts.priority.last_demand))
                facts.priority = value;
        };
        for (const auto source : sources) {
            const auto& entry = data.entries[source.index];
            if (entry.handle != source) continue;
            facts.order         = std::max(facts.order, entry.order);
            facts.explicit_hint = facts.explicit_hint || entry.explicit_hint;
            if (!entry.private_owner) include(entry.priority);
            for (const auto i : entry.proofs) {
                if (data.proof_stamps[i] == data.stamp) continue;
                data.proof_stamps[i] = data.stamp;
                const auto& proof    = cursor.owners[i];
                const auto* owner    = data.owners[i];
                bool covered         = owner != nullptr;
                for (const auto h : proof.points) {
                    if (!removed(h)) continue;
                    if (cursor.admission && cursor.admission->owner == proof.owner &&
                        cursor.admission->replacement == h)
                        continue;
                    bool fallback = false;
                    if (owner) {
                        const auto frontier = recovery_index.frontier(h);
                        for (const auto& point : owner->points) {
                            if (removed(point.handle) || !recovery_index.contains(point.handle))
                                continue;
                            const auto candidate = recovery_index.frontier(point.handle);
                            if (candidate >= proof.proven_frontier && candidate <= frontier &&
                                recovery_index.prefix(point.handle, h)) {
                                fallback = true;
                                break;
                            }
                        }
                    }
                    covered = covered && fallback;
                }
                if (!covered) include(proof.priority);
            }
        }
        return facts;
    }

    [[nodiscard]] CacheRetentionPriority action_priority(Session& session,
                                                         std::span<const Handle> sources,
                                                         const ReclaimCursor& cursor,
                                                         Evaluation& evaluation) const {
        return action_facts(session, sources, cursor, evaluation).priority;
    }

    struct ActionRank {
        CacheRetentionPriority priority;
        std::uint64_t loss        = 0;
        std::uint64_t action_cost = 0;
        std::uint64_t units       = 0;
        std::size_t order         = 0;
        std::uint64_t waiting     = 0;
        bool explicit_hint        = false;
        bool replaces_input       = false;
    };

    [[nodiscard]] ActionRank rank_action(Session& session, std::span<const Handle> sources,
                                         ActionFacts& facts, std::uint64_t units,
                                         Evaluation& evaluation, bool preserving = false) const {
        return {.priority      = facts.priority,
                .loss          = preserving ? 0 : action_loss(session, sources, facts, evaluation),
                .units         = std::max<std::uint64_t>(1, units),
                .order         = facts.order,
                .waiting       = preserving ? 0 : waiting_ticket(sources),
                .explicit_hint = facts.explicit_hint};
    }

    static bool less_rank(const ActionRank& left, const ActionRank& right) {
        if (bool(left.waiting) != bool(right.waiting)) { return !left.waiting; }
        if (left.waiting != right.waiting) { return left.waiting > right.waiting; }
        if (left.priority.reused() != right.priority.reused()) { return !left.priority.reused(); }
        const auto a =
            static_cast<unsigned __int128>(add_recovery_cost(left.loss, left.action_cost)) *
            right.units;
        const auto b =
            static_cast<unsigned __int128>(add_recovery_cost(right.loss, right.action_cost)) *
            left.units;
        if (a != b) return a < b;
        if (left.priority.last_demand != right.priority.last_demand)
            return left.priority.last_demand < right.priority.last_demand;
        if (left.explicit_hint != right.explicit_hint) return !left.explicit_hint;
        return left.order < right.order;
    }

    [[nodiscard]] Admission preservation_admission(Session& session,
                                                   std::span<const Handle> sources,
                                                   const ReclaimCursor& cursor,
                                                   CacheRetentionPriority priority) const {
        Admission result{.priority = priority,
                         .sources  = std::vector<Handle>(sources.begin(), sources.end())};
        std::uint32_t deepest = 0;
        for (const auto handle : sources) {
            const auto frontier = session.checkpoint_metadata(handle).frontier;
            if (!result.checkpoint || frontier > deepest) {
                result.checkpoint = handle;
                deepest           = frontier;
            }
            const auto found =
                std::find_if(cursor.victims.begin(), cursor.victims.end(),
                             [&](const auto& victim) { return victim.handle == handle; });
            const auto demand =
                found == cursor.victims.end() ? CacheRetentionPriority{} : found->demand;
            if (demand.reused() &&
                (!result.demand.reused() || demand.last_demand > result.demand.last_demand)) {
                result.demand = demand;
            }
        }
        if (result.demand.reused()) { result.priority = result.demand; }
        return result;
    }

    static bool needs_budget(const ReclaimCursor& cursor, std::span<const Handle> handles) {
        if (!cursor.admission || cursor.admission->demand.reused()) return false;
        if (cursor.admission->purpose != CheckpointRole::InputReplay) return true;
        return std::any_of(handles.begin(), handles.end(), [&](Handle h) {
            const auto victim = std::find_if(cursor.victims.begin(), cursor.victims.end(),
                                             [&](const auto& v) { return v.handle == h; });
            return victim != cursor.victims.end() && victim->priority.reused();
        });
    }

    bool admits_action(Session& session, const ReclaimActionQuote& quote,
                       const ReclaimCursor& cursor, Evaluation& evaluation) const {
        if (!cursor.admission) return true;
        const auto& a = *cursor.admission;
        const auto target =
            a.target
                ? a.target
                : (a.base ? session.recovery_target(*a.base, a.frontier, a.placement) : nullptr);
        const auto gain = target ? recovery(session, evaluation).gain(*target, {}, true)
                                 : recovery_gain(session, a, {}, evaluation);
        return affords_action(
            cursor, quote.loss_after,
            add_recovery_cost(quote.action_cost,
                              add_recovery_cost(quote.followup_cost, a.pending_action)),
            gain, quote.budgeted || cursor.ledger->bounded_actions);
    }

    static bool affords_action(const ReclaimCursor& cursor, std::uint64_t loss,
                               std::uint64_t action, std::uint64_t gain, bool bounded) {
        if (!cursor.admission) return true;
        const auto paid = add_recovery_cost(cursor.ledger->paid_actions, action);
        if (paid == UINT64_MAX || loss == UINT64_MAX) return false;
        // Cold InputReplay replacement keeps its explicit bootstrap exception. Current
        // copies still need positive payback, including when storage happens to be free.
        if (!paid && !bounded) return true;
        if (cursor.gain_limit) gain = std::min(gain, *cursor.gain_limit);
        const auto spent = add_recovery_cost(bounded ? loss : 0, paid);
        return spent < gain;
    }

    std::uint64_t restore_cost(Session& session, Handle h, PlacementId placement) const {
        const auto points = session.recovery_points({}, {&h, 1});
        if (points.empty()) return UINT64_MAX;
        for (const auto& restore : points.front().description->restores)
            if (restore.placement == placement) return action_cost(restore.transfers);
        return UINT64_MAX;
    }

    bool capture_improves(Session& session, Handle h, const CaptureCostQuote& quote) const {
        if (session.checkpoint_metadata(h).leased) return false;
        const auto old = restore_cost(session, h, quote.retained->placement);
        for (const auto& restore : quote.retained->restores)
            if (restore.placement == quote.retained->placement) {
                const auto cost =
                    add_recovery_cost(action_cost(quote.work), action_cost(restore.transfers));
                return cost != UINT64_MAX && cost < old;
            }
        return false;
    }

    template <class Demotion>
    ReclaimActionQuote quote_demotion(Session& session, const Demotion& plan,
                                      std::span<const Handle> victims, const ReclaimCursor& cursor,
                                      Evaluation& evaluation,
                                      const ContextAssessment& target) const {
        auto& values      = recovery(session, evaluation);
        const auto points = session.recovery_points(plan.released, plan.sources);
        auto projection   = values.project(points);
        auto released     = plan.released;
        if (!victims.empty()) {
            if (!evaluation.host ||
                !std::equal(victims.begin(), victims.end(), evaluation.host->selected.begin(),
                            evaluation.host->selected.end()))
                throw std::logic_error("demotion lost its complete Host release quote");
            released.merge(evaluation.host->released);
        }
        return {cursor.id,
                cursor.ledger->revision,
                {victims.begin(), victims.end()},
                values.total_loss(victims),
                price_context_transfer_requirements(costs_, session.demotion_work(plan, victims)),
                needs_budget(cursor, victims) || needs_budget(cursor, plan.sources),
                action_cost(session.maintenance_work(session.assess(target.demand, released)))};
    }

    [[nodiscard]] std::uint64_t recovery_gain(Session& session, const Admission& admission,
                                              std::span<const Handle> removed,
                                              Evaluation& evaluation) const {
        if (admission.base) {
            const auto target = admission.target
                                    ? admission.target
                                    : session.recovery_target(*admission.base, admission.frontier,
                                                              admission.placement);
            return recovery(session, evaluation).gain(*target, removed);
        }
        if (admission.checkpoint && session.valid_checkpoint(*admission.checkpoint)) {
            auto sources = admission.sources;
            if (sources.empty()) { sources.push_back(*admission.checkpoint); }
            return recovery(session, evaluation).loss(sources, removed);
        }
        return 0;
    }

    static bool admits_limits(CacheRetentionPriority victim, std::uint64_t loss,
                              const Admission& admission, const ReclaimCursor& cursor) {
        if (victim.reused() &&
            (!admission.priority.reused() || victim.last_demand >= admission.priority.last_demand))
            return false;
        if (admission.purpose == CheckpointRole::InputReplay && !victim.reused()) return true;
        // The fixed outer budget also covers Host victims of a preservation transfer.
        const auto spent = add_recovery_cost(
            cursor.ledger->loss_after,
            add_recovery_cost(cursor.ledger->paid_actions, admission.pending_action));
        return !cursor.gain_limit ||
               (spent < *cursor.gain_limit && loss < *cursor.gain_limit - spent);
    }

    [[nodiscard]] bool admits(Session& session, std::span<const Handle> removed, ActionFacts& facts,
                              const std::optional<Admission>& admission,
                              const ReclaimCursor& cursor, Evaluation& evaluation,
                              bool preserving = false) const {
        if (!preserving && !may_revoke(removed, cursor.rights)) { return false; }
        if (!admission) { return true; }
        const auto loss = preserving ? 0 : action_loss(session, removed, facts, evaluation);
        if (!admits_limits(facts.priority, loss, *admission, cursor)) return false;
        if (admission->demand.reused()) { return true; }
        return recovery_gain(session, *admission, preserving ? std::span<const Handle>{} : removed,
                             evaluation) > loss;
    }

    void commit_release(Session& session, std::span<const Handle> handles, ReclaimCursor& cursor,
                        std::uint64_t loss) {
        if (handles.empty()) { return; }
        if (!may_revoke(handles, cursor.rights)) {
            throw std::logic_error("cache release lacks waiting-source revocation rights");
        }
        const auto total = add_recovery_cost(cursor.ledger->loss_after, loss);
        for (const auto handle : handles) {
            if (!erase(session, handle, cursor.rights)) {
                throw std::logic_error("quoted cache release changed before commit");
            }
        }
        cursor.ledger->accept(handles, handles, total, 0, needs_budget(cursor, handles));
    }

    [[nodiscard]] std::optional<std::vector<Handle>>
    host_victims_in_order(Session& session, const HostRequirement& write,
                          std::optional<Handle> incoming, std::span<const Handle> later_snapshots,
                          std::span<const Handle> excluded, std::optional<Admission> admission,
                          const ReclaimCursor& cursor, Evaluation& evaluation) const {
        const auto demand     = session.host_demand(write);
        const auto assessment = session.assess(demand);
        if (assessment.feasible()) return std::vector<Handle>{};
        if (assessment.impossible() || assessment.waiting()) return std::nullopt;
        if (!admission && incoming) {
            admission = preservation_admission(
                session, {&*incoming, 1}, cursor,
                action_priority(session, {&*incoming, 1}, cursor, evaluation));
        }
        std::vector<Handle> allowed;
        const auto referenced = referenced_handles(session);
        for (const auto& victim : cursor.victims) {
            const auto handle = victim.handle;
            if (session.valid_checkpoint(handle) && referenced[handle.index]) {
                allowed.push_back(handle);
            }
        }
        // Paused snapshots are offered only by required recovery and keep their explicit
        // Engine order; ordinary optional writes cannot acquire these rights.
        if (!admission) {
            if (!later_snapshots.empty()) {
                include_required_snapshots(session, cursor, later_snapshots);
                evaluation.recovery.reset();
                evaluation.facts.reset();
            }
            for (const auto handle : later_snapshots) {
                if (!contains(allowed, handle) && session.valid_checkpoint(handle)) {
                    allowed.push_back(handle);
                }
            }
        }
        // Exclusions filter whole physical actions below; they do not change store facts.
        // All Device candidates and the final Host preflight reuse this synchronous inventory.
        if (!evaluation.host || evaluation.host->allowed != allowed)
            evaluation.host.emplace(
                typename Evaluation::Host{allowed, session.plan_releases(allowed, {}, demand)});
        const auto& plan       = evaluation.host->plan;
        auto protected_sources = std::vector<Handle>(excluded.begin(), excluded.end());
        if (incoming) protected_sources.push_back(*incoming);
        const std::array<const Admission*, 2> admissions{
            admission ? &*admission : nullptr, cursor.admission ? &*cursor.admission : nullptr};
        ReclaimSelection<ResourceManager> selection(*this, session, assessment, cursor, evaluation,
                                                    *plan.inventory, plan.releases, admissions,
                                                    protected_sources, later_snapshots);
        while (!session.can_place_host(write, selection.released())) {
            const auto next = selection.best();
            if (!next) return std::nullopt;
            selection.commit(*next);
        }
        evaluation.host->selected.assign(selection.handles().begin(), selection.handles().end());
        evaluation.host->released = selection.released();
        return evaluation.host->selected;
    }

    bool enabled_;
    ContextMachineCostModel costs_;
    PrefixIndex<Model> index_;
    std::vector<Owner> owners_;
    std::vector<Shared> shared_;
    std::vector<WaitingSource> waiting_;
    std::size_t demand_capacity_ = 256;
    std::list<Demand> demand_;
    std::map<DemandKey, typename std::list<Demand>::iterator> demand_lookup_;
    std::map<std::uint32_t, std::size_t> demand_frontiers_;
    std::uint64_t request_epoch_   = 0;
    std::uint64_t retention_clock_ = 0;
    std::uint64_t ordinal_         = 0;
    std::uint64_t decision_clock_  = 0;
};
} // namespace ninfer::runtime
