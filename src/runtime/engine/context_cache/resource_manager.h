#pragma once

#include "runtime/contract/resources.h"
#include "runtime/contract/placement.h"
#include "runtime/engine/context_cache/context_cost.h"
#include "runtime/engine/context_cache/prefix_index.h"
#include "runtime/engine/context_cache/types.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ninfer::runtime {

// Policy owns bounded logical continuations. Native owns checkpoint contents and every byte.
template <class Model>
class ResourceManager {
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

    using DemandKey =
        typename decltype(std::declval<const Base&>().prefix_shortlist_key(0))::value_type;

    // Optional writes carry only their own demand evidence. Necessary execution has no
    // admission restriction. The request owns base for the lifetime of the reclaim decision.
    struct Admission {
        CacheRetentionPriority priority;
        CacheRetentionPriority demand;
        const Base* base       = nullptr;
        std::uint32_t frontier = 0;
        std::optional<Handle> checkpoint;
        std::vector<Handle> sources;
        CheckpointOrigin origin = CheckpointOrigin::Semantic;
    };

    struct ReclaimCursor {
        struct Victim {
            Handle handle;
            CacheRetentionPriority priority;
            CacheRetentionPriority demand;
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
        std::uint64_t sacrificed = 0;
        ReclaimRights rights;
    };

    struct SourceChoice {
        Source source;
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
                                   std::uint64_t publication_order,
                                   std::span<const Handle> carried = {},
                                   std::span<const Handle> retired = {}) {
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
            if (choice.source.reused_tokens) {
                const auto used_at = ++clock_;
                if (const auto key = base.prefix_shortlist_key(choice.source.reused_tokens)) {
                    remember_demand(*key, 0,
                                    CacheRetentionPriority{.reused = true, .last_demand = used_at});
                }
                if (auto* owner = find_owner(choice.owner)) {
                    priority        = {.reused = true, .last_demand = used_at};
                    proven_frontier = choice.source.reused_tokens;
                    // Consuming a branch replaces its proof. A shallower independent
                    // Fork has its own new evidence and leaves its donor's pair intact.
                    if (choice.take_over || proven_frontier >= owner->proven_frontier) {
                        owner->priority        = priority;
                        owner->proven_frontier = proven_frontier;
                    }
                    if (choice.take_over) { token = owner->token; }
                } else if (auto* shared = find_shared(choice.shared_entry)) {
                    shared->priority = {.reused = true, .last_demand = used_at};
                }
            }
            if (!token) {
                token = ++ordinal_;
                owners_.push_back({.token           = token,
                                   .priority        = priority,
                                   .proven_frontier = proven_frontier,
                                   .retained_at     = ++clock_});
            }
        }
        auto* owner              = find_owner(token);
        owner->active            = true;
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
                if (summary.origin == CheckpointOrigin::Semantic)
                    entry.origin = CheckpointOrigin::Semantic;
                const auto cost = [&](Handle h) {
                    return price_context_transfer_requirements(
                        costs_, session.checkpoint_restore_requirements(h));
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
                 .retained_at = ++clock_,
                 .origin      = summary.origin});
            index_.insert(session, handle, summary.frontier, summary.role);
        } else {
            auto* owner = find_owner(token);
            if (!owner) {
                throw std::logic_error("checkpoint publication lost continuation owner");
            }
            const auto observed = demand_priority(session.checkpoint_key(handle, summary.frontier));
            if (summary.role == CheckpointRole::InputReplay && observed.reused &&
                (!owner->priority.reused || observed.last_demand > owner->priority.last_demand)) {
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

    [[nodiscard]] bool retain_shared_checkpoint(Session& session, const Base& base,
                                                std::uint32_t frontier, CheckpointOrigin origin,
                                                PlacementId placement = {}) {
        prune(session);
        const auto key = base.prefix_shortlist_key(frontier);
        if (!key) return false;
        for (auto& entry : shared_) {
            if (session.checkpoint_metadata(entry.handle).placement == placement &&
                session.checkpoint_metadata(entry.handle).frontier == frontier &&
                session.checkpoint_key(entry.handle, frontier) == *key &&
                session.inspect_source(base, entry.handle, false, {}, {}, placement)) {
                if (origin == CheckpointOrigin::Semantic) entry.origin = origin;
                return true;
            }
        }
        return false;
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
        return owner ? owner->priority : CacheRetentionPriority{};
    }

    bool recycle_input(Session& session, OwnerToken token) {
        auto* owner = find_owner(token);
        if (!owner) { return false; }
        const auto point =
            std::find_if(owner->points.begin(), owner->points.end(),
                         [](const auto& p) { return p.role == CheckpointRole::InputReplay; });
        if (point == owner->points.end()) { return false; }
        const auto handle = point->handle;
        if (session.valid_checkpoint(handle) && session.checkpoint_metadata(handle).leased) {
            return false;
        }
        return remove_point(session, *owner, handle);
    }

    // The caller supplies each request's strictly advancing, non-overlapping committed
    // prefill ranges. Replay and preparation never call this; one candidate is crossed at
    // most once by a request, including when other requests commit between its chunks.
    void observe_committed(const Base& base, std::uint64_t request_id, std::uint32_t begin,
                           std::uint32_t end, std::span<const std::uint32_t> frontiers) {
        if (!enabled_ || begin >= end) { return; }
        for (const auto frontier : frontiers) {
            if (frontier <= begin || frontier > end) { continue; }
            const auto key = base.prefix_shortlist_key(frontier);
            if (!key) { continue; }
            remember_demand(*key, request_id);
        }
    }

    [[nodiscard]] Admission
    capture_admission(Session&, OwnerToken owner, const Base& base, std::uint32_t frontier,
                      bool inherit_private    = true,
                      CheckpointOrigin origin = CheckpointOrigin::Semantic) const {
        const auto demand = demand_priority(base.prefix_shortlist_key(frontier));
        // A recurrent continuation's heat can rank its owner, but a new optional point
        // has not itself demonstrated repeat demand. Its added coverage remains bounded.
        auto priority = demand;
        if (!demand.reused && inherit_private) { priority = retention(owner); }
        return {.priority = priority,
                .demand   = demand,
                .base     = &base,
                .frontier = frontier,
                .origin   = origin};
    }

    [[nodiscard]] ReclaimCursor begin_reclaim(Session& session,
                                              std::optional<Admission> admission = std::nullopt,
                                              ReclaimRights rights               = {}) {
        prune(session);
        auto cursor      = reclaim_order(session);
        cursor.admission = admission;
        cursor.rights    = admission ? ReclaimRights{ReclaimPurpose::OptionalWrite} : rights;
        if (admission && !admission->demand.reused) {
            Evaluation evaluation;
            cursor.gain_limit = recovery_gain(session, *admission, {}, evaluation);
        }
        return cursor;
    }

    [[nodiscard]] std::optional<std::vector<Handle>>
    host_victims(Session& session, const HostRequirement& write, std::optional<Handle> incoming,
                 std::span<const Handle> later_snapshots = {},
                 std::span<const Handle> excluded        = {},
                 std::optional<Admission> admission      = std::nullopt,
                 const ReclaimCursor* decision = nullptr, ReclaimRights rights = {}) {
        prune(session);
        Evaluation evaluation;
        if (decision) {
            return host_victims_in_order(session, write, incoming, later_snapshots, excluded,
                                         decision->admission, *decision, evaluation);
        }
        auto cursor   = reclaim_order(session);
        cursor.rights = admission ? ReclaimRights{ReclaimPurpose::OptionalWrite} : rights;
        return host_victims_in_order(session, write, incoming, later_snapshots, excluded, admission,
                                     cursor, evaluation);
    }

    [[nodiscard]] std::uint64_t release_cost(Session& session,
                                             std::span<const Handle> handles) const {
        if (handles.empty()) return 0;
        Evaluation evaluation;
        return recovery(session, evaluation).loss(handles);
    }

    void adopt_release(Session& session, std::span<const Handle> handles, ReclaimCursor& cursor,
                       std::uint64_t loss) {
        for (const auto handle : handles) {
            if (session.valid_checkpoint(handle))
                throw std::logic_error("Session did not consume its quoted cache victim");
            forget(handle, true);
        }
        cursor.sacrificed =
            loss > UINT64_MAX - cursor.sacrificed ? UINT64_MAX : cursor.sacrificed + loss;
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
        if (current.requires_maintenance()) {
            const auto started = session.start_maintenance(current);
            if (started == ContextStart::Started)
                return session.has_context_operation() ? ReclaimProgress::Transferring
                                                       : ReclaimProgress::Changed;
            return started == ContextStart::Deferred ? ReclaimProgress::Deferred
                                                     : ReclaimProgress::Retry;
        }
        if (current.waiting()) return ReclaimProgress::Deferred;
        if (current.impossible()) return ReclaimProgress::Blocked;
        prune(session);
        std::vector<Handle> allowed;
        for (const auto& victim : cursor.victims) {
            if (references(victim.handle) && session.valid_checkpoint(victim.handle) &&
                !contains(excluded, victim.handle))
                allowed.push_back(victim.handle);
        }
        auto plans = session.plan_reclaim(allowed, excluded, current.demand);
        Evaluation evaluation;
        using Demotion = typename decltype(plans.demotions)::value_type;
        using Release  = typename decltype(plans.releases)::value_type;

        struct Action {
            std::optional<Demotion> demotion;
            std::optional<Release> release;
            std::vector<Handle> host_victims;
            std::optional<Admission> preservation;
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
        actions.reserve(plans.demotions.size() + plans.releases.size());
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
            rank.loss            = host_rank.loss;
            rank.waiting         = host_rank.waiting;
            rank.order           = std::max(rank.order, host_rank.order);
            if (host_rank.priority.reused &&
                (!rank.priority.reused ||
                 host_rank.priority.last_demand > rank.priority.last_demand))
                rank.priority = host_rank.priority;
            actions.push_back({.demotion     = std::move(plan),
                               .host_victims = std::move(*victims),
                               .preservation = std::move(preservation),
                               .rank         = rank});
        }
        for (auto& plan : plans.releases) {
            const auto bytes = session.released_bytes(plan.released, current);
            if (!bytes && !session.releases_metadata(plan.released, current)) continue;
            auto facts = action_facts(session, plan.sources, cursor, evaluation);
            if (!admits(session, plan.sources, facts, cursor.admission, cursor, evaluation))
                continue;
            auto rank =
                rank_action(session, plan.sources, facts,
                            session.useful_release_bytes(plan.released, current), evaluation);
            actions.push_back({.release = std::move(plan), .rank = rank});
        }
        std::stable_sort(actions.begin(), actions.end(),
                         [](const auto& a, const auto& b) { return less_rank(a.rank, b.rank); });
        auto selection = plans.inventory->select();
        for (std::size_t i = 0; i < actions.size(); ++i) {
            const auto& action = actions[i];
            if (action.demotion) {
                // Earlier tentative deletions remain uncommitted. A preservation action
                // uses the current directory, including all of those surviving sources.
                auto plan = *action.demotion;
                // Consecutive preservation actions with free Host destinations can share one
                // native transaction. Stop as soon as the complete demand fits; Host eviction
                // remains an individually ranked decision.
                if (action.host_victims.empty()) {
                    for (auto next = i + 1;
                         next < actions.size() &&
                         !session.assess(current.demand, plan.released).feasible();
                         ++next) {
                        if (!actions[next].demotion || !actions[next].host_victims.empty()) break;
                        const std::array parts{plan, *actions[next].demotion};
                        auto combined = session.combine_demotions(parts);
                        if (!session.can_place_host(session.demotion_host_requirements(combined)))
                            break;
                        plan = std::move(combined);
                    }
                }
                const auto host        = session.demotion_host_requirements(plan);
                auto protected_sources = std::vector<Handle>(excluded.begin(), excluded.end());
                protected_sources.insert(protected_sources.end(), plan.sources.begin(),
                                         plan.sources.end());
                const auto victims = host_victims_in_order(
                    session, host, action.preservation->checkpoint, {}, protected_sources,
                    action.preservation, cursor, evaluation);
                if (!victims || *victims != action.host_victims) continue;
                if (!may_revoke(action.host_victims, cursor.rights))
                    throw std::logic_error("demotion lacks Host victim revocation rights");
                const auto host_assessment = session.assess(session.host_demand(host));
                if (host_assessment.requires_maintenance()) {
                    const auto maintenance = session.start_maintenance(host_assessment);
                    if (maintenance == ContextStart::Started)
                        return session.has_context_operation() ? ReclaimProgress::Transferring
                                                               : ReclaimProgress::Changed;
                    return maintenance == ContextStart::Deferred ? ReclaimProgress::Deferred
                                                                 : ReclaimProgress::Retry;
                }
                const auto started = session.start_demote(plan, action.host_victims);
                if (started == ContextStart::Started) {
                    for (const auto victim : action.host_victims) {
                        if (session.valid_checkpoint(victim))
                            throw std::logic_error("demotion did not release its Host victims");
                        forget(victim, true);
                    }
                    if (cursor.gain_limit)
                        cursor.sacrificed = action.rank.loss > UINT64_MAX - cursor.sacrificed
                                                ? UINT64_MAX
                                                : cursor.sacrificed + action.rank.loss;
                    prune(session);
                    return session.has_context_operation() ? ReclaimProgress::Transferring
                                                           : ReclaimProgress::Changed;
                }
                // A failed preservation does not grant permission to delete its source.
                return started == ContextStart::Deferred ? ReclaimProgress::Deferred
                                                         : ReclaimProgress::Retry;
            }
            const auto before = session.released_bytes(selection.released(), current);
            auto probe        = selection.probe(action.release->sources);
            if (session.released_bytes(selection.released(), current) <= before &&
                !session.releases_metadata(action.release->released, current))
                continue;
            auto combined = action_facts(session, selection.handles(), cursor, evaluation);
            if (!admits(session, selection.handles(), combined, cursor.admission, cursor,
                        evaluation))
                continue;
            probe.commit();
            if (session.assess(current.demand, selection.released()).feasible()) break;
        }
        if (selection.handles().empty() ||
            !session.assess(current.demand, selection.released()).feasible())
            return ReclaimProgress::Blocked;
        commit_release(session, selection.handles(), cursor);
        return ReclaimProgress::Changed;
    }

    void release_all(Session& session) noexcept {
        const auto handles = victims();
        index_.clear();
        owners_.clear();
        shared_.clear();
        waiting_.clear();
        demand_.clear();
        for (const auto handle : handles) {
            if (session.valid_checkpoint(handle)) { (void)session.release_checkpoint(handle); }
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
        std::uint64_t request_id;
        CacheRetentionPriority priority;
        std::uint64_t observed_at;
    };

    void remember_demand(const DemandKey& key, std::uint64_t request_id,
                         std::optional<CacheRetentionPriority> adopted = std::nullopt) {
        auto found = std::find_if(demand_.begin(), demand_.end(),
                                  [&](const auto& entry) { return entry.key == key; });
        if (found != demand_.end()) {
            if (!adopted && found->request_id == request_id) { return; }
            found->request_id = request_id;
            found->priority =
                adopted.value_or(CacheRetentionPriority{.reused = true, .last_demand = clock_ + 1});
            if (!adopted) { ++clock_; }
            found->observed_at = clock_;
            return;
        }
        if (demand_.size() >= demand_capacity_) {
            const auto oldest =
                std::min_element(demand_.begin(), demand_.end(), [](const auto& a, const auto& b) {
                    return a.observed_at < b.observed_at;
                });
            demand_.erase(oldest);
        }
        demand_.push_back({key, request_id, adopted.value_or(CacheRetentionPriority{}), ++clock_});
    }

    [[nodiscard]] CacheRetentionPriority demand_priority(std::optional<DemandKey> key) const {
        if (!key) { return {}; }
        const auto found = std::find_if(demand_.begin(), demand_.end(),
                                        [&](const auto& entry) { return entry.key == *key; });
        return found == demand_.end() ? CacheRetentionPriority{} : found->priority;
    }

    struct Point {
        Handle handle;
        CheckpointRole role;
    };

    struct Owner {
        OwnerToken token;
        CacheRetentionPriority priority;
        std::uint32_t proven_frontier   = 0;
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
        CheckpointOrigin origin = CheckpointOrigin::Semantic;
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
        demand_capacity_ = std::max<std::size_t>(256, session.checkpoint_capacity() * 2);
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

    CacheRetentionPriority priority_for(Handle handle) const {
        CacheRetentionPriority result;
        const auto include = [&](CacheRetentionPriority priority) {
            if (priority.reused && (!result.reused || priority.last_demand > result.last_demand)) {
                result = priority;
            }
        };
        for (const auto& owner : owners_) {
            if (std::any_of(owner.points.begin(), owner.points.end(),
                            [&](const auto& point) { return point.handle == handle; })) {
                include(owner.priority);
            }
        }
        for (const auto& shared : shared_) {
            if (shared.handle == handle) { include(shared.priority); }
        }
        return result;
    }

    std::vector<Handle> victims() const {
        struct Ranked {
            Handle handle;
            CacheRetentionPriority priority;
            std::uint64_t age;
        };

        std::vector<Ranked> ranked;
        const auto add = [&](Handle handle, CacheRetentionPriority priority, std::uint64_t age) {
            const auto it = std::find_if(ranked.begin(), ranked.end(),
                                         [&](const auto& entry) { return entry.handle == handle; });
            if (it == ranked.end()) {
                ranked.push_back({handle, priority, age});
            } else if ((priority.reused && !it->priority.reused) ||
                       (priority.reused == it->priority.reused &&
                        (priority.reused ? priority.last_demand > it->priority.last_demand
                                         : age > it->age))) {
                *it = {handle, priority, age};
            }
        };
        for (const auto& owner : owners_) {
            for (const auto& point : owner.points) {
                add(point.handle, owner.priority, owner.retained_at);
            }
        }
        for (const auto& shared : shared_) {
            add(shared.handle, shared.priority, shared.retained_at);
        }
        for (const auto& waiting : waiting_) {
            for (const auto handle : waiting.held) { add(handle, {}, waiting.request); }
        }
        std::stable_sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
            if (a.priority.reused != b.priority.reused) { return !a.priority.reused; }
            return (a.priority.reused ? a.priority.last_demand : a.age) <
                   (b.priority.reused ? b.priority.last_demand : b.age);
        });
        std::vector<Handle> out;
        for (const auto& entry : ranked) { out.push_back(entry.handle); }
        return out;
    }

    [[nodiscard]] ReclaimCursor reclaim_order(Session& session) const {
        ReclaimCursor cursor;
        for (const auto handle : victims()) {
            const auto frontier = session.checkpoint_metadata(handle).frontier;
            cursor.victims.push_back({handle, priority_for(handle),
                                      demand_priority(session.checkpoint_key(handle, frontier))});
        }
        for (const auto& owner : owners_) {
            cursor.owners.push_back(
                {owner.token, owner.priority, owner.proven_frontier, points(owner.token)});
        }
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

    using Recovery =
        decltype(std::declval<Session&>().evaluate_recovery(std::span<const Handle>{}));
    using HostReleasePlan = decltype(std::declval<Session&>().plan_releases(
        std::span<const Handle>{}, std::span<const Handle>{}, ContextDemand{}));

    // These facts live only until this synchronous evaluation mutates resources or yields.
    // The fixed victim cursor may outlive them; newly published points still count as fallback.
    struct Evaluation {
        std::optional<std::vector<Handle>> retained;
        std::optional<Recovery> recovery;

        struct Incoming {
            const Base* base;
            std::uint32_t frontier;
            std::vector<Handle> candidates;
        };

        std::vector<Incoming> incoming;

        struct Host {
            std::vector<Handle> allowed;
            HostReleasePlan plan;
        };

        std::optional<Host> host;
    };

    struct ActionFacts {
        CacheRetentionPriority priority;
        std::size_t order = 0;
        std::optional<std::uint64_t> loss;
    };

    [[nodiscard]] std::span<const Handle> retained(Session& session, Evaluation& evaluation) const {
        if (!evaluation.retained) { evaluation.retained = advertised(session); }
        return *evaluation.retained;
    }

    [[nodiscard]] Recovery& recovery(Session& session, Evaluation& evaluation) const {
        if (!evaluation.recovery)
            evaluation.recovery.emplace(session.evaluate_recovery(retained(session, evaluation)));
        return *evaluation.recovery;
    }

    [[nodiscard]] std::uint64_t action_loss(Session& session, std::span<const Handle> sources,
                                            ActionFacts& facts, Evaluation& evaluation) const {
        if (!facts.loss) {
            facts.loss = sources.empty() ? 0 : recovery(session, evaluation).loss(sources);
        }
        return *facts.loss;
    }

    [[nodiscard]] CacheRetentionPriority action_priority(Session& session,
                                                         std::span<const Handle> sources,
                                                         const ReclaimCursor& cursor,
                                                         Evaluation& evaluation) const {
        CacheRetentionPriority priority;
        if (sources.empty()) { return priority; }
        const auto include = [&](CacheRetentionPriority candidate) {
            if (candidate.reused &&
                (!priority.reused || candidate.last_demand > priority.last_demand)) {
                priority = candidate;
            }
        };
        for (const auto& victim : cursor.victims) {
            if (!contains(sources, victim.handle)) { continue; }
            const auto private_owner =
                std::any_of(cursor.owners.begin(), cursor.owners.end(), [&](const auto& owner) {
                    return contains(owner.points, victim.handle);
                });
            if (!private_owner) { include(victim.priority); }
        }
        for (const auto& proof : cursor.owners) {
            if (!proof.priority.reused) { continue; }
            const auto* owner = find_owner(proof.owner);
            bool affected     = false;
            bool covered      = owner != nullptr;
            for (const auto handle : proof.points) {
                if (!contains(sources, handle)) { continue; }
                affected      = true;
                bool fallback = false;
                if (owner) {
                    const auto frontier = session.checkpoint_metadata(handle).frontier;
                    for (const auto& point : owner->points) {
                        if (contains(sources, point.handle) ||
                            !session.valid_checkpoint(point.handle)) {
                            continue;
                        }
                        const auto candidate = session.checkpoint_metadata(point.handle).frontier;
                        if (candidate < proof.proven_frontier || candidate > frontier) { continue; }
                        if (recovery(session, evaluation).prefix(point.handle, handle)) {
                            fallback = true;
                            break;
                        }
                    }
                }
                covered = covered && fallback;
            }
            // The hot grant protects already adopted progress. A later exact same-owner
            // checkpoint can cover that proof while an unproven suffix is reclaimed.
            if (affected && !covered) { include(proof.priority); }
        }
        return priority;
    }

    [[nodiscard]] ActionFacts action_facts(Session& session, std::span<const Handle> sources,
                                           const ReclaimCursor& cursor,
                                           Evaluation& evaluation) const {
        ActionFacts facts{.priority = action_priority(session, sources, cursor, evaluation)};
        if (!sources.empty()) {
            for (std::size_t i = 0; i < cursor.victims.size(); ++i) {
                if (contains(sources, cursor.victims[i].handle)) { facts.order = i; }
            }
        }
        return facts;
    }

    struct ActionRank {
        CacheRetentionPriority priority;
        std::uint64_t loss    = 0;
        std::uint64_t units   = 0;
        std::size_t order     = 0;
        std::uint64_t waiting = 0;
    };

    [[nodiscard]] ActionRank rank_action(Session& session, std::span<const Handle> sources,
                                         ActionFacts& facts, std::uint64_t units,
                                         Evaluation& evaluation, bool preserving = false) const {
        return {.priority = facts.priority,
                .loss     = preserving ? 0 : action_loss(session, sources, facts, evaluation),
                .units    = std::max<std::uint64_t>(1, units),
                .order    = facts.order,
                .waiting  = preserving ? 0 : waiting_ticket(sources)};
    }

    static bool less_rank(const ActionRank& left, const ActionRank& right) {
        if (bool(left.waiting) != bool(right.waiting)) { return !left.waiting; }
        if (left.waiting != right.waiting) { return left.waiting > right.waiting; }
        if (left.priority.reused != right.priority.reused) { return !left.priority.reused; }
        if (left.priority.reused && left.priority.last_demand != right.priority.last_demand) {
            return left.priority.last_demand < right.priority.last_demand;
        }
        if (!left.priority.reused) {
            const auto a = static_cast<unsigned __int128>(left.loss) * right.units;
            const auto b = static_cast<unsigned __int128>(right.loss) * left.units;
            if (a != b) { return a < b; }
        }
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
            if (demand.reused &&
                (!result.demand.reused || demand.last_demand > result.demand.last_demand)) {
                result.demand = demand;
            }
        }
        if (result.demand.reused) { result.priority = result.demand; }
        return result;
    }

    [[nodiscard]] const std::vector<Handle>& incoming_candidates(Session& session,
                                                                 const Admission& admission,
                                                                 Evaluation& evaluation) const {
        auto found = std::find_if(
            evaluation.incoming.begin(), evaluation.incoming.end(), [&](const auto& item) {
                return item.base == admission.base && item.frontier == admission.frontier;
            });
        if (found == evaluation.incoming.end()) {
            typename Evaluation::Incoming item{admission.base, admission.frontier, {}};
            for (const auto handle : retained(session, evaluation))
                if (session.checkpoint_recovery_frontier(handle, *admission.base,
                                                         admission.frontier))
                    item.candidates.push_back(handle);
            auto& index = recovery(session, evaluation);
            std::sort(item.candidates.begin(), item.candidates.end(),
                      [&](auto a, auto b) { return index.frontier(a) > index.frontier(b); });
            evaluation.incoming.push_back(std::move(item));
            found = std::prev(evaluation.incoming.end());
        }

        return found->candidates;
    }

    [[nodiscard]] std::uint64_t recovery_gain(Session& session, const Admission& admission,
                                              std::span<const Handle> removed,
                                              Evaluation& evaluation) const {
        if (admission.base) {
            const auto& candidates = incoming_candidates(session, admission, evaluation);
            return admission.frontier -
                   recovery(session, evaluation).best_remaining(candidates, removed);
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
        if (victim.reused &&
            (!admission.priority.reused || victim.last_demand >= admission.priority.last_demand))
            return false;
        // The fixed outer budget also covers Host victims of a preservation transfer.
        return !cursor.gain_limit || (cursor.sacrificed < *cursor.gain_limit &&
                                      loss < *cursor.gain_limit - cursor.sacrificed);
    }

    [[nodiscard]] bool admits(Session& session, std::span<const Handle> removed, ActionFacts& facts,
                              const std::optional<Admission>& admission,
                              const ReclaimCursor& cursor, Evaluation& evaluation,
                              bool preserving = false) const {
        if (!preserving && !may_revoke(removed, cursor.rights)) { return false; }
        if (!admission) { return true; }
        if (!preserving && admission->origin == CheckpointOrigin::Intermediate &&
            !admission->demand.reused &&
            std::any_of(shared_.begin(), shared_.end(), [&](const auto& entry) {
                return entry.origin == CheckpointOrigin::Semantic &&
                       contains(removed, entry.handle);
            }))
            return false;
        const auto loss = preserving ? 0 : action_loss(session, removed, facts, evaluation);
        if (!admits_limits(facts.priority, loss, *admission, cursor)) return false;
        if (admission->demand.reused) { return true; }
        return recovery_gain(session, *admission, preserving ? std::span<const Handle>{} : removed,
                             evaluation) > loss;
    }

    void commit_release(Session& session, std::span<const Handle> handles, ReclaimCursor& cursor,
                        std::optional<std::uint64_t> quoted_loss = std::nullopt) {
        if (handles.empty()) { return; }
        if (!may_revoke(handles, cursor.rights)) {
            throw std::logic_error("cache release lacks waiting-source revocation rights");
        }
        Evaluation evaluation;
        const auto loss =
            cursor.gain_limit
                ? (quoted_loss ? *quoted_loss : recovery(session, evaluation).loss(handles))
                : 0;
        for (const auto handle : handles) {
            if (!erase(session, handle, cursor.rights)) {
                throw std::logic_error("quoted cache release changed before commit");
            }
        }
        cursor.sacrificed =
            loss > UINT64_MAX - cursor.sacrificed ? UINT64_MAX : cursor.sacrificed + loss;
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
        for (const auto& victim : cursor.victims) {
            const auto handle = victim.handle;
            if (references(handle) && session.valid_checkpoint(handle)) {
                allowed.push_back(handle);
            }
        }
        // Paused snapshots are offered only by required recovery and keep their explicit
        // Engine order; ordinary optional writes cannot acquire these rights.
        if (!admission) {
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
        const auto& plan = evaluation.host->plan;
        auto selection   = plan.inventory->select();
        auto& index      = recovery(session, evaluation);

        // Scores are additive across semantic components. An owner's protection proof may
        // couple several components, so coarsen that partition before caching candidate facts.
        std::vector<std::uint32_t> parents(index.component_capacity());
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
            const auto join = [&](Handle handle) {
                if (!index.contains(handle)) return;
                const auto node = root(index.component(handle));
                if (first)
                    parents[node] = root(*first);
                else
                    first = node;
            };
            for (const auto h : proof.points) join(h);
            if (const auto* owner = find_owner(proof.owner))
                for (const auto& point : owner->points) join(point.handle);
        }
        const auto component = [&](Handle h) { return root(index.component(h)); };

        struct Group {
            std::vector<Handle> removed;
            std::uint64_t loss = 0, version = 0;
        };

        struct Part {
            std::uint32_t group;
            std::vector<Handle> additions;
            std::uint64_t version = UINT64_MAX;
            ActionFacts facts;
        };

        struct Candidate {
            std::vector<Part> parts;
            std::uint64_t waiting      = 0;
            std::size_t snapshot_order = 0;
            bool revocable = false, chosen = false;
        };

        std::vector<Group> groups(parents.size());
        std::vector<bool> committed(index.capacity());
        std::vector<Candidate> candidates;
        candidates.reserve(plan.releases.size());
        for (const auto& action : plan.releases) {
            Candidate candidate;
            if (std::any_of(action.sources.begin(), action.sources.end(),
                            [&](auto h) { return incoming == h || contains(excluded, h); })) {
                candidate.chosen = true;
                candidates.push_back(std::move(candidate));
                continue;
            }
            candidate.waiting   = waiting_ticket(action.sources);
            candidate.revocable = may_revoke(action.sources, cursor.rights);
            auto ordered        = action.sources;
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
                if (contains(action.sources, later_snapshots[i])) candidate.snapshot_order = i + 1;
            candidates.push_back(std::move(candidate));
        }

        struct Gain {
            const Admission* admission = nullptr;
            std::vector<Handle> matching, sources;
            std::vector<std::uint32_t> groups;
            std::size_t first = 0;
        };

        const auto prepare_gain = [&](const std::optional<Admission>& value) {
            Gain gain;
            if (!value) return gain;
            gain.admission = &*value;
            if (value->base)
                gain.matching = incoming_candidates(session, *value, evaluation);
            else if (value->checkpoint && index.contains(*value->checkpoint)) {
                gain.sources = value->sources;
                if (gain.sources.empty()) gain.sources.push_back(*value->checkpoint);
                for (const auto h : gain.sources)
                    if (index.contains(h)) gain.groups.push_back(component(h));
                std::sort(gain.groups.begin(), gain.groups.end());
                gain.groups.erase(std::unique(gain.groups.begin(), gain.groups.end()),
                                  gain.groups.end());
            }
            return gain;
        };
        std::array gains{prepare_gain(admission), prepare_gain(cursor.admission)};
        const auto gain_after = [&](Gain& gain, const Candidate& candidate) {
            if (gain.admission->base) {
                // Advance only through committed removals; a trial never changes this cursor.
                while (gain.first < gain.matching.size() &&
                       committed[gain.matching[gain.first].index])
                    ++gain.first;
                auto i = gain.first;
                while (i < gain.matching.size() && selection.contains(gain.matching[i])) ++i;
                const auto fallback =
                    i == gain.matching.size() ? 0 : index.frontier(gain.matching[i]);
                return std::uint64_t(gain.admission->frontier - fallback);
            }
            std::vector<Handle> unavailable;
            for (const auto group : gain.groups) {
                const auto& removed = groups[group].removed;
                unavailable.insert(unavailable.end(), removed.begin(), removed.end());
                for (const auto& part : candidate.parts)
                    if (part.group == group)
                        for (const auto h : part.additions)
                            if (!committed[h.index]) unavailable.push_back(h);
            }
            return index.loss(gain.sources, unavailable);
        };
        const auto include_priority = [](CacheRetentionPriority& current,
                                         CacheRetentionPriority next) {
            if (next.reused && (!current.reused || next.last_demand > current.last_demand))
                current = next;
        };
        const auto include_waiting = [](std::uint64_t current, std::uint64_t next) {
            return !current ? next : !next ? current : std::min(current, next);
        };
        ActionRank selected;
        std::size_t selected_snapshot = 0;
        while (!session.can_place_host(write, selection.released())) {
            std::optional<std::size_t> best;
            ActionRank best_rank;
            std::size_t best_snapshot = 0;
            for (std::size_t i = 0; i < candidates.size(); ++i) {
                auto& candidate = candidates[i];
                if (candidate.chosen || !candidate.revocable) continue;
                const auto before = session.released_bytes(selection.released(), assessment);
                auto probe        = selection.probe(plan.releases[i].sources);
                const auto amount = session.released_bytes(selection.released(), assessment);
                // Keep complete co-holder groups for metadata pressure as well: individually
                // empty catalog slots can jointly release a segment needed by this admission.
                if (amount <= before &&
                    !session.releases_metadata(plan.releases[i].released, assessment))
                    continue;
                auto rank = selected;
                for (auto& part : candidate.parts) {
                    const auto& group = groups[part.group];
                    if (part.version != group.version) {
                        auto removed = group.removed;
                        for (const auto h : part.additions)
                            if (!committed[h.index]) removed.push_back(h);
                        part.facts      = action_facts(session, removed, cursor, evaluation);
                        part.facts.loss = index.loss(removed);
                        part.version    = group.version;
                    }
                    rank.loss -= group.loss;
                    rank.loss += *part.facts.loss;
                    include_priority(rank.priority, part.facts.priority);
                    rank.order = std::max(rank.order, part.facts.order);
                }
                bool accepted = true;
                for (auto& gain : gains) {
                    if (!gain.admission) continue;
                    if (!admits_limits(rank.priority, rank.loss, *gain.admission, cursor) ||
                        (!gain.admission->demand.reused &&
                         gain_after(gain, candidate) <= rank.loss)) {
                        accepted = false;
                        break;
                    }
                }
                if (!accepted) continue;
                rank.units = std::max<std::uint64_t>(
                    1, session.useful_release_bytes(selection.released(), assessment));
                rank.waiting        = include_waiting(selected.waiting, candidate.waiting);
                const auto snapshot = std::max(selected_snapshot, candidate.snapshot_order);
                if (!best || snapshot < best_snapshot ||
                    (snapshot == best_snapshot && less_rank(rank, best_rank))) {
                    best          = i;
                    best_rank     = rank;
                    best_snapshot = snapshot;
                }
            }
            if (!best) return std::nullopt;
            auto& candidate = candidates[*best];
            auto chosen     = selection.probe(plan.releases[*best].sources);
            chosen.commit();
            for (const auto& part : candidate.parts) {
                auto& group = groups[part.group];
                for (const auto h : part.additions)
                    if (!committed[h.index]) {
                        committed[h.index] = true;
                        group.removed.push_back(h);
                    }
                group.loss = *part.facts.loss;
                ++group.version;
            }
            candidate.chosen  = true;
            selected          = best_rank;
            selected_snapshot = best_snapshot;
        }
        return std::vector<Handle>(selection.handles().begin(), selection.handles().end());
    }

    bool enabled_;
    ContextMachineCostModel costs_;
    PrefixIndex<Model> index_;
    std::vector<Owner> owners_;
    std::vector<Shared> shared_;
    std::vector<WaitingSource> waiting_;
    std::size_t demand_capacity_ = 256;
    std::vector<Demand> demand_;
    std::uint64_t clock_   = 0;
    std::uint64_t ordinal_ = 0;
};
} // namespace ninfer::runtime
