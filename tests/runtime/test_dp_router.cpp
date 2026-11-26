#include "runtime/engine/dp_router.h"

#include <array>
#include <atomic>
#include <future>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::runtime;
using Clock = DpRouter::Clock;

void check(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

PrefixKey make_key(std::uint64_t seed, std::uint32_t frontier, std::uint32_t domain = 7) {
    return PrefixKey{{seed ^ (std::uint64_t(frontier) * 0x9e3779b97f4a7c15ULL), seed + frontier},
                     frontier,
                     domain};
}

DpRouter::PendingItem make_item(DpRouter& router, std::uint32_t max_frontier = 0,
                                std::uint64_t seed = 0, std::uint64_t required_bytes = 0,
                                const std::atomic<bool>* cancelled   = nullptr,
                                Clock::time_point deadline           = Clock::time_point::max(),
                                std::size_t* key_evaluations_counter = nullptr) {
    const auto ids = router.acquire_outstanding(deadline);
    DpRouter::PendingItem item;
    item.request_id               = ids.request_id;
    item.publication_order        = ids.publication_order;
    item.cancelled                = cancelled;
    item.deadline                 = deadline;
    item.max_reusable_frontier    = max_frontier;
    item.estimated_required_bytes = required_bytes;
    if (max_frontier > 0) {
        item.key_at = [seed, key_evaluations_counter](std::uint32_t f) {
            if (key_evaluations_counter) ++(*key_evaluations_counter);
            return make_key(seed, f);
        };
    }
    return item;
}

void test_c_less_than_dp_and_non_divisible_c() {
    // Case 1: C < dp (C = 2, dp = 4, local_lanes = ceil(2/4) = 1).
    {
        DpRouter router({.replica_count        = 4,
                         .local_lanes          = 1,
                         .max_concurrency      = 2,
                         .max_pending_requests = 4});
        for (std::uint32_t r = 0; r < 4; ++r) {
            (void)router.update_replica_offer(r, {.epoch                      = 1,
                                                  .occupied_lanes             = 0,
                                                  .min_rank_free_bytes        = 1 << 20,
                                                  .min_rank_reclaimable_bytes = 1 << 20});
        }
        auto s1 = router.enqueue(make_item(router));
        auto s2 = router.enqueue(make_item(router));
        auto s3 = router.enqueue(make_item(router));
        check(s1.dispatches.size() == 1 && s1.dispatches[0].replica_id == 0, "r0 dispatch");
        check(s2.dispatches.size() == 1 && s2.dispatches[0].replica_id == 1, "r1 dispatch");
        // Even though replicas 2 and 3 have free lanes, GlobalResidencyGate caps at C = 2!
        check(s3.dispatches.empty() && router.resident_permit_count() == 2 &&
                  router.pending_count() == 1,
              "C < dp must cap global resident permits at C=2");

        router.acknowledge_dispatch(0, 1, 0);
        router.acknowledge_dispatch(1, 1, 0);
        // Releasing replica 0's slot immediately dispatches request 3 to replica 2 (round-robin).
        auto s4 = router.release_or_handoff_residency(
            0, false, ReplicaStatusOffer{.epoch = 1, .occupied_lanes = 0, .resource_revision = 1});
        check(s4.dispatches.size() == 1 && s4.dispatches[0].replica_id == 2,
              "released residency permit must dispatch queued item to next healthy replica");
    }

    // Case 2: Non-divisible C (C = 3, dp = 2, local_lanes = ceil(3/2) = 2).
    {
        DpRouter router({.replica_count        = 2,
                         .local_lanes          = 2,
                         .max_concurrency      = 3,
                         .max_pending_requests = 2});
        for (std::uint32_t r = 0; r < 2; ++r) {
            (void)router.update_replica_offer(r, {.epoch = 1, .occupied_lanes = 0});
        }
        check(router.enqueue(make_item(router)).dispatches.size() == 1, "1st of C=3");
        check(router.enqueue(make_item(router)).dispatches.size() == 1, "2nd of C=3");
        check(router.enqueue(make_item(router)).dispatches.size() == 1, "3rd of C=3");
        // 4th request queues despite 1 physical local lane still free on replica 1!
        check(router.enqueue(make_item(router)).dispatches.empty(),
              "non-divisible C=3 with 2x2 local lanes must hold 4th request in central queue");
        check(router.resident_permit_count() == 3 && router.outstanding_count() == 4,
              "permits accounting mismatch");
    }
}

void test_preemption_handoff_and_restore_priority_zero_starvation() {
    // Global C = 2 across dp = 2, local_lanes = 2, central queue full of new requests.
    DpRouter router(
        {.replica_count = 2, .local_lanes = 2, .max_concurrency = 2, .max_pending_requests = 4});
    for (std::uint32_t r = 0; r < 2; ++r) {
        (void)router.update_replica_offer(r, {.epoch = 1, .occupied_lanes = 0});
    }
    // Fill global residency C = 2 (both on replica 0/1) and queue 2 fresh contenders.
    (void)router.enqueue(make_item(router));
    (void)router.enqueue(make_item(router));
    (void)router.enqueue(make_item(router));
    (void)router.enqueue(make_item(router));
    router.acknowledge_dispatch(0, 1, 0);
    router.acknowledge_dispatch(1, 1, 0);
    check(router.resident_permit_count() == 2 && router.pending_count() == 2, "saturated setup");

    // Case A: Replica 0 pauses a victim specifically to restore a local paused_ request ->
    // direct in-place handoff! Central pending MUST NOT steal the permit.
    auto handoff = router.release_or_handoff_residency(
        0, /*handoff_to_local_restore=*/true,
        ReplicaStatusOffer{.epoch = 1, .occupied_lanes = 0, .resource_revision = 1});
    check(handoff.dispatches.empty() && handoff.restore_grants.size() == 1 &&
              handoff.restore_grants[0] == 0,
          "in-place preemption handoff must grant directly to local restore");
    check(router.try_acquire_restore_permit(0), "replica 0 must consume granted restore permit");
    check(router.resident_permit_count() == 2 && router.pending_count() == 2,
          "resident count remains 2 after in-place handoff");

    // Case B: Replica 0 has a runnable paused_ request waiting for a global residency permit while
    // central queue has 2 fresh requests waiting. When Replica 1 finishes a request, Replica 0's
    // paused_ restore MUST strictly win over central pending!
    check(!router.try_acquire_restore_permit(0),
          "restore permit cannot be acquired immediately while C=2 is full");
    auto restored = router.release_or_handoff_residency(
        1, /*handoff_to_local_restore=*/false,
        ReplicaStatusOffer{.epoch = 1, .occupied_lanes = 0, .resource_revision = 1});
    check(restored.dispatches.empty() && restored.restore_grants.size() == 1 &&
              restored.restore_grants[0] == 0,
          "waiting paused_ restore must strictly preempt central pending queue");
    check(router.try_acquire_restore_permit(0), "replica 0 takes global restore grant");
    check(router.pending_count() == 2, "central pending untouched while restore took permit");
}

void test_slow_consumer_and_dropped_handle_outstanding_gate() {
    // C = 1, max_pending = 1 => max_outstanding = 2.
    DpRouter router(
        {.replica_count = 1, .local_lanes = 1, .max_concurrency = 1, .max_pending_requests = 1});
    (void)router.update_replica_offer(0, {.epoch = 1, .occupied_lanes = 0});

    auto req1      = make_item(router);
    const auto id1 = req1.request_id;
    check(router.enqueue(std::move(req1)).dispatches.size() == 1, "req1 dispatched");
    router.acknowledge_dispatch(0, 1, 0);

    auto req2      = make_item(router);
    const auto id2 = req2.request_id;
    check(router.enqueue(std::move(req2)).dispatches.empty(), "req2 queued");

    // 3rd submission hits Overloaded because max_outstanding = 2.
    bool overloaded = false;
    try {
        (void)router.acquire_outstanding(Clock::time_point::max());
    } catch (const RequestError& err) { overloaded = (err.kind() == RequestErrorKind::Overloaded); }
    check(overloaded, "expected Overloaded when max_outstanding=2 is reached");

    // Req1 finishes on engine side (terminal snapshot published, residency released so req2 runs),
    // BUT slow consumer has NOT called wait()/destroyed handle yet -> outstanding_ stays 2!
    router.mark_request_terminal(id1);
    auto next = router.release_or_handoff_residency(
        0, false, ReplicaStatusOffer{.epoch = 1, .occupied_lanes = 0, .resource_revision = 1});
    check(next.dispatches.size() == 1 && next.dispatches[0].item.request_id == id2,
          "residency releases independently of slow consumer");
    check(router.outstanding_count() == 2,
          "slow consumer must still hold GlobalOutstandingGate permit");

    // Once consumer releases handle, outstanding drops to 1 and a new request is admitted.
    router.release_outstanding_permit();
    check(router.outstanding_count() == 1, "outstanding decremented after consumer release");
    const auto req3 = router.acquire_outstanding(Clock::time_point::max());
    router.abort_preparation(req3.request_id);
    check(router.outstanding_count() == 1, "abort_preparation releases outstanding permit");
}

void test_prefix_hints_epoch_invalidation_and_bounce_back() {
    DpRouter router(
        {.replica_count = 2, .local_lanes = 2, .max_concurrency = 4, .max_pending_requests = 4});
    for (std::uint32_t r = 0; r < 2; ++r) {
        (void)router.update_replica_offer(r, {.epoch                      = 1,
                                              .occupied_lanes             = 0,
                                              .min_rank_free_bytes        = 1024,
                                              .min_rank_reclaimable_bytes = 2048,
                                              .resource_revision          = 10});
    }

    // Replica 1 publishes prefix at frontier 128 and 512; Replica 0 publishes at 128.
    router.apply_prefix_delta(
        0, 1, {PrefixIndexDelta<PrefixKey>::Kind::Insert, make_key(42, 128), 128, {0}});
    router.apply_prefix_delta(
        1, 1, {PrefixIndexDelta<PrefixKey>::Kind::Insert, make_key(42, 128), 128, {0}});
    router.apply_prefix_delta(
        1, 1, {PrefixIndexDelta<PrefixKey>::Kind::Insert, make_key(42, 512), 512, {0}});

    std::size_t evals = 0;
    auto s1           = router.enqueue(make_item(router, /*max_frontier=*/600, /*seed=*/42,
                                                 /*required_bytes=*/512, nullptr, Clock::time_point::max(),
                                                 &evals));
    check(s1.dispatches.size() == 1 && s1.dispatches[0].replica_id == 1 &&
              s1.dispatches[0].matched_frontier == 512,
          "deepest prefix frontier on replica 1 must win");
    check(evals == 2, "only active frontiers {128, 512} should be lazily evaluated");
    router.acknowledge_dispatch(1, 0, 0);

    // Suppose Replica 1's assess_binding() finds transient fragmentation at revision 10 and
    // bounces the request back! Router must immediately reroute to Replica 0 (frontier 128)
    // without waiting on Replica 1!
    auto bounced =
        router.return_unadmitted(1, /*observed_revision=*/10, std::move(s1.dispatches[0].item));
    check(bounced.dispatches.size() == 1 && bounced.dispatches[0].replica_id == 0 &&
              bounced.dispatches[0].matched_frontier == 128,
          "bounce-back from replica 1 must immediately reroute to replica 0");
    router.acknowledge_dispatch(0, 0, 0);

    // If Replica 0 ALSO bounces at revision 10, request waits in central pending_ until either
    // replica advances resource_revision > 10!
    auto both_bounced = router.return_unadmitted(0, /*observed_revision=*/10,
                                                 std::move(bounced.dispatches[0].item));
    check(
        both_bounced.dispatches.empty() && router.pending_count() == 1 &&
            router.resident_permit_count() == 0,
        "request rejected by all replicas at current revision must wait without holding residency");

    // Unchanged revision 10 does not re-dispatch; revision 11 on Replica 1 re-dispatches!
    check(router
              .update_replica_offer(1, {.epoch                      = 1,
                                        .occupied_lanes             = 0,
                                        .min_rank_free_bytes        = 1024,
                                        .min_rank_reclaimable_bytes = 2048,
                                        .resource_revision          = 10})
              .dispatches.empty(),
          "unchanged revision must not spin bounce-back");
    auto retried = router.update_replica_offer(1, {.epoch                      = 1,
                                                   .occupied_lanes             = 0,
                                                   .min_rank_free_bytes        = 1024,
                                                   .min_rank_reclaimable_bytes = 2048,
                                                   .resource_revision          = 11});
    check(retried.dispatches.size() == 1 && retried.dispatches[0].replica_id == 1,
          "advancing resource_revision must re-dispatch bounced request");
    router.acknowledge_dispatch(1, 1, 0);
    (void)router.release_or_handoff_residency(1, false);

    // Stale epoch delta is ignored; epoch reset clears all Replica 1 hints!
    router.apply_prefix_delta(
        1, /*stale_epoch=*/0,
        {PrefixIndexDelta<PrefixKey>::Kind::Insert, make_key(42, 1024), 1024, {0}});
    router.reset_replica_epoch(1, 2);
    auto after_epoch = router.enqueue(make_item(router, 600, 42));
    check(after_epoch.dispatches.size() == 1 && after_epoch.dispatches[0].replica_id == 0 &&
              after_epoch.dispatches[0].matched_frontier == 128,
          "epoch reset on replica 1 must invalidate its hints so replica 0 wins at 128");
}

void test_concurrent_bounces_wait_for_a_revision() {
    DpRouter router({.replica_count = 2, .local_lanes = 2, .max_concurrency = 2});
    const ReplicaStatusOffer offer{.resource_revision = 10};
    for (std::uint32_t r = 0; r < 2; ++r) (void)router.update_replica_offer(r, offer);
    auto first  = router.enqueue(make_item(router));
    auto second = router.enqueue(make_item(router));
    check(first.dispatches.size() == 1 && second.dispatches.size() == 1 &&
              first.dispatches.front().replica_id != second.dispatches.front().replica_id,
          "both requests must start on different replicas");

    const auto bounce = [&](DpRouter::ScheduleOutcome outcome) {
        std::array<unsigned, 2> attempts{};
        while (!outcome.dispatches.empty()) {
            check(outcome.dispatches.size() == 1, "unexpected unrelated dispatch");
            auto grant = std::move(outcome.dispatches.front());
            check(++attempts.at(grant.replica_id) == 1, "retried the same replica and revision");
            outcome = router.return_unadmitted(grant.replica_id, 10, std::move(grant.item));
        }
        check(attempts[0] == 1 && attempts[1] == 1, "request did not try both replicas");
    };
    auto other = std::async(std::launch::async, bounce, std::move(first));
    bounce(std::move(second));
    other.get();
    check(router.pending_count() == 2 && router.resident_permit_count() == 0,
          "dual bounce must release both residency permits");
    check(router.sweep_and_schedule().dispatches.empty(), "unchanged offers rerouted bounces");

    const auto retry = router.update_replica_offer(1, {.resource_revision = 11});
    check(retry.dispatches.size() == 2 && retry.dispatches[0].replica_id == 1 &&
              retry.dispatches[1].replica_id == 1,
          "revision advance did not immediately retry both requests on the changed replica");
}

void test_restore_cancellation_and_pending_handoff() {
    DpRouter router({.replica_count = 2, .local_lanes = 3, .max_concurrency = 3});
    const auto first  = router.enqueue(make_item(router));
    const auto second = router.enqueue(make_item(router));
    auto pending      = router.enqueue(make_item(router));
    const auto local  = pending.dispatches.at(0).replica_id;
    const auto remote = second.dispatches.at(0).replica_id;
    check(local == first.dispatches.at(0).replica_id, "expected two permits on local replica");
    router.acknowledge_dispatch(local, 1, 1);
    router.acknowledge_dispatch(local, 2, 2);
    router.acknowledge_dispatch(remote, 1, 1);
    for (unsigned i = 0; i < 64; ++i)
        check(!router.try_acquire_restore_permit(local), "global limit bypassed by a retry");

    const auto blocked = ReplicaStatusOffer{.occupied_lanes      = 1,
                                            .resource_revision   = 10,
                                            .lane_sequence       = 3,
                                            .fresh_lanes_blocked = true};
    auto handoff =
        router.return_unadmitted(local, 10, std::move(pending.dispatches[0].item), blocked, true);
    check(handoff.dispatches.empty() && handoff.restore_grants == std::vector{local} &&
              router.resident_permit_count() == 3 && router.pending_count() == 1,
          "pending request did not transfer its permit to the local paused owner");

    // A second release racing with the unclaimed grant must return its own permit.
    auto released = router.release_or_handoff_residency(
        local, true,
        ReplicaStatusOffer{
            .resource_revision = 10, .lane_sequence = 4, .fresh_lanes_blocked = true});
    check(released.restore_grants.empty() && released.dispatches.size() == 1 &&
              released.dispatches.front().replica_id == remote &&
              !released.dispatches.front().item.rejected_revisions[local],
          "local handoff duplicated a granted restore or failed to unblock remote work");

    router.acknowledge_dispatch(remote, 2, 2);
    auto fresh = router.enqueue(make_item(router));
    check(fresh.dispatches.empty(), "grant was not counted as resident");
    auto cancelled = router.cancel_waiting_restore(local);
    check(cancelled.restore_grants.empty() && cancelled.dispatches.size() == 1 &&
              cancelled.dispatches.front().replica_id == remote,
          "cancel-after-grant did not return the permit and dispatch a waiting request");
    check(router.resident_permit_count() == 3, "could not saturate C after grant cancellation");
    check(router.cancel_waiting_restore(local).dispatches.empty() &&
              router.resident_permit_count() == 3,
          "cancelling twice released a permit owned by another request");
}

void test_slot_acknowledgements_and_capacity_tier() {
    DpRouter router({.replica_count = 2, .local_lanes = 2, .max_concurrency = 3});
    (void)router.update_replica_offer(0, {.min_rank_free_bytes       = 80,
                                          .metadata_free_bytes       = 9,
                                          .first_unit_bytes          = 70,
                                          .first_unit_metadata_bytes = 10});
    (void)router.update_replica_offer(1, {.min_rank_free_bytes       = 75,
                                          .metadata_free_bytes       = 10,
                                          .first_unit_bytes          = 70,
                                          .first_unit_metadata_bytes = 10});
    check(router.enqueue(make_item(router)).dispatches.at(0).replica_id == 1,
          "immediately bindable metadata tier must outrank nominally freer replica");
    check(router.enqueue(make_item(router)).dispatches.at(0).replica_id == 1,
          "second request should use replica 1's second lane");
    router.acknowledge_dispatch(1, 2, 2);
    router.acknowledge_dispatch(1, 1, 1);
    (void)router.update_replica_offer(1, {.occupied_lanes = 1, .lane_sequence = 1});
    check(router.enqueue(make_item(router)).dispatches.at(0).replica_id == 0,
          "old acknowledgement or offer reopened an occupied lane");
}

void test_bounded_bypass_charges_only_admitted_requests() {
    DpRouter router({.replica_count = 1, .local_lanes = 2, .max_concurrency = 2});
    for (unsigned i = 0; i < 8; ++i) {
        auto blocked               = make_item(router);
        blocked.rejected_revisions = {UINT64_MAX};
        check(router.enqueue(std::move(blocked)).dispatches.empty(), "expected blocked head");
    }

    for (unsigned i = 0; i < 3; ++i) {
        auto younger = router.enqueue(make_item(router));
        check(younger.dispatches.size() == 1, "bounded bypass did not reach a runnable request");
        auto returned = router.return_unadmitted(0, 0, std::move(younger.dispatches.front().item));
        check(returned.dispatches.empty() && router.resident_permit_count() == 0,
              "bounce lost its revision barrier or leaked a residency permit");
    }

    for (unsigned i = 0; i < 2; ++i) {
        auto younger = router.enqueue(make_item(router));
        check(younger.dispatches.size() == 1, "bounces burned a blocked request's bypass budget");
        router.acknowledge_dispatch(0, 1, 2 * i + 1);
        const auto id = younger.dispatches[0].item.request_id;
        router.commit_admission(id);
        router.mark_request_terminal(id);
        router.release_outstanding_permit();
        (void)router.release_or_handoff_residency(0, false,
                                                  ReplicaStatusOffer{.lane_sequence = 2 * i + 2});
    }
    check(router.enqueue(make_item(router)).dispatches.empty(), "exhausted bypass budget ignored");
}

void test_submission_cut_fence_and_failure_isolation() {
    DpRouter router(
        {.replica_count = 2, .local_lanes = 2, .max_concurrency = 2, .max_pending_requests = 4});
    for (std::uint32_t r = 0; r < 2; ++r) {
        (void)router.update_replica_offer(r, {.epoch = 1, .occupied_lanes = 0});
    }

    auto pre1      = make_item(router);
    const auto id1 = pre1.request_id;
    check(router.enqueue(std::move(pre1)).dispatches.size() == 1, "pre1 dispatched");
    router.acknowledge_dispatch(0, 1, 0);

    // Begin submission cut: id1 <= cut_request_id.
    const auto fence = router.begin_submission_cut();
    check(fence.cut_request_id == id1 && fence.healthy_participants.size() == 2,
          "cut_request_id must equal last assigned request_id");

    // Post-cut submit with short deadline times out while fence is open.
    bool timed_out = false;
    try {
        (void)router.acquire_outstanding(Clock::now() + std::chrono::milliseconds(5));
    } catch (const RequestError& err) {
        timed_out = (err.kind() == RequestErrorKind::QueueTimeout);
    }
    check(timed_out, "post-cut submission must respect pending_deadline while fence is open");

    // Completing id1 on worker thread unblocks wait_pre_cut_terminal() even before consumer
    // releases its GenerationHandle!
    std::thread finisher([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        router.mark_request_terminal(id1);
    });
    check(router.wait_pre_cut_terminal(fence.healthy_participants),
          "wait_pre_cut_terminal must succeed once pre-cut request publishes terminal state");
    finisher.join();
    router.end_submission_cut();
    (void)router.release_or_handoff_residency(0, false);
    router.release_outstanding_permit();

    // Failure isolation: isolate replica 1 -> idempotent on 2nd call; replica 0 stays healthy;
    // isolating replica 0 (last healthy) drains central pending with Unavailable.
    (void)router.enqueue(make_item(router)); // occupies r0 (C=2, 1st slot)
    (void)router.enqueue(make_item(router)); // occupies r1 -> wait, let's isolate r1 while 1 queued
    router.acknowledge_dispatch(0, 1, 0);
    router.acknowledge_dispatch(1, 1, 0);
    (void)router.enqueue(make_item(router)); // queued in central pending_

    auto iso1 = router.mark_replica_unhealthy(1);
    check(iso1.newly_isolated && iso1.healthy_left == 1, "replica 1 isolated");
    auto iso1_dup = router.mark_replica_unhealthy(1);
    check(!iso1_dup.newly_isolated && iso1_dup.healthy_left == 1,
          "mark_replica_unhealthy must be idempotent");
    // Step 2 of replica 1 cleanup releases its held residency permit -> queued request routes to
    // r0!
    auto rerouted = router.release_failed_replica_residencies(1);
    check(rerouted.dispatches.size() == 1 && rerouted.dispatches[0].replica_id == 0,
          "surviving replica 0 must absorb queued request after replica 1 releases residency");

    // Queue one more and kill last healthy replica 0 -> drained as Unavailable!
    router.acknowledge_dispatch(0, 2, 0);
    auto queued          = make_item(router);
    const auto queued_id = queued.request_id;
    (void)router.enqueue(std::move(queued));
    auto iso0 = router.mark_replica_unhealthy(0);
    check(iso0.newly_isolated && iso0.healthy_left == 0 && !router.is_available(),
          "last healthy replica failure must mark router unavailable");
    const auto unavailable = router.take_unavailable();
    check(unavailable && unavailable->request_id == queued_id && !router.take_unavailable(),
          "central pending must fail with Unavailable when last healthy replica dies");
}

} // namespace

int main() {
    try {
        test_c_less_than_dp_and_non_divisible_c();
        test_preemption_handoff_and_restore_priority_zero_starvation();
        test_slow_consumer_and_dropped_handle_outstanding_gate();
        test_prefix_hints_epoch_invalidation_and_bounce_back();
        test_submission_cut_fence_and_failure_isolation();
        test_concurrent_bounces_wait_for_a_revision();
        test_restore_cancellation_and_pending_handoff();
        test_slot_acknowledgements_and_capacity_tier();
        test_bounded_bypass_charges_only_admitted_requests();
        std::cout << "ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
