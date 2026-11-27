#include "runtime/executor/preparation.h"
#include "core/device.h"
#include "distributed/communication/preparation.h"
#include "runtime/engine/startup_memory.h"
#include "runtime/engine/placement_selection.h"

#include <algorithm>
#include <map>
#include <iostream>
#include <set>
#include <thread>

namespace ninfer::runtime {
namespace {
namespace dist = distributed;
namespace wire = dist::wire;

void enter_subset(dist::ExecutionGroup& group, std::span<const dist::RankId> ranks,
                  const GroupPreparation& prepared, std::uint32_t attempt, PreparationAction action,
                  std::optional<std::uint64_t> override_capacity                = std::nullopt,
                  const std::map<std::uint32_t, std::size_t>* host_reservations = nullptr) {
    std::vector<dist::OutgoingMessage> messages;
    messages.reserve(ranks.size());
    for (const auto rank_id : ranks) {
        const auto* capacity =
            prepared.capacities.empty() ? nullptr : &prepared.capacities.at(rank_id);
        wire::Encoder out;
        out.u64(group.topology().plan);
        out.u32(attempt);
        out.u8(static_cast<std::uint8_t>(action));
        out.u64(override_capacity ? *override_capacity : (capacity ? capacity->device_bytes : 0));
        out.u64(capacity ? capacity->control_reservation_bytes : 0);
        out.u64(capacity ? capacity->planned_slack_bytes : 0);
        out.u64(host_reservations ? host_reservations->at(group.topology().rank(rank_id).node) : 0);
        messages.push_back(
            {rank_id, std::make_shared<const wire::Message>(
                          wire::Message{wire::Kind::Enter, group.epoch(), std::move(out).take()})});
    }
    if (!group.try_send(messages, distributed::Traffic::Continuation))
        throw std::runtime_error("preparation control queue is full");
}

std::map<dist::RankId, wire::Payload> receive_subset(dist::ExecutionGroup& group,
                                                     std::span<const dist::RankId> expected_ranks,
                                                     wire::Kind kind,
                                                     std::chrono::milliseconds timeout) {
    std::set<dist::RankId> pending(expected_ranks.begin(), expected_ranks.end());
    std::map<dist::RankId, wire::Payload> out;
    const auto until = std::chrono::steady_clock::now() + timeout;
    while (!pending.empty()) {
        if (std::chrono::steady_clock::now() >= until)
            throw std::runtime_error("rank preparation phase timed out");
        if (auto msg = group.poll()) {
            if (msg->message.kind == wire::Kind::StartupProgress) continue;
            if (msg->message.epoch != group.epoch() || msg->message.kind != kind ||
                !pending.erase(msg->peer))
                throw std::runtime_error("unexpected message during rank preparation phase");
            out.emplace(msg->peer, std::move(msg->message.payload));
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    return out;
}

std::vector<dist::RankId> all_ranks(const dist::ExecutionTopology& topology) {
    std::vector<dist::RankId> ranks;
    ranks.reserve(topology.ranks.size());
    for (const auto& r : topology.ranks) ranks.push_back(r.rank);
    return ranks;
}

using PhysicalPoolKey = std::pair<std::uint32_t, int>;

PhysicalPoolKey pool_key_for(const dist::ExecutionTopology& topology,
                             std::span<const RankPreparationReport> reports, dist::RankId rank) {
    const auto& id    = topology.rank(rank);
    const bool shared = reports[rank].capacity.shared_physical_memory;
    return {id.node, shared ? -1 : id.device};
}

} // namespace

std::vector<std::vector<dist::RankId>>
build_materialization_waves(const dist::ExecutionTopology& topology,
                            std::span<const RankPreparationReport> reports,
                            const ClusterHostReservations& reservations) {
    std::map<std::uint32_t, std::size_t> host_available;
    for (const auto& rank : topology.ranks) {
        const auto available      = reports[rank.rank].capacity.available_host_bytes;
        const auto [it, inserted] = host_available.try_emplace(rank.node, available);
        if (!inserted) it->second = std::min(it->second, available);
    }
    for (auto& [node, bytes] : host_available)
        bytes -= std::min(bytes, reservations.by_node.at(node));

    std::vector<std::vector<dist::RankId>> waves;
    auto pending = all_ranks(topology);
    while (!pending.empty()) {
        auto& wave = waves.emplace_back();
        std::set<PhysicalPoolKey> pools;
        std::map<std::uint32_t, std::size_t> wave_available;
        std::erase_if(pending, [&](dist::RankId id) {
            const auto key = pool_key_for(topology, reports, id);
            if (pools.contains(key)) return false;
            const auto node                = key.first;
            const auto& report             = reports[id];
            auto& available                = host_available.at(node);
            const auto [it, first_on_node] = wave_available.try_emplace(node, available);
            auto& remaining                = it->second;
            const auto device_bytes =
                report.capacity.shared_physical_memory ? report.weight_bytes : 0;
            const auto host_bytes = report.host_weight_bytes;
            const auto staging    = report.staging_bytes;
            // A rank that exceeds the planning allowance runs alone on its node and uses
            // the live worker-side memory check. Staging is released between waves.
            const bool fits = device_bytes <= remaining && host_bytes <= remaining - device_bytes &&
                              staging <= remaining - device_bytes - host_bytes;
            if (!first_on_node && !fits) return false;
            remaining = fits ? remaining - device_bytes - host_bytes - staging : 0;
            available -= std::min<std::uint64_t>(available, device_bytes);
            available -= std::min<std::uint64_t>(available, host_bytes);
            wave.push_back(id);
            pools.insert(key);
            return true;
        });
    }
    return waves;
}

std::vector<std::vector<dist::RankId>>
build_graph_waves(const dist::ExecutionTopology& topology,
                  std::span<const RankPreparationReport> reports) {
    if (reports.size() != topology.ranks.size())
        throw std::invalid_argument("graph preparation is missing rank reports");
    std::vector<std::vector<dist::RankId>> waves;
    std::vector<std::set<PhysicalPoolKey>> wave_pools;
    std::vector<bool> included(reports.size());
    for (const auto& rep : topology.replicas) {
        if (included.at(rep.ranks.front())) continue;
        std::vector<dist::RankId> cohort;
        const auto include_replica = [&](dist::RankId peer) {
            for (auto rank : topology.replica_for_rank(peer).ranks) {
                if (included.at(rank)) continue;
                included[rank] = true;
                cohort.push_back(rank);
            }
        };
        include_replica(rep.ranks.front());
        std::set<PhysicalPoolKey> rep_pools;
        for (std::size_t i = 0; i < cohort.size(); ++i) {
            const auto rank = cohort[i];
            rep_pools.insert(pool_key_for(topology, reports, rank));
            for (auto id : reports[rank].graph_groups) {
                const auto& group = topology.group(id);
                (void)group.index(rank);
                for (auto peer : group.members) {
                    const auto& declared = reports[peer].graph_groups;
                    if (std::ranges::find(declared, id) == declared.end())
                        throw std::invalid_argument("graph group missing from peer preparation");
                    include_replica(peer);
                }
            }
        }
        std::size_t w = 0;
        for (; w < waves.size(); ++w) {
            const bool conflict =
                std::any_of(rep_pools.begin(), rep_pools.end(),
                            [&](const auto& k) { return wave_pools[w].contains(k); });
            if (!conflict) break;
        }
        if (w == waves.size()) {
            waves.emplace_back();
            wave_pools.emplace_back();
        }
        waves[w].insert(waves[w].end(), cohort.begin(), cohort.end());
        wave_pools[w].insert(rep_pools.begin(), rep_pools.end());
    }
    return waves;
}

namespace {

bool preserves_cluster_capacity(std::span<const ContextCapacityResolution> narrow,
                                std::span<const ContextCapacityResolution> baseline) {
    if (narrow.size() != baseline.size()) return false;
    for (std::size_t i = 0; i < narrow.size(); ++i)
        if (!keeps_context_capacity(narrow[i].device_bytes, baseline[i].device_bytes)) return false;
    return true;
}

struct AttemptFailure {
    std::string message;
    std::vector<dist::RankId> local_allocation_failures;
    std::map<PhysicalPoolKey, std::size_t> pool_shortfalls;
    std::size_t directory_shortfall = 0;
    bool terminal                   = false;
};

std::vector<RankCapacityOffer> capacity_offers(std::span<const RankPreparationReport> reports) {
    std::vector<RankCapacityOffer> offers;
    offers.reserve(reports.size());
    for (const auto& rep : reports) {
        auto offer            = rep.capacity;
        offer.available_bytes = rep.raw_available_bytes;
        offers.push_back(std::move(offer));
    }
    return offers;
}

std::vector<ContextCapacityResolution>
cluster_baseline_capacity_upper_bound(std::vector<RankCapacityOffer> offers,
                                      const CoordinatorMemoryOffer& coordinator,
                                      EnginePurpose purpose) {
    const auto policy = ContextCapacityPolicy::automatic();
    for (auto& offer : offers) offer.graph_reserve_bytes = 0;
    auto minimum = offers;
    for (auto& offer : minimum)
        offer.maximum_runtime_bytes = offer.plan.reservation_bytes(offer.plan.minimum_capacity());

    std::vector<ContextCapacityResolution> bounds;
    bounds.reserve(offers.size());
    for (std::size_t rank = 0; rank < offers.size(); ++rank) {
        // A rank's upper bound leaves only the mandatory backing and metadata for its peers.
        std::swap(minimum[rank], offers[rank]);
        bounds.push_back(
            resolve_cluster_context_capacity(policy, minimum, coordinator, purpose)[rank]);
        std::swap(minimum[rank], offers[rank]);
    }
    return bounds;
}

} // namespace

void encode_preparation_request(wire::Encoder& out, const RankPreparationRequest& r) {
    out.u64(r.plan);
    out.u64(r.memory_headroom_bytes);
    out.bytes(r.model_payload);
}

RankPreparationRequest decode_preparation_request(wire::Decoder& in) {
    return {in.u64(), in.u64(), in.bytes()};
}

void encode_preparation_report(wire::Encoder& out, const RankPreparationReport& r) {
    const auto& plan = r.capacity.plan;
    out.u64(plan.fixed_device_bytes);
    out.u64(plan.fixed_host_bytes);
    out.u32(plan.records.size());
    for (const auto& record : plan.records) {
        out.u64(record.bytes);
        out.u64(record.alignment);
    }
    out.u32(plan.working_sets.size());
    for (const auto& set : plan.working_sets) {
        out.string(set.name);
        out.u32(set.records.size());
        for (const auto& record : set.records) {
            out.u32(record.record_class.index);
            out.u32(record.count);
        }
    }
    out.u64(plan.metadata.fixed_bytes);
    out.u64(plan.metadata.bytes_per_block);
    out.u32(plan.metadata.bytes_per_record.size());
    for (const auto bytes : plan.metadata.bytes_per_record) out.u64(bytes);
    out.u32(plan.reference.tokens_per_request);
    out.u32(plan.reference.request_count);
    out.u64(plan.reference.device_bytes);
    out.u64(r.capacity.available_bytes);
    out.u64(r.capacity.available_host_bytes);
    out.u64(r.capacity.host_reservation_bytes);
    out.boolean(r.capacity.shared_physical_memory);
    out.u32(r.capacity.node);
    out.i32(r.capacity.device);
    out.boolean(r.capacity.physical_placement_set);
    out.u64(r.raw_available_bytes);
    out.u64(r.graph_reserve_bytes);
    out.u64(r.maximum_runtime_bytes);
    out.u64(r.weight_bytes);
    out.u64(r.host_weight_bytes);
    out.u64(r.staging_bytes);
    out.u64(r.read_bytes);
    out.u64(r.upload_bytes);
    out.f64(r.load_seconds);
    out.u64(r.device_objects);
    out.u64(r.host_objects);
    out.u64(r.free_after_weights_bytes);
    out.boolean(r.preparation_required);
    out.boolean(r.control_owner);
    out.u32(r.graph_groups.size());
    for (auto id : r.graph_groups) out.u32(id);
    out.string(r.hardware_class);
    out.string(r.cuda_sync_mode);
    out.string(r.failure);
}

RankPreparationReport decode_preparation_report(wire::Decoder& in) {
    RankPreparationReport r;
    auto& plan              = r.capacity.plan;
    plan.fixed_device_bytes = in.u64();
    plan.fixed_host_bytes   = in.u64();
    plan.records.resize(in.u32());
    for (auto& record : plan.records) record = {in.u64(), in.u64()};
    plan.working_sets.resize(in.u32());
    for (auto& set : plan.working_sets) {
        set.name = in.string();
        set.records.resize(in.u32());
        for (auto& record : set.records) record = {{in.u32()}, in.u32()};
    }
    plan.metadata.fixed_bytes     = in.u64();
    plan.metadata.bytes_per_block = in.u64();
    plan.metadata.bytes_per_record.resize(in.u32());
    for (auto& bytes : plan.metadata.bytes_per_record) bytes = in.u64();
    plan.reference                    = {in.u32(), in.u32(), in.u64()};
    r.capacity.available_bytes        = in.u64();
    r.capacity.available_host_bytes   = in.u64();
    r.capacity.host_reservation_bytes = in.u64();
    r.capacity.shared_physical_memory = in.boolean();
    r.capacity.node                   = in.u32();
    r.capacity.device                 = in.i32();
    r.capacity.physical_placement_set = in.boolean();
    r.raw_available_bytes             = in.u64();
    r.graph_reserve_bytes             = in.u64();
    r.maximum_runtime_bytes           = in.u64();
    r.capacity.graph_reserve_bytes    = r.graph_reserve_bytes;
    r.capacity.maximum_runtime_bytes  = r.maximum_runtime_bytes;
    r.weight_bytes                    = in.u64();
    r.host_weight_bytes               = in.u64();
    r.staging_bytes                   = in.u64();
    r.read_bytes                      = in.u64();
    r.upload_bytes                    = in.u64();
    r.load_seconds                    = in.f64();
    r.device_objects                  = in.u64();
    r.host_objects                    = in.u64();
    r.free_after_weights_bytes        = in.u64();
    r.preparation_required            = in.boolean();
    r.control_owner                   = in.boolean();
    r.graph_groups.resize(in.u32());
    for (auto& id : r.graph_groups) id = in.u32();
    r.hardware_class = in.string();
    r.cuda_sync_mode = in.string();
    r.failure        = in.string();
    return r;
}

GroupPreparation
prepare_execution_group(dist::ExecutionGroup& group, std::span<const wire::Bytes> payloads,
                        ContextCapacityPolicy policy, const GroupPreparationOptions& options,
                        std::chrono::milliseconds timeout, ControlPreparation control) {
    const auto& topology = group.topology();
    if (payloads.size() != topology.ranks.size())
        throw std::invalid_argument("preparation payload count differs from rank count");
    try {
        const auto everyone = all_ranks(topology);
        std::vector<dist::OutgoingMessage> requests;
        requests.reserve(topology.ranks.size());
        for (const auto& rank : topology.ranks) {
            wire::Encoder out;
            encode_preparation_request(
                out, {topology.plan, options.memory_headroom_bytes, payloads[rank.rank]});
            requests.push_back(
                {rank.rank, std::make_shared<const wire::Message>(wire::Message{
                                wire::Kind::Prepare, group.epoch(), std::move(out).take()})});
        }
        if (!group.try_send(requests))
            throw std::runtime_error("rank preparation payloads exceed control capacity");

        GroupPreparation state;
        state.reports.resize(topology.ranks.size());
        state.layouts.resize(topology.ranks.size());
        state.memory_samples.resize(topology.ranks.size());
        std::uint32_t current_attempt = 0;

        const auto recv_reports = [&](std::span<const dist::RankId> subset) {
            std::string first_failure;
            for (auto& [peer, bytes] :
                 receive_subset(group, subset, wire::Kind::Prepared, timeout)) {
                wire::Decoder in(bytes);
                if (in.u64() != topology.plan || in.u32() != current_attempt)
                    throw std::runtime_error("rank capacity belongs to another attempt");
                auto rep = decode_preparation_report(in);
                in.finish();
                if (!rep.failure.empty() && first_failure.empty())
                    first_failure = "rank " + std::to_string(peer) + ": " + rep.failure;
                state.reports.at(peer) = std::move(rep);
            }
            return first_failure;
        };

        const auto recv_allocated = [&](std::span<const dist::RankId> subset,
                                        PreparationAction action, AttemptFailure& failure) {
            for (auto& [peer, bytes] :
                 receive_subset(group, subset, wire::Kind::Allocated, timeout)) {
                wire::Decoder in(bytes);
                if (in.u64() != topology.plan || in.u32() != current_attempt)
                    throw std::runtime_error(
                        "rank preparation response belongs to another attempt");
                const bool ready            = in.boolean();
                const auto error            = in.string();
                const auto pool_shortfall   = in.u64();
                const auto host_shortfall   = in.u64();
                const auto retry_allocation = static_cast<AllocationRetry>(in.u8());
                const auto layout           = in.bytes();
                if (action == PreparationAction::CheckMemory ||
                    action == PreparationAction::CheckResidentMemory)
                    state.memory_samples.at(peer) = {in.u64(), in.u64(),     in.u64(),
                                                     in.u64(), in.boolean(), in.u64()};
                in.finish();
                if (ready) {
                    if (action == PreparationAction::CheckMemory)
                        state.layouts.at(peer) = wire::Bytes(layout.begin(), layout.end());
                    continue;
                }
                if (failure.message.empty())
                    failure.message = "rank " + std::to_string(peer) + ": " + error;
                if (pool_shortfall) {
                    auto& deficit =
                        failure.pool_shortfalls[pool_key_for(topology, state.reports, peer)];
                    deficit = std::max(deficit, pool_shortfall);
                }
                if (host_shortfall) {
                    if (topology.rank(peer).node == options.coordinator_node)
                        failure.directory_shortfall =
                            std::max(failure.directory_shortfall, host_shortfall);
                    else
                        failure.terminal = true;
                }
                if (retry_allocation == AllocationRetry::AllocatorFailure ||
                    (retry_allocation == AllocationRetry::ReduceCapacity &&
                     policy.mode == ContextCapacityMode::Automatic)) {
                    failure.local_allocation_failures.push_back(peer);
                } else if (!pool_shortfall && !host_shortfall) {
                    failure.terminal = true;
                }
            }
            return failure.message.empty();
        };

        const auto recv_probe = [&](std::span<const dist::RankId> subset) {
            std::string first_failure;
            for (auto& [peer, bytes] :
                 receive_subset(group, subset, wire::Kind::Allocated, timeout)) {
                wire::Decoder in(bytes);
                if (in.u64() != topology.plan || in.u32() != current_attempt)
                    throw std::runtime_error("rank probe response belongs to another attempt");
                const bool ready = in.boolean();
                const auto error = in.string();
                in.finish();
                if (!ready && first_failure.empty())
                    first_failure = "rank " + std::to_string(peer) + ": " + error;
            }
            return first_failure;
        };

        const auto coordinator_origin = device_memory_snapshot(Device::cpu());
        std::optional<StartupMemoryBudget> coordinator_budget;
        if (coordinator_origin.shared_physical_memory)
            coordinator_budget.emplace(coordinator_origin,
                                       device_budget(coordinator_origin.available_bytes()),
                                       options.memory_headroom_bytes, 0);
        const auto sample_coordinator_offer = [&]() {
            const auto snap = device_memory_snapshot(Device::cpu());
            return CoordinatorMemoryOffer{
                .node                 = options.coordinator_node,
                .available_host_bytes = snap.system_available_bytes,
                .available_unified_bytes =
                    coordinator_budget ? coordinator_budget->planning_available(snap) : SIZE_MAX,
                .shared_physical_memory = snap.shared_physical_memory,
                .frontend_host_bytes    = options.frontend_host_bytes,
            };
        };
        const auto sample_baseline = [&]() {
            enter_subset(group, everyone, state, current_attempt,
                         PreparationAction::SampleBaseline);
            (void)recv_reports(everyone);
        };

        (void)recv_reports(everyone);
        const auto connect = [&] {
            enter_subset(group, everyone, state, current_attempt,
                         PreparationAction::PrepareCommunication);
            dist::prepare_communication(group, timeout);
            (void)recv_reports(everyone);
        };
        if (options.placement_names.empty()) connect();

        const auto loading_reservations =
            cluster_host_reservations(capacity_offers(state.reports), sample_coordinator_offer());
        for (const auto& wave :
             build_materialization_waves(topology, state.reports, loading_reservations)) {
            enter_subset(group, wave, state, current_attempt, PreparationAction::MaterializeWeights,
                         std::nullopt, &loading_reservations.by_node);
            if (const auto error = recv_reports(wave); !error.empty())
                throw DeviceAllocationError(error.c_str());
        }
        sample_baseline();

        const auto release_all = [&](PreparationAction action = PreparationAction::Release) {
            if (control.release) control.release();
            for (auto& layout : state.layouts) layout = {};
            enter_subset(group, everyone, state, current_attempt, action);
            ++current_attempt;
            for (auto& [peer, bytes] :
                 receive_subset(group, everyone, wire::Kind::Prepared, timeout)) {
                wire::Decoder ack(bytes);
                if (ack.u64() != topology.plan || ack.u32() != current_attempt)
                    throw std::runtime_error(
                        "rank release acknowledgement belongs to another attempt");
                ack.finish();
            }
            sample_baseline();
        };

        const auto configure_targets = [&](bool narrow) {
            enter_subset(group, everyone, state, current_attempt,
                         PreparationAction::ConfigureTargets, narrow ? 1ULL : 0ULL);
            (void)recv_reports(everyone);
        };
        const auto calibrate     = [&]() -> std::string {
            if (std::none_of(state.reports.begin(), state.reports.end(),
                                 [](const auto& r) { return r.preparation_required; }))
                return {};
            for (const auto& wave : build_graph_waves(topology, state.reports)) {
                for (auto [probe, capture] :
                     {std::pair{PreparationAction::PrepareProbe, PreparationAction::Calibrate},
                      std::pair{PreparationAction::PrepareRebuildProbe,
                                PreparationAction::CalibrateRebuild}}) {
                    enter_subset(group, wave, state, current_attempt, probe);
                    if (auto error = recv_probe(wave); !error.empty()) return error;
                    enter_subset(group, wave, state, current_attempt, capture);
                    if (auto error = recv_reports(wave); !error.empty()) return error;
                }
            }
            sample_baseline();
            return {};
        };

        const PhysicalPoolKey coordinator_uma_pool{options.coordinator_node, -1};
        const bool coordinator_has_uma_workers =
            std::any_of(everyone.begin(), everyone.end(), [&](auto rank) {
                return pool_key_for(topology, state.reports, rank) == coordinator_uma_pool;
            });
        const auto check_memory = [&](PreparationAction action,
                                      const std::map<std::uint32_t, std::size_t>& reservations,
                                      AttemptFailure& failure) {
            enter_subset(group, everyone, state, current_attempt, action, std::nullopt,
                         &reservations);
            (void)recv_allocated(everyone, action, failure);

            // The coordinator may own a directory on a node with no workers at all.
            const auto required = reservations.at(options.coordinator_node);
            const auto snap     = device_memory_snapshot(Device::cpu());
            auto deficit        = host_memory_shortfall(snap, required);
            if (coordinator_budget)
                deficit = std::max(deficit, coordinator_budget->device_shortfall(snap, required));
            if (deficit) {
                if (failure.message.empty())
                    failure.message =
                        "Coordinator node " + std::to_string(options.coordinator_node) +
                        " memory budget exceeded by " + std::to_string(deficit) + " bytes";
                if (coordinator_has_uma_workers) {
                    auto& shortfall = failure.pool_shortfalls[coordinator_uma_pool];
                    shortfall       = std::max(shortfall, deficit);
                } else {
                    failure.directory_shortfall = std::max(failure.directory_shortfall, deficit);
                }
            }
            return failure.message.empty();
        };

        const auto correct_capacity = [&](const AttemptFailure& failure,
                                          std::vector<std::size_t>& runtime_limits,
                                          std::size_t& directory_limit) {
            if (failure.terminal) return false;
            if (policy.mode == ContextCapacityMode::Explicit)
                return !failure.local_allocation_failures.empty() &&
                       failure.pool_shortfalls.empty() && !failure.directory_shortfall;

            bool retry = !failure.local_allocation_failures.empty();
            for (const auto rank : failure.local_allocation_failures) {
                // No measured deficit is available for allocator failures. One block is enough
                // to change the request, and a minimum-sized allocation can retry after release.
                const auto& cap      = state.capacities.at(rank);
                runtime_limits[rank] = reduced_runtime_limit(cap, cap.block_bytes);
            }
            for (const auto& [pool, shortfall] : failure.pool_shortfalls) {
                auto remaining = shortfall;
                for (const auto rank : everyone) {
                    if (pool_key_for(topology, state.reports, rank) != pool) continue;
                    const auto& cap  = state.capacities.at(rank);
                    const auto floor = cap.minimum_runtime_reservation_bytes;
                    const auto previous =
                        std::min(runtime_limits[rank], cap.runtime_reservation_bytes);
                    remaining -= std::min(remaining, cap.runtime_reservation_bytes - previous);
                    const auto cut = std::min(remaining, previous - floor);
                    runtime_limits[rank] =
                        reduced_runtime_limit(cap, cap.runtime_reservation_bytes - previous + cut);
                    remaining -= cut;
                }
                // Clamp peers at their previous sizes so the next solve cannot refill the cut.
                retry |= remaining < shortfall;
            }
            if (failure.directory_shortfall) {
                const auto bytes = state.host_reservations.coordinator_directory_bytes;
                directory_limit  = bytes - std::min(bytes, failure.directory_shortfall);
                retry |= directory_limit < bytes;
            }
            return retry;
        };

        const auto block_score = [](std::span<const ContextCapacityResolution> capacities) {
            std::size_t blocks = SIZE_MAX;
            for (const auto& rank : capacities)
                blocks = std::min(blocks, rank.device_bytes / rank.block_bytes);
            return blocks;
        };
        const bool rank_by_capacity =
            options.maximize_context && options.placement_names.size() > 1;
        std::size_t accepted_blocks = 0;
        std::optional<std::vector<ContextCapacityResolution>> narrow_upper;
        std::vector<ContextCapacityPlan> baseline_plans;
        if (options.try_narrow_targets && policy.mode == ContextCapacityMode::Automatic) {
            configure_targets(false);
            baseline_plans.reserve(state.reports.size());
            for (const auto& report : state.reports) baseline_plans.push_back(report.capacity.plan);
        }

        const auto run_target_mode = [&](bool narrow) -> std::string {
            configure_targets(narrow);
            std::vector<std::size_t> runtime_limits(everyone.size(), SIZE_MAX);
            std::size_t directory_limit = SIZE_MAX;
            bool calibrated             = false;
            std::string first_failure;

            for (unsigned trial = 0; trial < 2; ++trial) {
                if (!calibrated) {
                    if (auto error = calibrate(); !error.empty()) {
                        if (first_failure.empty()) first_failure = std::move(error);
                        release_all();
                        continue;
                    }
                    calibrated = true;
                }

                auto offers      = capacity_offers(state.reports);
                auto coordinator = sample_coordinator_offer();
                for (const auto rank : everyone)
                    offers[rank].maximum_runtime_bytes =
                        std::min(offers[rank].maximum_runtime_bytes, runtime_limits[rank]);
                coordinator.maximum_directory_bytes = directory_limit;
                try {
                    state.capacities = resolve_cluster_context_capacity(policy, offers, coordinator,
                                                                        options.purpose);
                } catch (const ContextBudgetError& error) {
                    return first_failure.empty() ? error.what() : first_failure;
                }
                state.host_reservations =
                    cluster_host_reservations(offers, coordinator, state.capacities);

                AttemptFailure failure;
                for (const auto& wave : build_graph_waves(topology, state.reports)) {
                    enter_subset(group, wave, state, current_attempt, PreparationAction::Allocate,
                                 std::nullopt, &state.host_reservations.by_node);
                    if (!recv_allocated(wave, PreparationAction::Allocate, failure)) break;
                    enter_subset(group, wave, state, current_attempt,
                                 PreparationAction::PrepareExecution);
                    if (!recv_allocated(wave, PreparationAction::PrepareExecution, failure)) break;
                }
                if (failure.message.empty() &&
                    check_memory(PreparationAction::CheckMemory, state.host_reservations.by_node,
                                 failure)) {
                    if (control.prepare) {
                        // Check the full ledger before directory allocation, then verify the
                        // outstanding payload reservations against the actual resident footprint.
                        control.prepare(state.layouts);
                        auto outstanding = state.host_reservations.by_node;
                        outstanding.at(options.coordinator_node) -=
                            state.host_reservations.coordinator_directory_bytes;
                        (void)check_memory(PreparationAction::CheckResidentMemory, outstanding,
                                           failure);
                    }
                    if (failure.message.empty()) {
                        state.attempt = current_attempt;
                        if (rank_by_capacity)
                            accepted_blocks = block_score(resolve_cluster_context_capacity(
                                ContextCapacityPolicy::automatic(), offers, coordinator));
                        if (narrow && trial == 0 && !baseline_plans.empty()) {
                            // These offers were captured before Program, KV and directories
                            // existed.
                            for (const auto rank : everyone)
                                offers[rank].plan = baseline_plans[rank];
                            try {
                                narrow_upper = cluster_baseline_capacity_upper_bound(
                                    std::move(offers), coordinator, options.purpose);
                            } catch (const std::exception&) {
                                // Measure the full-width configuration when the bound is
                                // unavailable.
                            }
                        }
                        return {};
                    }
                }

                if (first_failure.empty()) first_failure = failure.message;
                const bool retry =
                    trial == 0 && correct_capacity(failure, runtime_limits, directory_limit);
                release_all();
                if (!retry) return first_failure;
            }
            return first_failure;
        };

        if (!options.placement_names.empty()) {
            std::vector<PlacementEstimate> candidates;
            std::vector<std::vector<std::vector<dist::RankId>>> waves(
                options.placement_names.size());
            std::string failures;
            const auto rejected = [&](std::size_t id, std::string_view reason) {
                auto message = options.placement_names.at(id) + ": " + std::string(reason);
                std::clog << "[placement] " << message << '\n';
                if (!failures.empty()) failures += "; ";
                failures += message;
            };
            const auto origin      = capacity_offers(state.reports);
            const auto coordinator = sample_coordinator_offer();
            for (std::size_t id = 0; id < options.placement_names.size(); ++id) {
                enter_subset(group, everyone, state, current_attempt,
                             PreparationAction::OfferPlacement, id);
                if (auto error = recv_reports(everyone); !error.empty()) {
                    rejected(id, error);
                    continue;
                }
                auto offers = origin;
                for (const auto rank : everyone)
                    offers[rank].plan = state.reports[rank].capacity.plan;
                try {
                    (void)resolve_cluster_context_capacity(policy, offers, coordinator,
                                                           options.purpose);
                    const auto upper =
                        rank_by_capacity
                            ? block_score(resolve_cluster_context_capacity(
                                  ContextCapacityPolicy::automatic(), offers, coordinator))
                            : 0;
                    const auto reservations = cluster_host_reservations(offers, coordinator);
                    waves[id] = build_materialization_waves(topology, state.reports, reservations);
                    candidates.push_back({id, 0, id, upper});
                } catch (const ContextBudgetError& error) { rejected(id, error.what()); }
            }
            const auto install = [&](std::size_t id) -> std::optional<std::size_t> {
                for (const auto& wave : waves[id]) {
                    enter_subset(group, wave, state, current_attempt,
                                 PreparationAction::MaterializePlacement, id);
                    if (auto error = recv_reports(wave); !error.empty()) {
                        rejected(id, error);
                        return {};
                    }
                }
                connect();
                sample_baseline();
                if (auto error = run_target_mode(false); !error.empty()) {
                    rejected(id, error);
                    return {};
                }
                return accepted_blocks;
            };
            const auto selected = select_prepared_placement(
                candidates, install, [&] { release_all(PreparationAction::ReleasePlacement); });
            if (selected) {
                std::clog << "[placement] selected " << options.placement_names[*selected];
                for (const auto& candidate : candidates) {
                    if (candidate.id == *selected) continue;
                    std::clog << "; alternative " << options.placement_names[candidate.id];
                    if (rank_by_capacity)
                        std::clog << " upper_bound_blocks=" << candidate.blocks;
                    else
                        std::clog << " preference=" << candidate.preference;
                }
                std::clog << '\n';
                return state;
            }
            throw ContextBudgetError("no feasible model placement: " + failures);
        }

        if (options.try_narrow_targets && run_target_mode(true).empty()) {
            if (policy.mode == ContextCapacityMode::Explicit ||
                (narrow_upper && preserves_cluster_capacity(state.capacities, *narrow_upper)))
                return state;
            const auto narrow = state.capacities;
            release_all();
            const auto baseline_error = run_target_mode(false);
            if (baseline_error.empty()) {
                if (!preserves_cluster_capacity(narrow, state.capacities)) return state;
                const auto baseline = state.capacities;
                release_all();
                if (run_target_mode(true).empty()) {
                    if (preserves_cluster_capacity(state.capacities, baseline)) return state;
                    release_all();
                }
            } else {
                // The measured baseline failed. Restore the working narrow configuration.
                if (run_target_mode(true).empty()) return state;
                throw ContextBudgetError(baseline_error);
            }
        }

        if (const auto error = run_target_mode(false); !error.empty())
            throw ContextBudgetError(error);
        return state;
    } catch (const std::exception& error) {
        if (control.release) control.release();
        group.fail(error.what());
        throw;
    }
}

void activate_execution_group(dist::ExecutionGroup& group, const GroupPreparation& prepared,
                              std::chrono::milliseconds timeout) {
    try {
        const auto everyone = all_ranks(group.topology());
        enter_subset(group, everyone, prepared, prepared.attempt, PreparationAction::Activate);
        for (const auto& [peer, bytes] :
             receive_subset(group, everyone, wire::Kind::Ready, timeout)) {
            (void)peer;
            wire::Decoder in(bytes);
            if (in.u64() != group.topology().plan || in.u32() != prepared.attempt)
                throw std::runtime_error("rank readiness belongs to another preparation attempt");
            in.finish();
        }
    } catch (const std::exception& error) {
        group.fail(error.what());
        throw;
    }
}

} // namespace ninfer::runtime
