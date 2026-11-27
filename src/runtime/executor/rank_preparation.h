#pragma once

#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "distributed/communication/preparation.h"
#include "runtime/executor/preparation.h"
#include "runtime/engine/startup_memory.h"
#include "runtime/engine/context_cache/context_cost.h"
#include "runtime/observability/snapshot.h"

#include <algorithm>
#include <limits>
#include <thread>

namespace ninfer::runtime {

inline distributed::wire::Message
receive_preparation_command(distributed::WorkerConnection& connection) {
    for (;;) {
        if (auto message = connection.poll()) return std::move(*message);
        connection.communication().check();
        connection.wait(std::chrono::milliseconds(5));
    }
}

// The caller retains the native preparation owner across failures so communication can be
// aborted before device resources are destroyed. Activation returns the accepted attempt;
// the caller constructs its execution loop and sends Ready only when that loop is usable.
// Local allocation failures support coordinated Release without reloading immutable weights.
// Collective failures propagate to the connection owner before releasing GPU resources.
template <class Preparation>
[[nodiscard]] std::optional<std::uint32_t>
prepare_rank(distributed::WorkerConnection& connection, std::unique_ptr<artifact::Reader> reader,
             std::span<const std::byte> payload, std::size_t memory_headroom_bytes,
             std::unique_ptr<Preparation>& preparing) {
    namespace wire                  = distributed::wire;
    auto& device                    = connection.device();
    const auto& topology            = connection.topology();
    const auto& identity            = connection.bootstrap().identity;
    const bool has_collective_peers = topology.replica_for_rank(identity.rank).ranks.size() > 1;
    const auto initial              = connection.memory_origin();
    const auto observer             = connection.startup_observer();
    {
        StartupPhaseScope phase(observer, StartupPhase::TargetPlan);
        preparing = std::make_unique<Preparation>(*reader, payload, identity, device, topology,
                                                  connection.communication(), observer);
        phase.complete();
    }
    const auto requirement      = preparing->memory();
    const auto staging_headroom = artifact::cuda_weight_staging_headroom(*reader);
    StartupMemoryBudget budget(initial, device_budget(initial.available_bytes()),
                               memory_headroom_bytes, 0);

    bool materialized = false;
    std::optional<DeviceMemorySnapshot> after_weights;
    std::int64_t retained_bytes = 0, base_retained_bytes = 0;
    std::uint32_t attempt       = 0;
    ContextCapacityPlan capacity_plan;
    std::size_t graph_reserve_bytes = 0;
    bool model_charges_control      = false;

    const auto report = [&](DeviceMemorySnapshot snapshot, std::string phase_failure = {},
                            const PlacementOffer* offered = nullptr) {
        const auto available = budget.planning_available(snapshot);
        wire::Encoder out;
        out.u64(topology.plan);
        out.u32(attempt);
        RankPreparationReport value;
        if (materialized) {
            value                     = preparing->report(SIZE_MAX);
            graph_reserve_bytes       = SIZE_MAX - value.capacity.available_bytes;
            value.graph_reserve_bytes = graph_reserve_bytes;
            value.capacity.available_bytes =
                available > graph_reserve_bytes ? available - graph_reserve_bytes : 0;
            capacity_plan         = value.capacity.plan;
            model_charges_control = value.control_owner;
        } else {
            value.capacity.available_bytes = available;
            value.weight_bytes             = requirement.weight_bytes;
            value.host_weight_bytes        = requirement.host_weight_bytes;
            value.staging_bytes            = staging_headroom;
        }
        if (offered) {
            value.capacity.plan     = offered->capacity;
            value.weight_bytes      = offered->device_bytes;
            value.host_weight_bytes = offered->host_bytes;
            value.staging_bytes     = offered->device_bytes ? staging_headroom : 0;
        }
        value.raw_available_bytes             = available;
        value.capacity.available_host_bytes   = snapshot.system_available_bytes;
        value.capacity.host_reservation_bytes = requirement.host_reservation_bytes;
        value.capacity.shared_physical_memory = snapshot.shared_physical_memory;
        value.capacity.node                   = identity.node;
        value.capacity.device                 = identity.device;
        value.capacity.graph_reserve_bytes    = value.graph_reserve_bytes;
        value.capacity.physical_placement_set = true;
        value.hardware_class                  = context_cost_hardware_class(
            device.properties().name, device.properties().major, device.properties().minor);
        value.cuda_sync_mode           = device.sync_mode();
        value.free_after_weights_bytes = after_weights ? after_weights->available_bytes() : 0;
        value.failure                  = std::move(phase_failure);
        encode_preparation_report(out, value);
        connection.send(wire::Kind::Prepared, std::move(out).take());
    };

    // Initial pre-materialization report so coordinator sees weight/device facts before Phase 1.
    report(device_memory_snapshot(device.device()));

    bool entered = false, accepted = false;
    ContextCapacity allocated_capacity;
    std::size_t control_reservation_bytes = 0, planned_slack_bytes = 0;
    std::string failure;
    std::size_t pool_shortfall       = 0;
    std::size_t host_shortfall       = 0;
    AllocationRetry retry_allocation = AllocationRetry::None;

    const auto describe = [&](DeviceMemorySnapshot after) {
        auto result                                 = preparing->describe();
        auto& observation                           = result.observation;
        observation.rank                            = identity.rank;
        observation.node                            = identity.node;
        observation.hardware                        = device.properties().name;
        observation.communication                   = connection.communication().statistics();
        observation.stamp                           = {0, observation_time_ns()};
        observation.physical_memory                 = PhysicalMemorySample{after.total_bytes,
                                                           after.cuda_free_bytes,
                                                           after.system_total_bytes,
                                                           after.system_available_bytes,
                                                           after.shared_physical_memory,
                                                           observation.stamp.sampled_ns};
        result.budget.limit_bytes                   = budget.limit();
        result.budget.context_control_budget_bytes  = control_reservation_bytes;
        result.budget.headroom_bytes                = budget.headroom();
        result.budget.host_reservation_bytes        = requirement.host_reservation_bytes;
        result.budget.available_after_weights_bytes = after_weights->available_bytes();
        result.budget.available_after_startup_bytes = after.available_bytes();
        result.budget.planned_slack_bytes           = planned_slack_bytes;
        result.budget.retained_bytes                = retained_bytes;
        wire::Encoder out;
        encode(out, result);
        return std::move(out).take();
    };

    const auto allocated = [&](wire::Bytes description,
                               const PhysicalMemorySample* sample = nullptr) {
        wire::Encoder out;
        out.u64(topology.plan);
        out.u32(attempt);
        out.boolean(accepted);
        out.string(failure);
        out.u64(pool_shortfall);
        out.u64(host_shortfall);
        out.u8(static_cast<std::uint8_t>(retry_allocation));
        out.bytes(description);
        if (sample) {
            out.u64(sample->cuda_total_bytes);
            out.u64(sample->cuda_free_bytes);
            out.u64(sample->system_total_bytes);
            out.u64(sample->system_available_bytes);
            out.boolean(sample->shared_physical_memory);
            out.u64(sample->sampled_ns);
        }
        connection.send(wire::Kind::Allocated, std::move(out).take());
    };

    const auto can_reduce_rank = [&] {
        return allocated_capacity.device_bytes > capacity_plan.minimum_capacity().device_bytes;
    };
    const auto allocation_failed = [&](const std::exception& error, AllocationRetry retry) {
        accepted         = false;
        failure          = error.what();
        retry_allocation = retry;
    };

    for (;;) {
        auto message = receive_preparation_command(connection);
        if (message.kind == wire::Kind::Shutdown) { return std::nullopt; }
        if (message.kind != wire::Kind::Enter)
            throw std::logic_error("unexpected rank preparation command");
        wire::Decoder command(message.payload);
        if (command.u64() != topology.plan || command.u32() != attempt)
            throw std::logic_error("rank preparation attempt differs");
        const auto action = static_cast<PreparationAction>(command.u8());
        const ContextCapacity capacity{command.u64()};
        const auto planned_control_bytes = command.u64();
        const auto planning_slack        = command.u64();
        const auto node_host_reservation = command.u64();
        command.finish();

        if (action == PreparationAction::PrepareCommunication) {
            if (!distributed::prepare_communication(connection,
                                                    preparing->communication_requirements(),
                                                    connection.bootstrap().communication_policy))
                return std::nullopt;
            device.synchronize();
            report(device_memory_snapshot(device.device()));
            continue;
        }
        if (action == PreparationAction::MaterializeWeights) {
            const auto before_weights = device_memory_snapshot(device.device());
            {
                StartupPhaseScope phase(observer, StartupPhase::RuntimeMemoryCheck);
                auto available_weights = budget.planning_available(before_weights);
                if (before_weights.shared_physical_memory)
                    available_weights -= std::min(available_weights, node_host_reservation);
                if (requirement.weight_bytes > available_weights ||
                    (before_weights.shared_physical_memory &&
                     requirement.host_weight_bytes > available_weights - requirement.weight_bytes))
                    throw DeviceAllocationError("rank immutable weights exceed device budget",
                                                requirement.weight_bytes);
                const auto host_available = before_weights.system_available_bytes;
                if (node_host_reservation > host_available ||
                    requirement.host_weight_bytes > host_available - node_host_reservation)
                    throw std::invalid_argument("rank immutable Host weights exceed available RAM");
                phase.complete();
            }
            preparing->materialize();
            if constexpr (!requires { preparing->offer_placement(0); }) reader.reset();
            device.synchronize();
            const auto after_materialize = device_memory_snapshot(device.device());
            materialized                 = true;
            const auto loaded            = preparing->report(SIZE_MAX);
            retained_bytes               = retained_startup_bytes(
                before_weights, after_materialize,
                loaded.weight_bytes +
                    (before_weights.shared_physical_memory ? loaded.host_weight_bytes : 0));
            base_retained_bytes = retained_bytes;
            report(after_materialize);
            continue;
        }
        if constexpr (requires { preparing->offer_placement(0); }) {
            if (action == PreparationAction::OfferPlacement) {
                const auto before = device_memory_snapshot(device.device());
                try {
                    auto offer = preparing->offer_placement(capacity.device_bytes);
                    report(before, {}, &offer);
                } catch (const artifact::ArtifactIoError& error) {
                    report(before, error.what());
                } catch (const std::system_error& error) { report(before, error.what()); }
                continue;
            }
            if (action == PreparationAction::MaterializePlacement) {
                const auto before = device_memory_snapshot(device.device());
                const auto base   = preparing->report(SIZE_MAX);
                try {
                    preparing->materialize_placement(capacity.device_bytes);
                    device.synchronize();
                    const auto after  = device_memory_snapshot(device.device());
                    const auto loaded = preparing->report(SIZE_MAX);
                    retained_bytes =
                        base_retained_bytes +
                        retained_startup_bytes(
                            before, after,
                            loaded.weight_bytes - base.weight_bytes +
                                (before.shared_physical_memory
                                     ? loaded.host_weight_bytes - base.host_weight_bytes
                                     : 0));
                    after_weights.reset();
                    report(after);
                } catch (const DeviceAllocationError& error) {
                    report(device_memory_snapshot(device.device()), error.what());
                } catch (const artifact::ArtifactIoError& error) {
                    report(device_memory_snapshot(device.device()), error.what());
                } catch (const std::bad_alloc& error) {
                    report(device_memory_snapshot(device.device()), error.what());
                } catch (const std::system_error& error) {
                    report(device_memory_snapshot(device.device()), error.what());
                }
                continue;
            }
        }
        if (action == PreparationAction::SampleBaseline) {
            device.synchronize();
            const auto snapshot = device_memory_snapshot(device.device());
            if (!after_weights) after_weights = snapshot;
            report(snapshot);
            continue;
        }
        if (action == PreparationAction::ConfigureTargets) {
            if constexpr (requires { preparing->select_narrow_targets(true); }) {
                preparing->select_narrow_targets(capacity.device_bytes != 0);
            }
            report(device_memory_snapshot(device.device()));
            continue;
        }
        if (action == PreparationAction::PrepareProbe ||
            action == PreparationAction::PrepareRebuildProbe) {
            StartupPhaseScope phase(observer, StartupPhase::ProgramInitialize);
            bool success = true;
            failure.clear();
            try {
                if constexpr (requires {
                                  preparing->prepare_probe();
                                  preparing->prepare_rebuild_probe();
                              }) {
                    if (action == PreparationAction::PrepareProbe)
                        preparing->prepare_probe();
                    else
                        preparing->prepare_rebuild_probe();
                } else {
                    throw std::logic_error("rank preparation has no calibration probe");
                }
            } catch (const DeviceAllocationError& error) {
                success = false;
                failure = error.what();
            }
            if (success)
                phase.complete();
            else
                phase.fail();
            wire::Encoder out;
            out.u64(topology.plan);
            out.u32(attempt);
            out.boolean(success);
            out.string(failure);
            connection.send(wire::Kind::Allocated, std::move(out).take());
            continue;
        }
        if (action == PreparationAction::Calibrate ||
            action == PreparationAction::CalibrateRebuild) {
            StartupPhaseScope phase(observer, StartupPhase::CudaGraphCalibration);
            std::string cal_failure;
            try {
                if constexpr (requires {
                                  preparing->calibrate();
                                  preparing->calibrate_rebuild();
                              }) {
                    if (action == PreparationAction::Calibrate)
                        preparing->calibrate();
                    else
                        preparing->calibrate_rebuild();
                } else {
                    throw std::logic_error("rank preparation has no calibration phase");
                }
                phase.complete();
            } catch (const DeviceAllocationError& error) {
                if (has_collective_peers) throw;
                cal_failure = error.what();
                phase.fail();
            }
            report(device_memory_snapshot(device.device()), std::move(cal_failure));
            continue;
        }
        if (action == PreparationAction::Release || action == PreparationAction::ReleasePlacement) {
            connection.startup_attempt(++attempt);
            StartupPhaseScope phase(observer, StartupPhase::RuntimeRelease);
            if constexpr (requires { preparing->release_placement(); }) {
                if (action == PreparationAction::ReleasePlacement) {
                    preparing->release_placement();
                    device.synchronize();
                    connection.communication().release_preparation(
                        std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(connection.bootstrap().connect_timeout_ms));
                    after_weights.reset();
                    retained_bytes = base_retained_bytes;
                } else
                    preparing->release_resources();
            } else
                preparing->release_resources();
            device.synchronize();
            failure.clear();
            entered = accepted = false;
            phase.complete();
            // Acknowledge release separately from sampling the settled physical pool.
            wire::Encoder out;
            out.u64(topology.plan);
            out.u32(attempt);
            connection.send(wire::Kind::Prepared, std::move(out).take());
            continue;
        }
        if (action == PreparationAction::Activate) {
            if (!accepted || capacity != allocated_capacity)
                throw std::logic_error("rank activation has no accepted allocation");
            return attempt;
        }
        failure.clear();
        pool_shortfall   = 0;
        host_shortfall   = 0;
        retry_allocation = AllocationRetry::None;
        if (action == PreparationAction::CheckMemory ||
            action == PreparationAction::CheckResidentMemory) {
            if (!entered || capacity != allocated_capacity)
                throw std::logic_error("rank memory check has no allocation");
            StartupPhaseScope phase(observer, StartupPhase::RuntimeMemoryCheck);
            wire::Bytes description;
            PhysicalMemorySample sample;
            try {
                if (action == PreparationAction::CheckMemory)
                    connection.communication().sample_startup_memory();
                const auto snapshot     = device_memory_snapshot(device.device());
                const auto host_deficit = host_memory_shortfall(snapshot, node_host_reservation);
                pool_shortfall          = budget.device_shortfall(snapshot, node_host_reservation);
                if (snapshot.shared_physical_memory)
                    pool_shortfall = std::max(pool_shortfall, host_deficit);
                accepted = pool_shortfall == 0 && host_deficit == 0;
                if (accepted) {
                    sample = {snapshot.total_bytes,
                              snapshot.cuda_free_bytes,
                              snapshot.system_total_bytes,
                              snapshot.system_available_bytes,
                              snapshot.shared_physical_memory,
                              observation_time_ns()};
                    if (action == PreparationAction::CheckMemory) description = describe(snapshot);
                } else if (!snapshot.shared_physical_memory && host_deficit) {
                    host_shortfall = host_deficit;
                    failure        = "startup Host reservations exceed available RAM by " +
                              std::to_string(host_deficit) + " bytes";
                } else {
                    failure = "startup device budget exceeded by " +
                              std::to_string(pool_shortfall) + " bytes";
                }
            } catch (const DeviceAllocationError& error) {
                allocation_failed(error, AllocationRetry::AllocatorFailure);
            } catch (const ContextBudgetError& error) {
                allocation_failed(error, can_reduce_rank() ? AllocationRetry::ReduceCapacity
                                                           : AllocationRetry::None);
            }
            if (accepted)
                phase.complete();
            else
                phase.fail();
            allocated(std::move(description), &sample);
            continue;
        }
        if (action == PreparationAction::PrepareExecution) {
            if (!accepted || capacity != allocated_capacity)
                throw std::logic_error("rank execution preparation has no accepted allocation");
            StartupPhaseScope phase(observer, StartupPhase::CudaGraphPrepare);
            try {
                if constexpr (requires { preparing->prepare_execution(); })
                    preparing->prepare_execution();
            } catch (const DeviceAllocationError& error) {
                if (has_collective_peers) throw;
                allocation_failed(error, AllocationRetry::AllocatorFailure);
            }
            if (accepted)
                phase.complete();
            else
                phase.fail();
            allocated({});
            continue;
        }
        if (action != PreparationAction::Allocate || entered)
            throw std::logic_error("rank allocation is already entered");
        StartupPhaseScope phase(observer, StartupPhase::ProgramInitialize);
        const auto before         = device_memory_snapshot(device.device());
        const auto available      = budget.allocation_available(before);
        control_reservation_bytes = planned_control_bytes;
        planned_slack_bytes       = planning_slack;
        auto remaining            = available;
        auto host_remaining       = before.system_available_bytes;
        if (before.shared_physical_memory) {
            remaining -= std::min(remaining, node_host_reservation);
        } else {
            host_remaining -= std::min(host_remaining, node_host_reservation);
        }

        entered            = true;
        allocated_capacity = capacity;
        try {
            // Model planners may include a local directory share in their explicit check.
            // The cluster already charged the entire directory to the coordinator node.
            const auto local_control =
                model_charges_control ? capacity_plan.control_bytes(capacity) : 0;
            const auto runtime_bytes = capacity_plan.reservation_bytes(capacity);
            pool_shortfall           = budget.device_shortfall(before, node_host_reservation,
                                                               graph_reserve_bytes + runtime_bytes);
            if (pool_shortfall) {
                failure = "rank allocation exceeds available device memory by " +
                          std::to_string(pool_shortfall) + " bytes";
                accepted = false;
            } else if (!before.shared_physical_memory &&
                       (host_memory_shortfall(before, node_host_reservation) ||
                        capacity_plan.fixed_host_bytes > host_remaining)) {
                host_shortfall           = host_memory_shortfall(before, node_host_reservation);
                const auto fixed_deficit = capacity_plan.fixed_host_bytes -
                                           std::min(capacity_plan.fixed_host_bytes, host_remaining);
                host_shortfall += std::min(fixed_deficit, SIZE_MAX - host_shortfall);
                failure = "rank allocation exceeds available Host RAM by " +
                          std::to_string(host_shortfall) + " bytes";
                accepted = false;
            } else {
                if (before.shared_physical_memory)
                    remaining += local_control;
                else
                    host_remaining += local_control;
                preparing->prepare(capacity, remaining, host_remaining);
                accepted = true;
            }
        } catch (const DeviceAllocationError& error) {
            allocation_failed(error, AllocationRetry::AllocatorFailure);
        } catch (const ContextBudgetError& error) {
            allocation_failed(error, can_reduce_rank() ? AllocationRetry::ReduceCapacity
                                                       : AllocationRetry::None);
        }
        if (accepted)
            phase.complete();
        else
            phase.fail();
        allocated({});
    }
}

} // namespace ninfer::runtime
