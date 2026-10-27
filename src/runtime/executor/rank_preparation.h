#pragma once

#include "artifact/reader.h"
#include "runtime/executor/preparation.h"
#include "runtime/engine/startup_memory.h"
#include "runtime/engine/context_cache/context_cost.h"
#include "runtime/observability/snapshot.h"

#include <algorithm>
#include <thread>

namespace ninfer::runtime {

inline distributed::wire::Message
receive_preparation_command(distributed::WorkerConnection& connection) {
    for (;;) {
        connection.communication().check();
        if (auto message = connection.poll()) return std::move(*message);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

// The caller retains the native preparation owner across failures so communication can be
// aborted before device resources are destroyed. Activation returns the accepted attempt;
// the caller constructs its execution loop and sends Ready only when that loop is usable.
// Eager preparation ends at allocation; captured programs additionally implement the probe,
// calibration and prepare_execution hooks.
template <class Preparation>
[[nodiscard]] std::optional<std::uint32_t>
prepare_rank(distributed::WorkerConnection& connection, std::unique_ptr<artifact::Reader> reader,
             std::span<const std::byte> payload, std::unique_ptr<Preparation>& preparing) {
    namespace wire       = distributed::wire;
    auto& device         = connection.device();
    const auto& topology = connection.topology();
    const auto initial   = connection.memory_origin();
    preparing = std::make_unique<Preparation>(*reader, payload, connection.bootstrap().identity,
                                              device, topology, connection.communication());
    const auto requirement = preparing->memory();
    StartupMemoryBudget budget(initial, device_budget(initial.available_bytes()),
                               std::max(kDefaultContextHeadroomBytes, requirement.headroom_bytes),
                               requirement.host_reservation_bytes);
    const auto before_weights    = device_memory_snapshot(device.device());
    const auto available_weights = budget.available(before_weights);
    if (requirement.weight_bytes > available_weights ||
        (before_weights.shared_physical_memory &&
         requirement.host_weight_bytes > available_weights - requirement.weight_bytes))
        throw DeviceAllocationError("rank immutable weights exceed device budget",
                                    requirement.weight_bytes);
    if (requirement.host_weight_bytes > budget.available_host(before_weights))
        throw std::invalid_argument("rank immutable Host weights exceed available RAM");
    preparing->materialize();
    reader.reset();
    const auto after_weights    = device_memory_snapshot(device.device());
    std::uint32_t attempt       = 0;
    std::size_t maximum_runtime = SIZE_MAX;
    ContextCapacityPlan capacity_plan;
    const auto report = [&] {
        const auto snapshot  = device_memory_snapshot(device.device());
        const auto available = budget.available(snapshot);
        wire::Encoder out;
        out.u64(topology.plan);
        out.u32(attempt);
        auto value                     = preparing->report(available);
        value.capacity.available_bytes = std::min(value.capacity.available_bytes, maximum_runtime);
        value.capacity.available_host_bytes   = budget.available_host(snapshot);
        value.capacity.shared_physical_memory = snapshot.shared_physical_memory;
        capacity_plan                         = value.capacity.plan;
        value.hardware_class                  = context_cost_hardware_class(
            device.properties().name, device.properties().major, device.properties().minor);
        value.cuda_sync_mode           = device.sync_mode();
        value.free_after_weights_bytes = after_weights.available_bytes();
        encode_preparation_report(out, value);
        connection.send(wire::Kind::Prepared, std::move(out).take());
    };
    report();
    bool entered = false, accepted = false;
    ContextCapacity allocated_capacity;
    DeviceMemorySnapshot allocation_snapshot{};
    std::size_t control_reservation_bytes = 0;
    std::string failure;
    const auto describe = [&] {
        auto result = preparing->describe();
        connection.communication().sample_startup_memory();
        const auto after                            = device_memory_snapshot(device.device());
        auto& observation                           = result.observation;
        const auto& identity                        = connection.bootstrap().identity;
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
        result.budget.available_after_weights_bytes = after_weights.available_bytes();
        result.budget.available_after_startup_bytes = after.available_bytes();
        result.budget.planned_slack_bytes           = budget.available(after);
        result.budget.retained_bytes =
            budget.retained(allocation_snapshot, requirement.weight_bytes);
        accepted = budget.fits(after);
        if (!accepted) {
            const auto cost = result.budget.runtime_reservation_bytes +
                              (after.shared_physical_memory ? control_reservation_bytes : 0);
            const auto deficit = budget.shortfall(after);
            maximum_runtime =
                std::min(maximum_runtime, budget.corrected_runtime_limit(after, cost));
            failure = "startup memory budget exceeded by " + std::to_string(deficit) + " bytes";
        }
        wire::Encoder out;
        if (accepted) encode(out, result);
        return std::move(out).take();
    };
    const auto allocated = [&](wire::Bytes description) {
        wire::Encoder out;
        out.u64(topology.plan);
        out.u32(attempt);
        out.boolean(accepted);
        out.string(failure);
        out.bytes(description);
        connection.send(wire::Kind::Allocated, std::move(out).take());
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
        command.finish();
        if (action == PreparationAction::PrepareCommunication) {
            preparing->prepare_communication();
            device.synchronize();
            connection.communication().sample_startup_memory();
            report();
            continue;
        }
        if (action == PreparationAction::PrepareProbe ||
            action == PreparationAction::PrepareRebuildProbe) {
            bool success = true;
            try {
                if constexpr (requires {
                                  preparing->prepare_probe();
                                  preparing->prepare_rebuild_probe();
                              }) {
                    if (action == PreparationAction::PrepareProbe)
                        preparing->prepare_probe();
                    else
                        preparing->prepare_rebuild_probe();
                } else
                    throw std::logic_error("rank preparation has no calibration probe");
            } catch (const DeviceAllocationError& error) {
                success = false;
                failure = error.what();
            }
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
            if constexpr (requires {
                              preparing->calibrate();
                              preparing->calibrate_rebuild();
                          }) {
                if (action == PreparationAction::Calibrate)
                    preparing->calibrate();
                else
                    preparing->calibrate_rebuild();
            } else
                throw std::logic_error("rank preparation has no calibration phase");
            report();
            continue;
        }
        if (action == PreparationAction::Release) {
            if (!entered) throw std::logic_error("rank has not entered this allocation attempt");
            preparing->release_resources();
            device.synchronize();
            failure.clear();
            entered = accepted = false;
            ++attempt;
            report();
            continue;
        }
        if (action == PreparationAction::Activate) {
            if (!accepted || capacity != allocated_capacity)
                throw std::logic_error("rank activation has no accepted allocation");
            return attempt;
        }
        if (action == PreparationAction::CheckMemory) {
            if (!entered || capacity != allocated_capacity)
                throw std::logic_error("rank memory check has no allocation");
            allocated(describe());
            continue;
        }
        if (action == PreparationAction::PrepareExecution) {
            if (!accepted || capacity != allocated_capacity)
                throw std::logic_error("rank execution preparation has no accepted allocation");
            if constexpr (requires { preparing->prepare_execution(); })
                preparing->prepare_execution();
            allocated(describe());
            continue;
        }
        if (action != PreparationAction::Allocate || entered)
            throw std::logic_error("rank allocation is already entered");
        const auto before         = device_memory_snapshot(device.device());
        allocation_snapshot       = before;
        const auto available      = budget.available(before);
        control_reservation_bytes = planned_control_bytes;
        const auto local_control  = capacity_plan.control_bytes(capacity);
        const auto remote_control =
            planned_control_bytes > local_control ? planned_control_bytes - local_control : 0;
        auto remaining      = available;
        auto host_remaining = budget.available_host(before);
        if (before.shared_physical_memory)
            remaining = remaining > remote_control ? remaining - remote_control : 0;
        else
            host_remaining = host_remaining > remote_control ? host_remaining - remote_control : 0;
        preparing->prepare(capacity, remaining, host_remaining);
        auto description   = describe();
        entered            = true;
        allocated_capacity = capacity;
        allocated(std::move(description));
    }
}

} // namespace ninfer::runtime
