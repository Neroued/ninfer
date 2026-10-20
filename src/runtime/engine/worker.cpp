#include "runtime/engine/worker.h"
#include "distributed/bootstrap.h"

#include "models/qwen4_exp/parallel/preparation.h"
#include "runtime/executor/preparation.h"
#include "runtime/executor/worker_loop.h"
#include "runtime/engine/startup_memory.h"
#include "runtime/engine/context_cache/context_cost.h"
#include "core/platform.h"

#include <thread>
#include <iostream>

namespace ninfer::runtime {
namespace {
namespace wire = distributed::wire;

wire::Message receive(distributed::WorkerConnection& connection) {
    for (;;) {
        connection.communication().check();
        if (auto message = connection.poll()) return std::move(*message);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

// All owning model objects outlive this catch boundary. Abort communication before their
// destruction can wait on device work, both during preparation and after activation.
template <class Preparation>
void run_rank(distributed::WorkerConnection& connection, std::unique_ptr<artifact::Reader> reader,
              std::span<const std::byte> payload) {
    using Types          = typename Preparation::Types;
    auto& device         = connection.device();
    const auto& topology = connection.topology();
    std::unique_ptr<Preparation> preparing;
    std::unique_ptr<typename Preparation::Instance> instance;
    try {
        const auto initial = device_memory_snapshot(device.device());
        preparing = std::make_unique<Preparation>(*reader, payload, connection.bootstrap().identity,
                                                  device, topology, connection.communication());
        const auto requirement = preparing->memory();
        StartupMemoryBudget budget(
            initial, device_budget(initial.available_bytes()), requirement.headroom_bytes,
            NINFER_SHARED_PHYSICAL_MEMORY ? requirement.host_reservation_bytes : 0);
        if (requirement.weight_bytes > budget.available(device_memory_snapshot(device.device())))
            throw DeviceAllocationError("rank immutable weights exceed device budget",
                                        requirement.weight_bytes);
        preparing->materialize();
        reader.reset();
        const auto after_weights = device_memory_snapshot(device.device());
        std::uint32_t attempt    = 0;
        std::size_t correction   = 0;
        ContextCapacityCurve capacity_curve;
        const auto report = [&] {
            const auto available = budget.available(device_memory_snapshot(device.device()));
            wire::Encoder out;
            out.u64(topology.plan);
            out.u32(attempt);
            auto value     = preparing->report(available > correction ? available - correction : 0);
            capacity_curve = value.capacity.curve;
            value.hardware_class = context_cost_hardware_class(
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
        std::size_t opaque_growth             = 0;
        std::size_t control_reservation_bytes = 0;
        const auto describe                   = [&] {
            auto result                                 = preparing->describe();
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
            result.budget.headroom_bytes                = budget.headroom();
            result.budget.available_after_weights_bytes = after_weights.available_bytes();
            result.budget.available_after_startup_bytes = after.available_bytes();
            result.budget.planned_slack_bytes           = budget.available(after);
            result.budget.retained_bytes =
                budget.retained(allocation_snapshot, requirement.weight_bytes);
            const auto planned =
                result.budget.runtime_reservation_bytes + control_reservation_bytes +
                (result.budget.graphs ? result.budget.graphs->rebuild_estimate_bytes : 0);
            const auto observed = device_memory_growth(allocation_snapshot, after);
            opaque_growth = observed > static_cast<std::int64_t>(planned) ? observed - planned : 0;
            accepted      = budget.fits(after);
            wire::Encoder out;
            if (accepted) encode(out, result);
            return std::move(out).take();
        };
        const auto allocated = [&](wire::Bytes description) {
            wire::Encoder out;
            out.u64(topology.plan);
            out.u32(attempt);
            out.boolean(accepted);
            out.bytes(description);
            connection.send(wire::Kind::Allocated, std::move(out).take());
        };
        for (;;) {
            auto message = receive(connection);
            if (message.kind == wire::Kind::Shutdown) {
                preparing.reset();
                connection.shutdown();
                return;
            }
            if (message.kind != wire::Kind::Enter)
                throw std::logic_error("unexpected rank preparation command");
            wire::Decoder command(message.payload);
            if (command.u64() != topology.plan || command.u32() != attempt)
                throw std::logic_error("rank preparation attempt differs");
            const auto action = static_cast<PreparationAction>(command.u8());
            const ContextCapacity capacity{command.u32(), command.u32()};
            command.finish();
            if (action == PreparationAction::PrepareProbe) {
                bool success = true;
                try {
                    preparing->prepare_probe();
                } catch (const DeviceAllocationError&) { success = false; }
                wire::Encoder out;
                out.u64(topology.plan);
                out.u32(attempt);
                out.boolean(success);
                connection.send(wire::Kind::Allocated, std::move(out).take());
                continue;
            }
            if (action == PreparationAction::Calibrate) {
                preparing->calibrate();
                report();
                continue;
            }
            if (action == PreparationAction::Release) {
                if (!entered)
                    throw std::logic_error("rank has not entered this allocation attempt");
                preparing->release_program();
                const auto retained = std::max<std::int64_t>(
                    0, device_memory_growth(allocation_snapshot,
                                            device_memory_snapshot(device.device())));
                correction =
                    std::max(correction, opaque_growth > static_cast<std::uint64_t>(retained)
                                             ? opaque_growth - retained
                                             : 0);
                entered = accepted = false;
                ++attempt;
                report();
                continue;
            }
            if (action == PreparationAction::Activate) {
                if (!accepted || capacity != allocated_capacity)
                    throw std::logic_error("rank activation has no accepted allocation");
                auto codec = preparing->codec(topology);
                instance   = std::move(*preparing).finish();
                preparing.reset();
                WorkerLoop<typename Types::ExecutionTypes, typename Types::RankProgram,
                           decltype(codec)>
                    loop(connection, *instance->program, std::move(codec));
                wire::Encoder ready;
                ready.u64(topology.plan);
                ready.u32(attempt);
                connection.send(wire::Kind::Ready, std::move(ready).take());
                while (loop.advance()) std::this_thread::yield();
                instance.reset();
                connection.shutdown();
                return;
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
                preparing->prepare_execution();
                allocated(describe());
                continue;
            }
            if (action != PreparationAction::Allocate || entered)
                throw std::logic_error("rank allocation is already entered");
            const auto before         = device_memory_snapshot(device.device());
            allocation_snapshot       = before;
            opaque_growth             = 0;
            const auto available      = budget.available(before);
            control_reservation_bytes = capacity_curve.control_bytes(capacity);
            wire::Bytes description;
            try {
                preparing->prepare(capacity, available > correction ? available - correction : 0);
                description = describe();
            } catch (const DeviceAllocationError&) { accepted = false; }
            entered            = true;
            allocated_capacity = capacity;
            allocated(std::move(description));
        }
    } catch (const std::exception& error) {
        connection.fail(error.what());
        throw;
    }
}

void warm_communication(distributed::WorkerConnection& connection) {
    auto& device = connection.device();
    DeviceBuffer words(device.device(), 4096);
    try {
        words.fill();
        for (const auto& group : connection.topology().groups) {
            const auto rank = connection.bootstrap().identity.rank;
            if (std::find(group.members.begin(), group.members.end(), rank) == group.members.end())
                continue;
            connection.communication().group(group.id).enqueue(
                distributed::Collective::AllReduce, words.data(), words.data(), 1024, DType::FP32,
                distributed::Reduction::Sum, 0, device.execution_view());
        }
        device.synchronize();
    } catch (const std::exception& error) {
        connection.fail(error.what());
        throw;
    }
}
} // namespace

void run_worker(distributed::WorkerBootstrap bootstrap) {
    distributed::WorkerConnection connection(std::move(bootstrap));
    try {
        std::cerr << distributed::rank_description(connection.bootstrap().identity) << std::endl;
        warm_communication(connection);
        auto message = receive(connection);
        if (message.kind == wire::Kind::Shutdown) {
            connection.shutdown();
            return;
        }
        if (message.kind != wire::Kind::Prepare)
            throw std::runtime_error("worker expected a model preparation request");
        wire::Decoder in(message.payload);
        if (in.u64() != connection.topology().plan)
            throw std::runtime_error("worker model plan identity differs");
        const auto payload = in.bytes();
        in.finish();
        run_rank<models::qwen4_exp::RankPreparation>(
            connection, std::make_unique<artifact::Reader>(connection.bootstrap().artifact),
            payload);
    } catch (const std::exception& error) {
        connection.fail(error.what());
        throw;
    }
}

} // namespace ninfer::runtime
