#include "runtime/engine/worker.h"
#include "distributed/bootstrap.h"

#include "models/qwen4_exp/parallel/preparation.h"
#include "runtime/executor/preparation.h"
#include "runtime/executor/worker_loop.h"
#include "runtime/engine/startup_memory.h"
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
                                                  device);
        const auto requirement = preparing->memory();
        StartupMemoryBudget budget(
            initial, device_budget(initial.free_bytes), requirement.headroom_bytes,
            NINFER_SHARED_PHYSICAL_MEMORY ? requirement.host_reservation_bytes : 0);
        if (requirement.weight_bytes > budget.available(device_memory_snapshot(device.device())))
            throw DeviceAllocationError("rank immutable weights exceed device budget",
                                        requirement.weight_bytes);
        preparing->materialize();
        reader.reset();
        std::uint32_t attempt  = 0;
        std::size_t correction = 0;
        const auto report      = [&] {
            const auto available = budget.available(device_memory_snapshot(device.device()));
            wire::Encoder out;
            out.u64(topology.plan);
            out.u32(attempt);
            encode_preparation_report(
                out, preparing->report(available > correction ? available - correction : 0));
            connection.send(wire::Kind::Prepared, std::move(out).take());
        };
        report();
        bool entered = false, accepted = false;
        std::uint32_t allocated_pages = 0;
        DeviceMemorySnapshot allocation_snapshot{};
        std::size_t opaque_growth = 0;
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
            const auto pages  = command.u32();
            command.finish();
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
                if (!accepted || pages != allocated_pages)
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
            if (action != PreparationAction::Allocate || entered)
                throw std::logic_error("rank allocation is already entered");
            const auto before    = device_memory_snapshot(device.device());
            allocation_snapshot  = before;
            opaque_growth        = 0;
            const auto available = budget.available(before);
            wire::Encoder description;
            try {
                preparing->prepare(pages, available > correction ? available - correction : 0);
                auto result         = preparing->describe();
                const auto after    = device_memory_snapshot(device.device());
                const auto planned  = result.memory.runtime_reservation_bytes;
                const auto observed = device_memory_growth(before, after);
                opaque_growth       = observed > static_cast<std::int64_t>(planned)
                                          ? static_cast<std::size_t>(observed - planned)
                                          : 0;
                accepted            = budget.fits(after);
                if (accepted) encode(description, result);
            } catch (const DeviceAllocationError&) { accepted = false; }
            entered         = true;
            allocated_pages = pages;
            wire::Encoder out;
            out.u64(topology.plan);
            out.u32(attempt);
            out.boolean(accepted);
            out.bytes(std::move(description).take());
            connection.send(wire::Kind::Allocated, std::move(out).take());
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
