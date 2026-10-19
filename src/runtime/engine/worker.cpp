#include "runtime/engine/worker.h"
#include "distributed/bootstrap.h"

#include "models/qwen4_exp/parallel/preparation.h"
#include "models/qwen4_exp/parallel/wire.h"
#include "runtime/executor/preparation.h"
#include "runtime/engine/startup_memory.h"
#include "core/platform.h"

#include <thread>
#include <iostream>

namespace ninfer::runtime {
namespace q4 = models::qwen4_exp;

namespace {
namespace wire = distributed::wire;

wire::Message receive(distributed::WorkerConnection& connection) {
    for (;;) {
        connection.communication().check();
        if (auto message = connection.poll()) return std::move(*message);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void run_qwen4_exp_preparation(distributed::WorkerConnection& connection, wire::Bytes payload) {
    wire::Decoder in(payload);
    auto request = q4::decode_rank_load(in);
    in.finish();
    const auto& topology = connection.topology();
    const auto& identity = connection.bootstrap().identity;
    if (request.rank.rank != identity.rank || request.rank.size != identity.tp.size ||
        identity.ep.size != request.rank.size)
        throw std::invalid_argument("Qwen4Exp rank plan differs from worker identity");
    artifact::Reader reader(connection.bootstrap().artifact);
    if (request.artifact != reader.artifact_id())
        throw artifact::ArtifactError("Qwen4Exp worker artifact differs from coordinator");
    auto definition = q4::read_definition(reader, models::load_options(request.options));
    auto& device    = connection.device();
    auto load       = q4::resolve_rank_load(reader, std::move(definition), std::move(request.rank),
                                            device.device());
    const auto initial = device_memory_snapshot(device.device());
    runtime::StartupMemoryBudget budget(
        initial, runtime::device_budget(initial.free_bytes),
        request.options.kv_capacity.automatic_headroom_bytes,
        NINFER_SHARED_PHYSICAL_MEMORY ? request.host_reservation_bytes : 0);
    if (load.materialization().device_capacity_bytes > budget.available(initial))
        throw DeviceAllocationError("rank immutable weights exceed device budget",
                                    load.materialization().device_capacity_bytes);
    auto prepared =
        std::make_unique<q4::RankPreparation>(std::move(load), std::move(request.options), device);
    std::uint32_t attempt  = 0;
    std::size_t correction = 0;
    const auto report      = [&] {
        const auto& s        = prepared->model().storage_stats();
        const auto available = budget.available(device_memory_snapshot(device.device()));
        runtime::RankPreparationReport r{
                 {prepared->capacity_curve(), available > correction ? available - correction : 0},
            s.device_capacity_bytes + prepared->model().metadata_bytes(),
            s.host_capacity_bytes,
            s.peak_staging_bytes,
            s.read_bytes,
            s.h2d_bytes,
            s.upload_seconds};
        wire::Encoder out;
        out.u64(topology.plan);
        out.u32(attempt);
        runtime::encode_preparation_report(out, r);
        connection.send(wire::Kind::Prepared, std::move(out).take());
    };
    report();
    bool allocated = false;
    DeviceMemorySnapshot allocation_snapshot{};
    std::size_t opaque_growth = 0;
    for (;;) {
        auto message = receive(connection);
        if (message.kind == wire::Kind::Shutdown) {
            prepared.reset();
            connection.shutdown();
            return;
        }
        if (message.kind != wire::Kind::Enter)
            throw std::logic_error("unexpected Qwen4Exp preparation command");
        wire::Decoder command(message.payload);
        if (command.u64() != topology.plan || command.u32() != attempt)
            throw std::logic_error("Qwen4Exp preparation attempt differs");
        const auto action = static_cast<runtime::PreparationAction>(command.u8());
        const auto pages  = command.u32();
        command.finish();
        if (action == runtime::PreparationAction::Release) {
            if (!allocated) throw std::logic_error("rank resources have not entered this attempt");
            prepared->release_resources();
            const auto retained = std::max<std::int64_t>(
                0,
                device_memory_growth(allocation_snapshot, device_memory_snapshot(device.device())));
            correction = std::max(correction, opaque_growth > static_cast<std::uint64_t>(retained)
                                                  ? opaque_growth - retained
                                                  : 0);
            allocated  = false;
            ++attempt;
            report();
            continue;
        }
        if (action != runtime::PreparationAction::Allocate || allocated)
            throw std::logic_error("Qwen4Exp rank resources already entered");
        const auto before    = device_memory_snapshot(device.device());
        allocation_snapshot  = before;
        opaque_growth        = 0;
        const auto available = budget.available(before);
        bool success         = false;
        try {
            prepared->prepare(pages, available > correction ? available - correction : 0);
            const auto after    = device_memory_snapshot(device.device());
            const auto planned  = prepared->resources().plan.device_reservation_bytes();
            const auto observed = device_memory_growth(before, after);
            opaque_growth       = observed > static_cast<std::int64_t>(planned)
                                      ? static_cast<std::size_t>(observed - planned)
                                      : 0;
            success             = budget.fits(after);
        } catch (const DeviceAllocationError&) {
            // Rebuild only mutable resources. The next report charges retained allocator memory.
            success = false;
        }
        allocated = true;
        wire::Encoder out;
        out.u64(topology.plan);
        out.u32(attempt);
        out.boolean(success);
        wire::Encoder layout;
        if (success) q4::encode_rank_layout(layout, prepared->layout());
        out.bytes(std::move(layout).take());
        connection.send(wire::Kind::Ready, std::move(out).take());
    }
}

} // namespace

void run_worker(distributed::WorkerBootstrap bootstrap) {
    distributed::WorkerConnection connection(std::move(bootstrap));
    try {
        std::cerr << distributed::rank_description(connection.bootstrap().identity) << std::endl;
        // Settle lazy communicator allocations before taking the model memory budget snapshot.
        {
            auto& device = connection.device();
            DeviceBuffer words(device.device(), 4096);
            words.fill();
            for (const auto& g : connection.topology().groups) {
                const auto rank = connection.bootstrap().identity.rank;
                if (std::find(g.members.begin(), g.members.end(), rank) == g.members.end())
                    continue;
                connection.communication().group(g.id).enqueue(
                    distributed::Collective::AllReduce, words.data(), words.data(), 1024,
                    DType::FP32, distributed::Reduction::Sum, 0, device.execution_view());
            }
            device.synchronize();
        }
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
        wire::Bytes native(payload.begin(), payload.end());
        in.finish();
        run_qwen4_exp_preparation(connection, std::move(native));
    } catch (const std::exception& error) {
        connection.fail(error.what());
        throw;
    }
}

} // namespace ninfer::runtime
