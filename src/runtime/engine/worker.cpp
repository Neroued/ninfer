#include "runtime/engine/worker.h"
#include "distributed/bootstrap.h"

#include "models/qwen4_exp/parallel/preparation.h"
#include "runtime/executor/rank_preparation.h"
#include "runtime/executor/worker_loop.h"

#include <thread>
#include <iostream>

namespace ninfer::runtime {
namespace {
namespace wire = distributed::wire;

// Abort communication before releasing owners that may wait on device work.
template <class Preparation>
void run_rank(distributed::WorkerConnection& connection, std::unique_ptr<artifact::Reader> reader,
              std::span<const std::byte> payload) {
    using Types = typename Preparation::Types;
    std::unique_ptr<Preparation> preparing;
    std::unique_ptr<typename Preparation::Instance> instance;
    try {
        const auto attempt = prepare_rank(connection, std::move(reader), payload, preparing);
        if (attempt) {
            auto codec = preparing->codec(connection.topology());
            instance   = std::move(*preparing).finish();
            preparing.reset();
            WorkerLoop<typename Types::ExecutionTypes, typename Types::RankProgram, decltype(codec)>
                loop(connection, *instance->program, std::move(codec));
            wire::Encoder ready;
            ready.u64(connection.topology().plan);
            ready.u32(*attempt);
            connection.send(wire::Kind::Ready, std::move(ready).take());
            while (loop.advance()) std::this_thread::yield();
        }
        preparing.reset();
        instance.reset();
        connection.shutdown();
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
        auto message = receive_preparation_command(connection);
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
