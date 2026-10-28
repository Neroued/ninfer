#include "runtime/engine/worker.h"
#include "distributed/bootstrap.h"

#include "models/qwen4_exp/parallel/preparation.h"
#include "models/glm5next/startup.h"
#include "models/glm5next/parallel/codec.h"
#include "models/registry.h"
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
            const auto observer = connection.startup_observer();
            StartupPhaseScope phase(observer, StartupPhase::EngineFinalize);
            auto codec = preparing->codec(connection.topology());
            instance   = std::move(*preparing).finish();
            preparing.reset();
            WorkerLoop<typename Types::ExecutionTypes, typename Types::RankProgram, decltype(codec)>
                loop(connection, *instance->program, std::move(codec));
            wire::Encoder ready;
            ready.u64(connection.topology().plan);
            ready.u32(*attempt);
            phase.complete();
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

} // namespace

void run_worker(distributed::WorkerBootstrap bootstrap) {
    distributed::WorkerConnection connection(std::move(bootstrap));
    try {
        std::cerr << distributed::rank_description(connection.bootstrap().identity) << std::endl;
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
        const auto observer = connection.startup_observer();
        std::unique_ptr<artifact::Reader> reader;
        {
            StartupPhaseScope phase(observer, StartupPhase::ArtifactInspect);
            reader = std::make_unique<artifact::Reader>(connection.bootstrap().artifact);
            phase.complete();
        }
        switch (models::resolve_architecture(reader->directory())) {
        case models::Architecture::Qwen4Exp:
            run_rank<models::qwen4_exp::RankPreparation>(connection, std::move(reader), payload);
            break;
        case models::Architecture::Glm5Next:
            run_rank<models::glm5next::RankPreparation>(connection, std::move(reader), payload);
            break;
        default:
            throw std::invalid_argument("architecture has no distributed worker implementation");
        }
    } catch (const std::exception& error) {
        connection.fail(error.what());
        throw;
    }
}

} // namespace ninfer::runtime
