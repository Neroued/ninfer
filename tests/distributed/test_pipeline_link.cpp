#include "core/buffer.h"
#include "distributed/bootstrap.h"
#include "distributed/communication/preparation.h"
#include "distributed/pipeline_link.h"
#include "product/config/options.h"

#include <algorithm>
#include <atomic>
#include <iostream>
#include <thread>

namespace {
using namespace ninfer;
namespace dist = distributed;
namespace wire = dist::wire;

void require(bool value, const char* reason) {
    if (!value) throw std::runtime_error(reason);
}

void run(dist::WorkerConnection& connection) {
    const auto rank      = connection.identity().rank;
    const auto& topology = connection.topology();
    const auto* input    = topology.pipeline_input(rank);
    const auto* output   = topology.pipeline_output(rank);
    require(bool(input) != bool(output), "PP2 requires one directed link per rank");
    const auto& edge               = input ? *input : *output;
    constexpr std::size_t capacity = 1024;
    const std::array requirements{dist::PpStageLink::requirements(edge, rank, capacity)};
    if (!dist::prepare_communication(connection, requirements,
                                     connection.bootstrap().communication_policy))
        throw std::runtime_error("pipeline communication stopped during preparation");
    auto& device      = connection.device();
    const auto gpu    = device.device();
    const auto stream = device.execution_view();
    auto& group       = connection.communication().group(edge.id);
    DeviceBuffer backing(gpu, dist::PpStageLink::slots * capacity);
    dist::PpStageLink link(group, device, backing.span());
    std::array<DeviceBuffer, 3> residual, metadata;
    const auto value = [](std::size_t i, bool metadata) {
        return std::uint8_t(17 + 31 * i + 2 * metadata);
    };
    for (std::size_t i = 0; i < residual.size(); ++i) {
        residual[i] = DeviceBuffer(gpu, 101 + 47 * i);
        metadata[i] = DeviceBuffer(gpu, 27 + i);
        residual[i].fill(output ? value(i, false) : 0);
        metadata[i].fill(output ? value(i, true) : 0);
    }
    const auto transfer = [&](std::size_t i) {
        if (output) {
            const std::array<ConstMemorySpan, 2> fields{residual[i].span(), metadata[i].span()};
            return link.try_send(10 + i, fields, stream);
        }
        const std::array fields{residual[i].span(), metadata[i].span()};
        return link.try_receive(10 + i, fields, stream);
    };

    // Hold the consumer/producer stream so backpressure does not depend on network timing.
    struct Gate {
        std::atomic<bool> open = false, exited = false;

        void release() {
            open = true;
            open.notify_all();
            exited.wait(false);
        }
    } gate;

    CUDA_CHECK(cudaLaunchHostFunc(
        stream.stream(),
        [](void* payload) {
            auto& gate = *static_cast<Gate*>(payload);
            gate.open.wait(false);
            gate.exited = true;
            gate.exited.notify_all();
        },
        &gate));
    std::optional<dist::PpStageLink::Transfer> first, second;
    try {
        first  = transfer(0);
        second = transfer(1);
        require(first && second, "prepared link did not admit both slots");
        require(!transfer(2) && !link.retired(*first), "link reused a live packet");
    } catch (...) {
        gate.release();
        throw;
    }
    gate.release();
    link.drain();
    require(link.retired(*first) && link.retired(*second) && transfer(2),
            "retired packet was not reusable");
    link.drain();
    if (input) {
        for (std::size_t i = 0; i < residual.size(); ++i) {
            for (bool aux : {false, true}) {
                const auto& field = aux ? metadata[i] : residual[i];
                std::vector<std::uint8_t> bytes(field.size());
                field.copy_to_host(bytes.data(), bytes.size());
                require(std::ranges::all_of(bytes, [&](auto b) { return b == value(i, aux); }),
                        "pipeline crossed a tensor field or microbatch boundary");
            }
        }
    }
    const auto stats = group.statistics();
    const auto backend =
        connection.bootstrap().communication_policy == dist::CommunicationPolicy::Native
            ? CommunicationBackend::Native
            : CommunicationBackend::Nccl;
    const auto selected =
        std::ranges::find(stats.backends, backend, &CommunicationBackendStatistics::backend);
    const auto direction = output ? 4U : 5U;
    require(selected != stats.backends.end() && selected->eager.calls[direction] == 3 &&
                selected->eager.payload_bytes[direction] == 528,
            "selected backend did not transfer all three boundary payloads");
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--worker") {
            distributed::WorkerConnection c(distributed::read_worker_bootstrap());
            try {
                run(c);
                c.send(wire::Kind::Ready);
                for (;;) {
                    if (auto msg = c.poll()) {
                        require(msg->kind == wire::Kind::Shutdown,
                                "unexpected pipeline control message");
                        break;
                    }
                    c.wait(std::chrono::milliseconds(5));
                }
                c.shutdown();
            } catch (const std::exception& e) {
                c.fail(e.what());
                throw;
            }
        } else {
            if (argc != 2)
                throw std::invalid_argument("usage: ninfer_pipeline_link_test DEPLOYMENT");
            auto deployment = product::read_deployment_file(argv[1]);
            require(deployment.parallel.pp == 2 && deployment.parallel.tp == 1 &&
                        deployment.communication_policy != distributed::CommunicationPolicy::Auto,
                    "pipeline integration requires pp2/tp1 and an explicit transport");
            auto topology = distributed::build_cluster_topology(deployment);
            const std::array<std::string, 1> args{"--worker"};
            distributed::ExecutionGroup group(
                deployment, std::move(topology), {},
                [](auto rank, auto line) { std::cerr << "rank " << rank << ' ' << line << '\n'; },
                args);
            distributed::prepare_communication(group, std::chrono::seconds(60));
            (void)group.receive_all(wire::Kind::Ready, std::chrono::seconds(90));
            group.shutdown();
            std::cout << "PASS pp2/tp1 three multi-field packets on two slots\n";
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
