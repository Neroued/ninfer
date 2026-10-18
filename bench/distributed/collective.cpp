#include "core/buffer.h"
#include "core/decode_graph.h"
#include "distributed/bootstrap.h"
#include "ninfer/ops/collective.h"
#include "product/config/options.h"

#include <array>
#include <iomanip>
#include <iostream>
#include <thread>

namespace {
using namespace ninfer;
namespace dist = distributed;
namespace wire = distributed::wire;
using Clock    = std::chrono::steady_clock;

void wait(cudaEvent_t event, dist::WorkerConnection& connection) {
    for (;;) {
        const auto status = cudaEventQuery(event);
        if (status == cudaSuccess) return;
        if (status != cudaErrorNotReady) CUDA_CHECK(status);
        connection.communication().check();
        std::this_thread::yield();
    }
}

struct Events {
    cudaEvent_t start{}, end{};

    Events() {
        CUDA_CHECK(cudaEventCreate(&start));
        try {
            CUDA_CHECK(cudaEventCreate(&end));
        } catch (...) {
            cudaEventDestroy(start);
            throw;
        }
    }

    ~Events() {
        cudaEventDestroy(end);
        cudaEventDestroy(start);
    }
};

int worker() {
    dist::WorkerConnection connection(dist::read_worker_bootstrap());
    try {
        {
            auto& device      = connection.device();
            const auto& group = connection.communication().group(0);
            DeviceBuffer input(device.device(), 20ULL << 20), output(device.device(), 20ULL << 20);
            {
                std::vector<std::uint16_t> values(input.size() / sizeof(std::uint16_t));
                std::uint64_t state = 0x123456789abcdefULL + group.rank();
                for (auto& value : values) {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    value = static_cast<std::uint16_t>(0x3f00U | (state & 0x7fU));
                }
                input.copy_from_host(values.data(), input.size());
            }
            Events events;
            connection.send(wire::Kind::Ready);
            for (;;) {
                auto command = connection.poll();
                if (!command) {
                    std::this_thread::yield();
                    continue;
                }
                if (command->kind == wire::Kind::Shutdown) break;
                if (command->kind != wire::Kind::Prepare)
                    throw std::logic_error("unexpected collective benchmark command");
                wire::Decoder request(command->payload);
                const auto bytes = request.u64();
                const auto chain = request.u32(), repeats = request.u32();
                const bool graph = request.boolean();
                request.finish();
                if (!bytes || bytes > input.size() || bytes % 2 || !chain || !repeats)
                    throw std::invalid_argument("invalid collective benchmark shape");
                const auto type      = bytes == 8 ? DType::U64 : DType::BF16;
                const auto reduction = bytes == 8 ? dist::Reduction::Max : dist::Reduction::Sum;
                Tensor x(input.span(), type, {static_cast<int>(bytes / dtype_size(type))});
                Tensor y(output.span(), type, {static_cast<int>(bytes / dtype_size(type))});
                const auto enqueue = [&] {
                    for (std::uint32_t i = 0; i < chain; ++i)
                        ops::all_reduce(x, y, reduction, group, device.execution_view());
                };
                DecodeGraphDefinition definition;
                DecodeGraphExecutable executable;
                try {
                    enqueue();
                    CUDA_CHECK(cudaEventRecord(events.end, device.stream()));
                    wait(events.end, connection);
                    if (graph) {
                        definition.capture(device.stream(), enqueue);
                        executable.instantiate(definition);
                    }
                    const auto run = [&] {
                        if (graph)
                            executable.launch(device.stream());
                        else
                            enqueue();
                    };
                    for (unsigned i = 0; i < 5; ++i) run();
                    CUDA_CHECK(cudaEventRecord(events.end, device.stream()));
                    wait(events.end, connection);
                    // A common start fence keeps prior control-plane skew outside the timed range.
                    ops::all_reduce(x, y, reduction, group, device.execution_view());
                    CUDA_CHECK(cudaEventRecord(events.start, device.stream()));
                    const auto host_start = Clock::now();
                    for (std::uint32_t i = 0; i < repeats; ++i) run();
                    const auto host_end = Clock::now();
                    CUDA_CHECK(cudaEventRecord(events.end, device.stream()));
                    wait(events.end, connection);
                    float ms = 0;
                    CUDA_CHECK(cudaEventElapsedTime(&ms, events.start, events.end));
                    wire::Encoder result;
                    result.f64(ms * 1000.0 / repeats);
                    result.f64(
                        std::chrono::duration<double, std::micro>(host_end - host_start).count() /
                        repeats);
                    connection.send(wire::Kind::Ready, std::move(result).take());
                } catch (const std::exception& error) {
                    connection.fail(error.what());
                    throw;
                }
            }
        }
        connection.shutdown();
        return 0;
    } catch (const std::exception& error) {
        connection.fail(error.what());
        throw;
    }
}

int coordinator(int argc, char** argv) {
    if (argc != 3 || std::string_view(argv[1]) != "--config")
        throw std::invalid_argument("usage: ninfer_collective_bench --config FILE");
    const auto deployment = product::read_deployment_file(argv[2]);
    dist::ExecutionTopology topology;
    topology.plan        = 1;
    topology.output_rank = 0;
    const auto size      = static_cast<std::uint32_t>(deployment.ranks.size());
    if (size != 2) throw std::invalid_argument("collective benchmark requires two ranks");
    topology.ranks.resize(size);
    for (const auto& r : deployment.ranks)
        topology.ranks[r.rank] = {r.rank,         r.node,         0,  r.device,
                                  {r.rank, size}, {r.rank, size}, {}, {}};
    topology.groups = {{0, {0, 1}}};
    topology.roles  = {{"tp", 0}, {"ep", 0}};
    const std::array<std::string, 1> args{"--worker"};
    dist::ExecutionGroup group(
        deployment, topology,
        [](auto rank, auto line) { std::cerr << "rank=" << rank << " " << line << '\n'; }, args);
    const auto receive = [&] {
        std::vector<wire::Bytes> results(size);
        std::vector<bool> seen(size);
        std::size_t count   = 0;
        const auto deadline = Clock::now() + std::chrono::seconds(90);
        while (count != size) {
            if (Clock::now() >= deadline)
                throw std::runtime_error("collective benchmark timed out");
            if (auto value = group.transport().poll()) {
                if (value->message.kind == wire::Kind::Failed) {
                    wire::Decoder failure(value->message.payload);
                    throw std::runtime_error(failure.string());
                }
                if (value->message.epoch != group.epoch() ||
                    value->message.kind != wire::Kind::Ready || seen.at(value->peer))
                    throw std::runtime_error("invalid benchmark completion");
                seen[value->peer]    = true;
                results[value->peer] = std::move(value->message.payload);
                ++count;
            } else {
                group.check_processes();
                std::this_thread::yield();
            }
        }
        return results;
    };
    (void)receive();
    std::cout
        << "bytes,chain,mode,rank0_gpu_us,rank1_gpu_us,max_gpu_us,host_enqueue_us,logical_GBps\n";
    for (const std::uint64_t bytes : {8ULL, 5120ULL, 40960ULL, 327680ULL, 5ULL << 20, 20ULL << 20})
        for (const std::uint32_t chain : {1U, 96U})
            for (bool graph : {false, true}) {
                wire::Encoder request;
                request.u64(bytes);
                request.u32(chain);
                request.u32(chain == 1 ? 100 : 10);
                request.boolean(graph);
                auto message = std::make_shared<wire::Message>(
                    wire::Message{wire::Kind::Prepare, group.epoch(), std::move(request).take()});
                const std::array sends{dist::OutgoingMessage{0, message},
                                       dist::OutgoingMessage{1, message}};
                if (!group.transport().try_send(sends))
                    throw std::runtime_error("benchmark command queue is full");
                const auto results = receive();
                std::array<double, 2> gpu{}, host{};
                for (unsigned r = 0; r < size; ++r) {
                    wire::Decoder result(results[r]);
                    gpu[r]  = result.f64();
                    host[r] = result.f64();
                    result.finish();
                }
                const auto maximum = std::max(gpu[0], gpu[1]);
                std::cout << bytes << ',' << chain << ',' << (graph ? "graph" : "eager") << ','
                          << std::fixed << std::setprecision(3) << gpu[0] << ',' << gpu[1] << ','
                          << maximum << ',' << std::max(host[0], host[1]) << ','
                          << bytes * chain / maximum / 1000.0 << '\n';
                std::cout.flush();
            }
    group.shutdown();
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--worker") return worker();
        return coordinator(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
