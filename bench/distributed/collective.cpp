#include "core/buffer.h"
#include "core/decode_graph.h"
#include "distributed/communication/preparation.h"
#include "ninfer/ops/collective.h"
#include "ninfer/ops/rmsnorm.h"
#include "product/config/options.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <thread>

namespace {
using namespace ninfer;
namespace dist = distributed;
namespace wire = distributed::wire;
using Clock    = std::chrono::steady_clock;

dist::CommunicationPolicy policy(std::string_view value) {
    if (value == "nccl") return dist::CommunicationPolicy::Nccl;
    if (value == "native") return dist::CommunicationPolicy::Native;
    if (value == "auto") return dist::CommunicationPolicy::Auto;
    throw std::invalid_argument("backend must be nccl, native or auto");
}

std::size_t number(std::string_view value) {
    std::size_t result;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || !result)
        throw std::invalid_argument("benchmark count must be a positive integer");
    return result;
}

struct Options {
    std::string config, backend = "auto", operation = "sum", mode = "graph";
    std::string workload = "collective";
    unsigned width = 4096, maximum_m = 64, samples = 15, chain = 32;
    unsigned work_rows = 256;
    std::vector<std::size_t> bytes;
};

Options options(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; i += 2) {
        const std::string_view flag(argv[i]);
        if (flag == "--help") {
            std::cout
                << "ninfer_collective_bench --config FILE [--backend auto|native|nccl] "
                   "[--op sum|fp16|fp32|words|max|gather|candidates|broadcast|scatter|peer|send] "
                   "[--mode graph|eager] "
                   "[--width 4096] [--max-m 64] [--bytes N,N,...] [--chain 32] [--samples 15]\n"
                   "[--workload collective|pipeline|skew|overlap|compute] [--work-rows 256]\n"
                   "Backend defaults to auto. Graph events time the device body after priming the "
                   "selected operation "
                   "and an untimed collective fence. "
                   "Each reported sample uses the slower rank. Pipeline measures "
                   "RMSNorm -> SUM -> RMSNorm; skew adds rank-1 RMSNorm; overlap runs "
                   "RMSNorm on another stream; compute measures that RMSNorm alone.\n"
                   "Peer measures bidirectional exchange; send measures rank 0 -> rank 1.\n";
            std::exit(0);
        }
        if (i + 1 == argc) throw std::invalid_argument("benchmark option is missing its value");
        const std::string value(argv[i + 1]);
        if (flag == "--config")
            o.config = value;
        else if (flag == "--backend")
            o.backend = value;
        else if (flag == "--op")
            o.operation = value;
        else if (flag == "--mode")
            o.mode = value;
        else if (flag == "--workload")
            o.workload = value;
        else if (flag == "--work-rows")
            o.work_rows = number(value);
        else if (flag == "--width")
            o.width = number(value);
        else if (flag == "--max-m")
            o.maximum_m = number(value);
        else if (flag == "--chain")
            o.chain = number(value);
        else if (flag == "--samples")
            o.samples = number(value);
        else if (flag == "--bytes") {
            std::string_view list(value);
            while (!list.empty()) {
                const auto comma = list.find(',');
                o.bytes.push_back(number(list.substr(0, comma)));
                if (comma == list.npos) break;
                list.remove_prefix(comma + 1);
            }
        } else
            throw std::invalid_argument("unknown benchmark option " + std::string(flag));
    }
    (void)policy(o.backend);
    if (o.config.empty() || (o.mode != "graph" && o.mode != "eager") || o.chain > 100)
        throw std::invalid_argument("invalid config, mode or chain (maximum chain is 100)");
    if (o.workload != "collective" && o.workload != "pipeline" && o.workload != "skew" &&
        o.workload != "overlap" && o.workload != "compute")
        throw std::invalid_argument("unknown benchmark workload");
    if (o.workload != "collective" && o.operation != "sum")
        throw std::invalid_argument("compute workloads require BF16 SUM");
    if (o.bytes.empty())
        for (unsigned m = 1; m <= o.maximum_m; ++m) o.bytes.push_back(2ULL * o.width * m);
    return o;
}

dist::CollectiveCall make_call(std::string_view name, std::size_t bytes) {
    dist::CollectiveCall c{dist::Collective::AllReduce, DType::BF16, 0};
    if (name == "fp16")
        c.dtype = DType::FP16;
    else if (name == "fp32")
        c.dtype = DType::FP32;
    else if (name == "scatter")
        c.operation = dist::Collective::ReduceScatter;
    else if ((name == "peer" || name == "send"))
        c.dtype = DType::U8;
    else if (name == "words")
        c.dtype = DType::U32;
    else if (name == "max") {
        c.dtype     = DType::U64;
        c.reduction = dist::Reduction::Max;
    } else if (name == "gather")
        c.operation = dist::Collective::AllGather;
    else if (name == "candidates") {
        c.operation = dist::Collective::AllGather;
        c.dtype     = DType::I64;
    } else if (name == "broadcast") {
        c.operation = dist::Collective::Broadcast;
        c.root      = 1;
    } else if (name != "sum")
        throw std::invalid_argument("unknown benchmark operation");
    if (bytes % dtype_size(c.dtype))
        throw std::invalid_argument("payload is not a whole number of elements");
    c.count = bytes / dtype_size(c.dtype);
    return c;
}

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
    cudaEvent_t begin{}, end{};

    Events() {
        CUDA_CHECK(cudaEventCreate(&begin));
        try {
            CUDA_CHECK(cudaEventCreate(&end));
        } catch (...) {
            cudaEventDestroy(begin);
            throw;
        }
    }

    ~Events() {
        cudaEventDestroy(end);
        cudaEventDestroy(begin);
    }
};

struct Compute {
    Compute(const DeviceContext& device, unsigned width, unsigned rows, std::size_t bytes)
        : input(device.device(), std::max<std::size_t>(bytes, 2ULL * width * rows)),
          output(device.device(), input.size()), weight(device.device(), 2ULL * width),
          stream(device), begin(device), end(device),
          x(input.span(), DType::BF16, {static_cast<int>(width), static_cast<int>(rows)}),
          y(output.span(), DType::BF16, {static_cast<int>(width), static_cast<int>(rows)}),
          w(weight.span(), DType::BF16, {static_cast<int>(width)}) {
        std::vector<std::uint16_t> values(input.size() / 2, 0x3f80);
        input.copy_from_host(values.data(), input.size());
        weight.copy_from_host(values.data(), weight.size());
    }

    void run(CudaExecutionView cuda) { ops::rmsnorm(x, w, 1e-5f, false, y, cuda); }

    DeviceBuffer input, output, weight;
    CudaStream stream;
    CudaCompletionEvent begin, end;
    Tensor x, y, w;
};

int worker(dist::CommunicationPolicy backend, std::size_t maximum, std::string_view operation) {
    dist::WorkerConnection connection(dist::read_worker_bootstrap());
    const bool point_to_point = operation == "peer" || operation == "send";
    const bool one_way        = operation == "send";
    try {
        const auto call = make_call(operation, maximum);
        std::array requirements{dist::GroupRequirements{
            0,
            {{call.operation, call.dtype, 1, call.count, 1, call.reduction, call.root},
             {dist::Collective::AllReduce, DType::U64, 1, 1, 1, dist::Reduction::Max}},
            {}}};
        if (point_to_point) {
            requirements[0].collectives.erase(requirements[0].collectives.begin());
            const auto rank = connection.bootstrap().identity.rank;
            requirements[0].peers.push_back(
                {1 - rank, one_way && rank == 1 ? 0 : maximum, one_way && rank == 0 ? 0 : maximum});
        }
        if (!dist::prepare_communication(connection, requirements, backend)) {
            connection.shutdown();
            return 0;
        }
        for (const auto& group : connection.communication().statistics())
            for (const auto& route : group.backends) {
                const auto& m = route.memory;
                std::cerr << "communication group=" << group.group << " backend="
                          << (route.backend == CommunicationBackend::Native ? "native" : "nccl")
                          << " owned_host_bytes=" << m.owned_host_bytes
                          << " owned_device_bytes=" << m.owned_device_bytes
                          << " registered_host_bytes=" << m.registered_host_bytes;
                if (m.library)
                    std::cerr << " library_device_bytes=" << m.library->tracked_device_bytes;
                std::cerr << '\n';
            }
        {
            auto& device      = connection.device();
            const auto& group = connection.communication().group(0);
            DeviceBuffer seed(device.device(), 2 * maximum), data(device.device(), 3 * maximum);
            DeviceBuffer fence_storage(device.device(), sizeof(std::uint64_t));
            fence_storage.fill();
            Tensor fence(fence_storage.span(), DType::U64, {1});
            Events events;
            connection.send(wire::Kind::Ready);
            for (;;) {
                auto message = connection.poll();
                if (!message) {
                    std::this_thread::yield();
                    continue;
                }
                if (message->kind == wire::Kind::Shutdown) break;
                if (message->kind != wire::Kind::Prepare)
                    throw std::logic_error("unexpected benchmark command");
                wire::Decoder in(message->payload);
                const auto name  = in.string();
                const auto bytes = in.u64();
                const auto chain = in.u32(), samples = in.u32();
                const bool graph    = in.boolean();
                const auto workload = in.string();
                const auto columns = in.u32(), work_rows = in.u32();
                in.finish();
                const auto c     = make_call(name, bytes);
                const auto width = dtype_size(c.dtype);
                const auto input_bytes =
                    bytes * (c.operation == dist::Collective::ReduceScatter ? 2 : 1);
                std::vector<std::byte> values(input_bytes);
                std::uint64_t random = 0x9e3779b97f4a7c15ULL + group.rank();
                for (std::size_t i = 0; i < input_bytes / width; ++i) {
                    random ^= random << 13;
                    random ^= random >> 7;
                    random ^= random << 17;
                    const std::uint64_t value =
                        c.dtype == DType::BF16   ? 0x3f00U | (random & 127U)
                        : c.dtype == DType::FP16 ? 0x3800U | (random & 1023U)
                        : c.dtype == DType::FP32 ? 0x3f000000U | (random & 0x7fffffU)
                                                 : random;
                    std::memcpy(values.data() + i * width, &value, width);
                }
                seed.copy_from_host(values.data(), input_bytes);
                const auto offset =
                    c.operation == dist::Collective::AllGather ? group.rank() * bytes : 0;
                auto source = Tensor(data.span().subspan(offset, input_bytes), c.dtype,
                                     {static_cast<int>(input_bytes / width)});
                const auto output_offset =
                    point_to_point || c.operation == dist::Collective::ReduceScatter ? 2 * maximum
                                                                                     : 0;
                const auto output_count =
                    c.count * (c.operation == dist::Collective::AllGather ? 2 : 1);
                auto destination = Tensor(data.span().subspan(output_offset), c.dtype,
                                          {static_cast<int>(output_count)});
                std::unique_ptr<Compute> compute;
                if (workload != "collective")
                    compute = std::make_unique<Compute>(device, columns, work_rows, bytes);
                const auto enqueue = [&] {
                    if (point_to_point) {
                        const dist::PeerTransfer transfer{
                            1 - group.rank(),
                            {source.data, one_way && group.rank() == 1 ? 0 : bytes,
                             device.device()},
                            {destination.data, one_way && group.rank() == 0 ? 0 : bytes,
                             device.device()}};
                        group.exchange({&transfer, 1}, device.execution_view());
                        return;
                    }
                    switch (c.operation) {
                    case dist::Collective::AllReduce:
                        ops::all_reduce(source, destination, c.reduction, group,
                                        device.execution_view());
                        break;
                    case dist::Collective::AllGather:
                        ops::all_gather(source, destination, group, device.execution_view());
                        break;
                    case dist::Collective::ReduceScatter:
                        ops::reduce_scatter(source, destination, c.reduction, group,
                                            device.execution_view());
                        break;
                    case dist::Collective::Broadcast:
                        ops::broadcast(destination, c.root, group, device.execution_view());
                        break;
                    default:
                        throw std::logic_error("unsupported benchmark collective");
                    }
                };
                const auto body = [&](bool capturing = false) {
                    CUDA_CHECK(cudaMemcpyAsync(source.data, seed.data(), input_bytes,
                                               cudaMemcpyDeviceToDevice, device.stream()));
                    enqueue();
                    ops::all_reduce(fence, fence, dist::Reduction::Max, group,
                                    device.execution_view());
                    CUDA_CHECK(cudaEventRecordWithFlags(events.begin, device.stream(),
                                                        capturing ? cudaEventRecordExternal : 0));
                    for (unsigned i = 0; i < chain; ++i) {
                        if (workload == "pipeline") {
                            const auto shape = {static_cast<int>(columns),
                                                static_cast<int>(bytes / (2 * columns))};
                            auto x           = source.view(shape);
                            auto produced    = Tensor(compute->input.span(), DType::BF16, shape);
                            auto reduced     = Tensor(compute->output.span(), DType::BF16, shape);
                            ops::rmsnorm(x, compute->w, 1e-5f, false, produced,
                                         device.execution_view());
                            ops::all_reduce(produced, reduced, c.reduction, group,
                                            device.execution_view());
                            ops::rmsnorm(reduced, compute->w, 1e-5f, false, x,
                                         device.execution_view());
                        } else if (workload == "overlap") {
                            compute->begin.record(device.stream());
                            compute->begin.wait(compute->stream.stream());
                            compute->run(device.execution_view(compute->stream.stream()));
                            compute->end.record(compute->stream.stream());
                            enqueue();
                            compute->end.wait(device.stream());
                        } else {
                            if (workload == "compute" || (workload == "skew" && group.rank() == 1))
                                compute->run(device.execution_view());
                            if (workload != "compute") enqueue();
                        }
                    }
                    CUDA_CHECK(cudaEventRecordWithFlags(events.end, device.stream(),
                                                        capturing ? cudaEventRecordExternal : 0));
                };
                DecodeGraphDefinition definition;
                DecodeGraphExecutable executable;
                CommunicationGraphWork work;
                connection.communication().set_active(true);
                body();
                wait(events.end, connection);
                if (graph) {
                    group.record_graph(
                        work, [&] { definition.capture(device.stream(), [&] { body(true); }); });
                    executable.instantiate(definition);
                    executable.upload(device.stream());
                }
                const auto run = [&] {
                    if (graph) {
                        executable.launch(device.stream());
                        group.replayed(work);
                    } else
                        body();
                };
                for (unsigned i = 0; i < 5; ++i) {
                    run();
                    wait(events.end, connection);
                }
                wire::Encoder out;
                out.u32(samples);
                for (unsigned i = 0; i < samples; ++i) {
                    const auto start = Clock::now();
                    run();
                    const auto elapsed =
                        std::chrono::duration<double, std::micro>(Clock::now() - start).count();
                    wait(events.end, connection);
                    float ms = 0;
                    CUDA_CHECK(cudaEventElapsedTime(&ms, events.begin, events.end));
                    out.f64(ms * 1000.0 / chain);
                    out.f64(elapsed);
                }
                connection.communication().set_active(false);
                connection.send(wire::Kind::Ready, std::move(out).take());
            }
        }
        connection.shutdown();
        return 0;
    } catch (const std::exception& error) {
        connection.fail(error.what());
        throw;
    }
}

double percentile(std::vector<double> values, double p) {
    std::sort(values.begin(), values.end());
    const auto position = p * (values.size() - 1);
    const auto index    = static_cast<std::size_t>(position);
    return values[index] +
           (position - index) * (values[std::min(index + 1, values.size() - 1)] - values[index]);
}

int coordinator(int argc, char** argv) {
    const auto o       = options(argc, argv);
    const auto maximum = *std::max_element(o.bytes.begin(), o.bytes.end());
    for (auto bytes : o.bytes) (void)make_call(o.operation, bytes);
    if (o.workload == "pipeline")
        for (auto bytes : o.bytes)
            if (bytes % (2ULL * o.width))
                throw std::invalid_argument("pipeline payload must contain whole RMSNorm rows");
    if (maximum > (1ULL << 30)) throw std::invalid_argument("benchmark payload exceeds 1 GiB");
    const auto deployment = product::read_deployment_file(o.config);
    if (deployment.ranks.size() != 2)
        throw std::invalid_argument("collective benchmark requires two ranks");
    dist::ExecutionTopology topology;
    topology.plan        = 1;
    topology.output_rank = 0;
    topology.ranks.resize(2);
    for (const auto& r : deployment.ranks)
        topology.ranks[r.rank] = {r.rank, r.node, 0, r.device, {r.rank, 2}, {r.rank, 2}, {}, {}};
    topology.groups = {{0,
                        {0, 1},
                        (o.operation == "peer" || o.operation == "send")
                            ? dist::GroupUsage::PeerExchange
                            : dist::GroupUsage::General}};
    topology.roles  = {{"tp", 0}, {"ep", 0}};
    const std::array<std::string, 4> args{"--worker", o.backend, std::to_string(maximum),
                                          o.operation};
    dist::ExecutionGroup group(
        deployment, topology, {},
        [](auto rank, auto line) { std::cerr << "rank=" << rank << " " << line << '\n'; }, args);
    dist::prepare_communication(group, std::chrono::seconds(90));
    (void)group.receive_all(wire::Kind::Ready, std::chrono::seconds(90));
    std::cout << "backend,operation,bytes,chain,mode,workload,work_rows,p50_us,p95_us,min_us,"
                 "host_p50_us,payload_GBps\n";
    for (auto bytes : o.bytes) {
        wire::Encoder request;
        request.string(o.operation);
        request.u64(bytes);
        request.u32(o.chain);
        request.u32(o.samples);
        request.boolean(o.mode == "graph");
        request.string(o.workload);
        request.u32(o.width);
        request.u32(o.work_rows);
        const auto message = std::make_shared<wire::Message>(
            wire::Message{wire::Kind::Prepare, group.epoch(), std::move(request).take()});
        const std::array sends{dist::OutgoingMessage{0, message},
                               dist::OutgoingMessage{1, message}};
        if (!group.try_send(sends))
            throw std::runtime_error("benchmark control capacity exhausted");
        std::array<std::vector<double>, 2> gpu, host;
        for (auto& result : group.receive_all(wire::Kind::Ready, std::chrono::seconds(90))) {
            wire::Decoder in(result.message.payload);
            if (in.u32() != o.samples) throw std::runtime_error("benchmark sample count differs");
            for (unsigned i = 0; i < o.samples; ++i) {
                gpu[result.peer].push_back(in.f64());
                host[result.peer].push_back(in.f64());
            }
            in.finish();
        }
        std::vector<double> times(o.samples), submissions(o.samples);
        for (unsigned i = 0; i < o.samples; ++i) {
            times[i]       = std::max(gpu[0][i], gpu[1][i]);
            submissions[i] = std::max(host[0][i], host[1][i]);
        }
        const auto median       = percentile(times, .5);
        const auto payload_rate = o.workload == "compute" ? 0.0 : bytes / median / 1000.0;
        std::cout << o.backend << ',' << o.operation << ',' << bytes << ',' << o.chain << ','
                  << o.mode << ',' << o.workload << ',' << o.work_rows << ',' << std::fixed
                  << std::setprecision(3) << median << ',' << percentile(times, .95) << ','
                  << *std::min_element(times.begin(), times.end()) << ','
                  << percentile(submissions, .5) << ',' << payload_rate << '\n';
        std::cout.flush();
    }
    group.shutdown();
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 5 && std::string_view(argv[1]) == "--worker")
            return worker(policy(argv[2]), number(argv[3]), argv[4]);
        return coordinator(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
