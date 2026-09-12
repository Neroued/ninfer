// Measures five public feature-slice captures into a strided lane-owned pool.
#include "ninfer/ops/scatter.h"
#include "core/device.h"
#include "ninfer_bench_common.h"

#include <cuda_profiler_api.h>
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;

namespace {
struct Options {
    int rows = 5120, warmup = 10, repeat = 100, graph_calls = 1;
    std::vector<int> widths{1, 2, 3, 4, 5, 8, 9, 15, 16}, batches{1, 8};
    std::string counts = "ragged", execution = "graph", cache = "cold", csv;
    bool profile = false;
};

int integer(const std::string& text, int lower, int upper) {
    std::size_t end = 0;
    const int value = std::stoi(text, &end);
    if (end != text.size() || value < lower || value > upper)
        throw std::invalid_argument("integer out of range");
    return value;
}

std::vector<int> list(const std::string& text, int upper) {
    std::vector<int> result;
    std::size_t start = 0;
    while (start < text.size()) {
        const auto end = text.find(',', start);
        result.push_back(integer(text.substr(start, end - start), 1, upper));
        if (end == std::string::npos) break;
        start = end + 1;
    }
    if (result.empty()) throw std::invalid_argument("empty extent list");
    return result;
}

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--profile") {
            o.profile = true;
            continue;
        }
        if (arg == "--help") {
            std::puts("usage: ninfer_scatter_bf16_batch_bench [--rows D] [--widths W,...] "
                      "[--batches B,...] [--counts full|one|ragged|zero] [--execution eager|graph] "
                      "[--cache cold|warm] [--graph-calls 1..64] [--warmup N] [--repeat N] "
                      "[--csv-out PATH] [--profile]");
            std::exit(0);
        }
        if (++i >= argc) throw std::invalid_argument("missing option value");
        const std::string value = argv[i];
        if (arg == "--rows")
            o.rows = integer(value, 8, 25600);
        else if (arg == "--widths")
            o.widths = list(value, 4096);
        else if (arg == "--batches")
            o.batches = list(value, 8);
        else if (arg == "--counts")
            o.counts = value;
        else if (arg == "--execution")
            o.execution = value;
        else if (arg == "--cache")
            o.cache = value;
        else if (arg == "--graph-calls")
            o.graph_calls = integer(value, 1, 64);
        else if (arg == "--warmup")
            o.warmup = integer(value, 0, 10000);
        else if (arg == "--repeat")
            o.repeat = integer(value, 1, 10000);
        else if (arg == "--csv-out")
            o.csv = value;
        else
            throw std::invalid_argument("unknown option: " + arg);
    }
    if (o.rows % 8 ||
        (o.counts != "full" && o.counts != "one" && o.counts != "ragged" && o.counts != "zero") ||
        (o.execution != "eager" && o.execution != "graph") ||
        (o.cache != "cold" && o.cache != "warm"))
        throw std::invalid_argument("invalid workload or execution mode");
    if (o.graph_calls > 1 && (o.execution != "graph" || o.profile))
        throw std::invalid_argument("bundles require Graph and no profiling");
    if (o.profile && (o.widths.size() != 1 || o.batches.size() != 1))
        throw std::invalid_argument("profile one shape");
    return o;
}

__global__ void initialize_bits(unsigned* values, std::size_t n, unsigned salt) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        values[i] = (static_cast<unsigned>(i) * 0x9e3779b9U) ^
                    (static_cast<unsigned>(i >> 13) * 0x85ebca6bU) ^ salt;
}

struct Case {
    DeviceBuffer source, lanes, valid, destination;
    Tensor l, v;
    std::array<Tensor, 5> x, y;
    std::size_t live = 0;

    Case(int d, int w, int b, const std::string& mode)
        : source(std::size_t(5) * d * w * b * 2), lanes(b * 4), valid(b * 4),
          destination(std::size_t(5) * d * w * 8 * 2), l(lanes.p, DType::I32, {b}),
          v(valid.p, DType::I32, {b}) {
        constexpr int order[]{7, 0, 4, 2, 6, 1, 5, 3};
        std::vector<int> hl(b), hv(b);
        for (int i = 0; i < b; ++i) {
            hl[i] = order[i];
            hv[i] = mode == "zero"   ? 0
                    : mode == "full" ? w
                    : mode == "one"  ? 1
                    : i % 4 == 0     ? w
                    : i % 4 == 1     ? w - 1
                    : i % 4 == 2     ? 1
                                     : 0;
            live += hv[i];
        }
        CUDA_CHECK(cudaMemcpy(lanes.p, hl.data(), b * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(valid.p, hv.data(), b * 4, cudaMemcpyHostToDevice));
        initialize_bits<<<(source.bytes / 4 + 255) / 256, 256>>>(static_cast<unsigned*>(source.p),
                                                                 source.bytes / 4, 12345);
        initialize_bits<<<(destination.bytes / 4 + 255) / 256, 256>>>(
            static_cast<unsigned*>(destination.p), destination.bytes / 4, 91357);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        for (int slice = 0; slice < 5; ++slice) {
            x[slice] = Tensor(static_cast<char*>(source.p) + std::size_t(slice) * d * w * b * 2,
                              DType::BF16, {d, w, b});
            y[slice] =
                Tensor(static_cast<char*>(destination.p) + slice * d * 2, DType::BF16, {d, w, 8});
            y[slice].nb[1] = std::int64_t(5) * d * 2;
            y[slice].nb[2] = std::int64_t(5) * d * w * 2;
            y[slice].nb[3] = std::int64_t(5) * d * w * 8 * 2;
        }
    }

    void launch(cudaStream_t stream) {
        for (int slice = 0; slice < 5; ++slice)
            ops::scatter_bf16_batch(x[slice], l, v, y[slice], stream);
    }
};
} // namespace

int main(int argc, char** argv) {
    try {
        const Options o = parse(argc, argv);
        int devices     = 0;
        CUDA_CHECK(cudaGetDeviceCount(&devices));
        if (!devices) return 77;
        DeviceContext device;
        DeviceBuffer flush(std::size_t{256} << 20);
        std::ofstream csv;
        if (!o.csv.empty()) {
            csv.open(o.csv);
            if (!csv) throw std::runtime_error("cannot open CSV output");
            csv << "D,W,B,counts,live_columns,execution,cache,graph_nodes,graph_calls,workspace_"
                   "bytes,useful_bytes,median_us,min_us,p95_us\n";
        }
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        std::printf("GPU=%s CUDART=%d\n", prop.name, CUDART_VERSION);
        for (const int w : o.widths)
            for (const int b : o.batches) {
                Case data(o.rows, w, b, o.counts);
                bench::TimedGraph graph;
                if (o.execution == "graph") {
                    data.launch(device.stream);
                    CUDA_CHECK(cudaStreamSynchronize(device.stream));
                    graph.capture(device.stream, [&](cudaStream_t stream) {
                        for (int call = 0; call < o.graph_calls; ++call) data.launch(stream);
                    });
                }
                const auto launch = [&](cudaStream_t stream) { data.launch(stream); };
                if (o.profile) {
                    for (int i = 0; i < o.warmup; ++i) {
                        if (o.execution == "graph")
                            graph.launch(device.stream);
                        else
                            data.launch(device.stream);
                    }
                    if (o.cache == "cold") bench::flush_l2(flush, device.stream);
                    CUDA_CHECK(cudaStreamSynchronize(device.stream));
                    CUDA_CHECK(cudaProfilerStart());
                    if (o.execution == "graph")
                        graph.launch(device.stream);
                    else
                        data.launch(device.stream);
                    CUDA_CHECK(cudaStreamSynchronize(device.stream));
                    CUDA_CHECK(cudaProfilerStop());
                    return 0;
                }
                bench::ColdTiming time;
                if (o.execution == "graph")
                    time = o.cache == "cold"
                               ? bench::measure_cold_graph(graph, flush, device.stream, o.warmup,
                                                           o.repeat)
                               : bench::measure_graph(graph, device.stream, o.warmup, o.repeat);
                else
                    time = o.cache == "cold"
                               ? bench::measure_cold_launch(launch, flush, device.stream, o.warmup,
                                                            o.repeat)
                               : bench::measure_launch(launch, device.stream, o.warmup, o.repeat);
                time.median_us /= o.graph_calls;
                time.min_us /= o.graph_calls;
                time.p95_us /= o.graph_calls;
                const std::size_t bytes = std::size_t(5) * (o.rows * data.live * 4 + b * 8);
                const auto nodes        = o.execution == "graph" ? graph.nodes() : 0;
                std::printf(
                    "D=%d W=%d B=%d counts=%s live=%zu execution=%s cache=%s nodes=%zu "
                    "capture_calls=%d slices=5 scratch=0 median=%.3f min=%.3f p95=%.3f us\n",
                    o.rows, w, b, o.counts.c_str(), data.live, o.execution.c_str(), o.cache.c_str(),
                    nodes, o.graph_calls, time.median_us, time.min_us, time.p95_us);
                if (csv)
                    csv << o.rows << ',' << w << ',' << b << ',' << o.counts << ',' << data.live
                        << ',' << o.execution << ',' << o.cache << ',' << nodes << ','
                        << o.graph_calls << ",0," << bytes << ',' << time.median_us << ','
                        << time.min_us << ',' << time.p95_us << '\n';
            }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "scatter_bf16_batch_bench: %s\n", error.what());
        return 1;
    }
}
