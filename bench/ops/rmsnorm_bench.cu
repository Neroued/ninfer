// Measures public RMSNorm and GatedRMSNorm calls with represented BF16 inputs.
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/gated_rmsnorm.h"
#include "core/device.h"
#include "ninfer_bench_common.h"

#include <cuda_profiler_api.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;

namespace {
struct Options {
    int warmup = 10, repeat = 100, graph_calls = 1;
    std::vector<int> tokens{1, 2, 8, 16, 32, 64, 96, 128, 256, 1024, 2048};
    std::string kind = "dflash2_hidden", execution = "graph", cache = "cold", csv;
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
            std::puts("usage: ninfer_rmsnorm_bench [--kind dflash2_hidden|hidden27|target_q27|"
                      "target_k27|dflash_hidden|dflash_q|dflash_k|target_hidden35|target_q35|"
                      "target_k35|gated35|gated27] [--tokens T,...] [--execution eager|graph] "
                      "[--cache cold|warm] [--graph-calls 1..64] [--warmup N] [--repeat N] "
                      "[--csv-out PATH] [--profile]");
            std::exit(0);
        }
        if (++i >= argc) throw std::invalid_argument("missing option value");
        const std::string value = argv[i];
        if (arg == "--kind")
            o.kind = value;
        else if (arg == "--tokens")
            o.tokens = list(value, 65536);
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
    if ((o.execution != "eager" && o.execution != "graph") ||
        (o.cache != "cold" && o.cache != "warm"))
        throw std::invalid_argument("invalid workload or execution mode");
    if (o.graph_calls > 1 && (o.execution != "graph" || o.profile))
        throw std::invalid_argument("bundles require Graph and no profiling");
    if (o.profile && o.tokens.size() != 1) throw std::invalid_argument("profile one shape");
    return o;
}

struct Shape {
    const char* name;
    int d, heads;
    bool offset, gated;
};

constexpr Shape shapes[]{
    {"dflash2_hidden", 5120, 1, false, false}, {"hidden27", 5120, 1, true, false},
    {"target_q27", 256, 24, true, false},      {"target_k27", 256, 4, true, false},
    {"dflash_hidden", 2048, 1, false, false},  {"dflash_q", 128, 32, false, false},
    {"dflash_k", 128, 8, false, false},        {"target_hidden35", 2048, 1, true, false},
    {"target_q35", 256, 16, true, false},      {"target_k35", 256, 2, true, false},
    {"gated35", 128, 32, false, true},         {"gated27", 128, 48, false, true}};

const Shape& find_shape(const std::string& name) {
    for (const auto& shape : shapes)
        if (name == shape.name) return shape;
    throw std::invalid_argument("unknown norm kind: " + name);
}

DeviceBuffer varied(std::size_t n, unsigned seed) {
    std::vector<__nv_bfloat16> values(n);
    for (auto& v : values) {
        seed = seed * 1664525U + 1013904223U;
        v    = __float2bfloat16(static_cast<float>(seed >> 8) * (2.0f / 16777216.0f) - 1.0f);
    }
    DeviceBuffer buffer(n * 2);
    CUDA_CHECK(cudaMemcpy(buffer.p, values.data(), n * 2, cudaMemcpyHostToDevice));
    return buffer;
}

struct Case {
    Shape shape;
    DeviceBuffer source, weight, gate, output;
    Tensor x, w, z, y;

    Case(const Shape& s, int t)
        : shape(s), source(varied(std::size_t(s.d) * s.heads * t, 1234)), weight(varied(s.d, 9876)),
          gate(s.gated ? varied(std::size_t(s.d) * s.heads * t, 321) : DeviceBuffer()),
          output(std::size_t(s.d) * s.heads * t * 2), x(source.p, DType::BF16, {s.d, s.heads, t}),
          w(weight.p, DType::BF16, {s.d}), z(gate.p, DType::BF16, {s.d, s.heads, t}),
          y(output.p, DType::BF16, {s.d, s.heads, t}) {}

    void launch(cudaStream_t stream) {
        if (shape.gated)
            ops::gated_rmsnorm(x, w, z, 1.e-6f, y, stream);
        else
            ops::rmsnorm(x, w, 1.e-6f, shape.offset, y, stream);
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
            csv << "kind,D,heads,T,execution,cache,graph_nodes,graph_calls,workspace_bytes,logical_"
                   "bytes,median_us,min_us,p95_us\n";
        }
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        std::printf("GPU=%s CUDART=%d\n", prop.name, CUDART_VERSION);
        const Shape& shape = find_shape(o.kind);
        for (const int t : o.tokens) {
            Case data(shape, t);
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
                time =
                    o.cache == "cold"
                        ? bench::measure_cold_graph(graph, flush, device.stream, o.warmup, o.repeat)
                        : bench::measure_graph(graph, device.stream, o.warmup, o.repeat);
            else
                time = o.cache == "cold"
                           ? bench::measure_cold_launch(launch, flush, device.stream, o.warmup,
                                                        o.repeat)
                           : bench::measure_launch(launch, device.stream, o.warmup, o.repeat);
            time.median_us /= o.graph_calls;
            time.min_us /= o.graph_calls;
            time.p95_us /= o.graph_calls;
            const std::size_t bytes =
                std::size_t(shape.d) * 2 * ((shape.gated ? 3 : 2) * shape.heads * t + 1);
            const auto nodes = o.execution == "graph" ? graph.nodes() : 0;
            std::printf("%s T=%d D=%d H=%d execution=%s cache=%s nodes=%zu calls=%d scratch=0 "
                        "median=%.3f min=%.3f p95=%.3f us\n",
                        shape.name, t, shape.d, shape.heads, o.execution.c_str(), o.cache.c_str(),
                        nodes, o.graph_calls, time.median_us, time.min_us, time.p95_us);
            if (csv)
                csv << shape.name << ',' << shape.d << ',' << shape.heads << ',' << t << ','
                    << o.execution << ',' << o.cache << ',' << nodes << ',' << o.graph_calls
                    << ",0," << bytes << ',' << time.median_us << ',' << time.min_us << ','
                    << time.p95_us << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "rmsnorm_bench: %s\n", error.what());
        return 1;
    }
}
