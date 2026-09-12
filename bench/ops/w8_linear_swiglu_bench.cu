// Complete public-Op benchmark for the registered W8 LinearSwiGLU profiles.

#include "ninfer/ops/linear_swiglu.h"

#include "core/device.h"
#include "ninfer_bench_common.h"
#include "quantized_weight.cuh"

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;

namespace {

constexpr std::size_t kFlushBytes = 256ULL << 20;

struct Problem {
    const char* name;
    std::int32_t gate_up_rows;
    std::int32_t output_rows;
    std::int32_t hidden;
};

constexpr Problem kCompanion{"companion", 12288, 6144, 2048};
constexpr Problem kDFlash2{"dflash2", 34816, 17408, 5120};

struct Options {
    std::vector<std::int32_t> t_sweep;
    int warmup = 5;
    int repeat = 30;
    std::string csv_out;
    std::string problem = kCompanion.name;
    bool profile        = false;
    bool graph          = false;
};

struct Result {
    std::int32_t t;
    bench::ColdTiming timing;
    std::size_t workspace_bytes;
    std::size_t graph_nodes;
};

std::vector<std::int32_t> parse_t_sweep(std::string_view raw) {
    std::vector<std::int32_t> result;
    std::size_t begin = 0;
    while (begin < raw.size()) {
        const std::size_t end = raw.find(',', begin);
        const std::string token(
            raw.substr(begin, end == std::string_view::npos ? raw.size() - begin : end - begin));
        const long value = std::stol(token);
        if (value <= 0 || value > std::numeric_limits<std::int32_t>::max()) {
            throw std::invalid_argument("--t-sweep values must be positive int32");
        }
        result.push_back(static_cast<std::int32_t>(value));
        if (end == std::string_view::npos) { break; }
        begin = end + 1;
    }
    if (result.empty()) { throw std::invalid_argument("--t-sweep must not be empty"); }
    return result;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        const auto next = [&](const char* label) -> std::string_view {
            if (++i >= argc) { throw std::invalid_argument(std::string("missing ") + label); }
            return argv[i];
        };
        if (arg == "--t-sweep") {
            options.t_sweep = parse_t_sweep(next("--t-sweep value"));
        } else if (arg == "--problem") {
            options.problem = next("--problem value");
        } else if (arg == "--warmup") {
            options.warmup = std::stoi(std::string(next("--warmup value")));
        } else if (arg == "--repeat") {
            options.repeat = std::stoi(std::string(next("--repeat value")));
        } else if (arg == "--csv-out") {
            options.csv_out = next("--csv-out path");
        } else if (arg == "--profile") {
            options.profile = true;
        } else if (arg == "--execution") {
            const auto mode = next("--execution value");
            if (mode != "eager" && mode != "graph")
                throw std::invalid_argument("execution must be eager or graph");
            options.graph = mode == "graph";
        } else if (arg == "--help" || arg == "-h") {
            std::printf("Usage: %s [--problem companion|dflash2] [--t-sweep 1,2,...] "
                        "[--warmup N] [--repeat N] [--csv-out PATH] [--profile] "
                        "[--execution eager|graph]\n",
                        argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(arg));
        }
    }
    if (options.warmup < 0 || options.repeat <= 0) {
        throw std::invalid_argument("--warmup must be nonnegative and --repeat positive");
    }
    if (options.problem != kCompanion.name && options.problem != kDFlash2.name) {
        throw std::invalid_argument("--problem must be companion or dflash2");
    }
    if (options.t_sweep.empty()) {
        if (options.problem == kDFlash2.name) {
            options.t_sweep = {1,  8,  16, 32, 40, 41, 51, 52,  63,  64,
                               65, 80, 81, 88, 89, 96, 97, 128, 129, 1024};
        } else {
            options.t_sweep = {
                1,  2,  3,  4,   5,   6,   7,   8,   9,   10,  11,  12,  13,
                14, 15, 16, 17,  24,  31,  32,  33,  48,  63,  64,  65,  80,
                95, 96, 97, 127, 128, 129, 192, 256, 384, 512, 768, 896, 1024,
            };
        }
    }
    if (options.profile && options.t_sweep.size() != 1) {
        throw std::invalid_argument("--profile requires exactly one T");
    }
    if (options.profile && !options.csv_out.empty())
        throw std::invalid_argument("--profile does not write timing CSV");
    return options;
}

void append(std::vector<Result>& results, const Problem& problem, int t, bench::ColdTiming timing,
            std::size_t weight_bytes, std::size_t workspace_bytes, std::size_t graph_nodes,
            bool graph) {
    const double flops = 2.0 * problem.gate_up_rows * problem.hidden * t;
    const double bytes = static_cast<double>(weight_bytes) +
                         2.0 * (static_cast<double>(problem.hidden) + problem.output_rows) * t;
    std::printf("entry=linear_swiglu problem=%s T=%d execution=%s graph_nodes=%zu workspace=%zu "
                "median=%.3f us min=%.3f us p95=%.3f us logical=%.1f GB/s math=%.2f TFLOP/s\n",
                problem.name, t, graph ? "graph" : "eager", graph_nodes, workspace_bytes,
                timing.median_us, timing.min_us, timing.p95_us, bytes / timing.median_us / 1e3,
                flops / timing.median_us / 1e6);
    results.push_back({t, timing, workspace_bytes, graph_nodes});
}

void write_csv(const Options& options, const std::vector<Result>& results) {
    if (options.csv_out.empty()) return;
    const std::filesystem::path path(options.csv_out);
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot open timing CSV");
    out << "problem,entry,T,execution,workspace_bytes,graph_nodes,median_us,min_us,p95_us,warmup,"
           "repeat,flush_bytes\n";
    for (const auto& r : results) {
        out << options.problem << ",linear_swiglu," << r.t << ','
            << (options.graph ? "graph" : "eager") << ',' << r.workspace_bytes << ',';
        if (r.graph_nodes) out << r.graph_nodes;
        out << ',' << r.timing.median_us << ',' << r.timing.min_us << ',' << r.timing.p95_us << ','
            << options.warmup << ',' << options.repeat << ',' << kFlushBytes << '\n';
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        const Options options  = parse_options(argc, argv);
        const Problem& problem = options.problem == kDFlash2.name ? kDFlash2 : kCompanion;
        DeviceContext context;
        const cudaStream_t stream = context.stream;
        std::printf("# gpu=%s sm=%d%d cuda_runtime=%d cache=cold\n", context.props.name,
                    context.props.major, context.props.minor, CUDART_VERSION);
        const auto [minimum, maximum] =
            std::minmax_element(options.t_sweep.begin(), options.t_sweep.end());
        DeviceBuffer flush(kFlushBytes);
        DeviceBuffer input =
            bench::make_bf16(static_cast<std::size_t>(problem.hidden) * (*maximum));
        DeviceBuffer output(static_cast<std::size_t>(problem.output_rows) * (*maximum) * 2);
        auto packed =
            bench::make_row_split_weight(QType::W8G32_F16S, problem.gate_up_rows, problem.hidden,
                                         problem.hidden, {0x31, 0x00, 0x3c00});
        const auto capacity = ops::linear_swiglu_workspace_capacity_bytes(
            QType::W8G32_F16S, problem.gate_up_rows, problem.hidden, ops::LinearPolicy::A16Only,
            *minimum, *maximum);
        WorkspaceArena workspace(std::max<std::size_t>(capacity, 1));
        std::vector<Result> results;
        for (const auto t : options.t_sweep) {
            Tensor x(input.p, DType::BF16, {problem.hidden, t});
            Tensor out(output.p, DType::BF16, {problem.output_rows, t});
            const auto body = [&](cudaStream_t launch_stream) {
                ops::linear_swiglu(x, packed.weight, out, ops::LinearPolicy::A16Only, workspace,
                                   launch_stream);
            };
            bench::TimedGraph graph;
            if (options.graph) graph.capture(stream, body);
            if (options.profile) {
                const auto launch = [&] {
                    if (options.graph)
                        graph.launch(stream);
                    else
                        body(stream);
                };
                for (int i = 0; i < options.warmup; ++i) {
                    bench::flush_l2(flush, stream);
                    launch();
                }
                bench::flush_l2(flush, stream);
                CUDA_CHECK(cudaStreamSynchronize(stream));
                std::printf(
                    "PROFILE entry=linear_swiglu problem=%s T=%d execution=%s graph_nodes=%zu\n",
                    problem.name, t, options.graph ? "graph" : "eager", graph.nodes());
                CUDA_CHECK(cudaProfilerStart());
                launch();
                CUDA_CHECK(cudaStreamSynchronize(stream));
                CUDA_CHECK(cudaProfilerStop());
                continue;
            }
            const auto timing = options.graph
                                    ? bench::measure_cold_graph(graph, flush, stream,
                                                                options.warmup, options.repeat)
                                    : bench::measure_cold_launch(body, flush, stream,
                                                                 options.warmup, options.repeat);
            const auto exact  = ops::linear_swiglu_workspace_capacity_bytes(
                QType::W8G32_F16S, problem.gate_up_rows, problem.hidden, ops::LinearPolicy::A16Only,
                t, t);
            append(results, problem, t, timing, packed.model_weight_bytes(), exact, graph.nodes(),
                   options.graph);
        }
        write_csv(options, results);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_w8_linear_swiglu_bench: %s\n", error.what());
        return 1;
    }
}
