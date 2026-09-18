// Cold-cache public Op benchmark for the registered Q4 LinearSwiGLU profile.

#include "core/weight.h"
#include "ninfer/ops/linear_swiglu.h"

#include "core/device.h"
#include "ninfer_bench_common.h"
#include "quantized_weight.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;

namespace {

constexpr std::int32_t kGateUpRows = 34816;
constexpr std::int32_t kOutputRows = 17408;
constexpr std::int32_t kHidden     = 5120;
constexpr std::size_t kFlushBytes  = 256ULL << 20;

struct Options {
    std::vector<std::int32_t> tokens{1, 2, 4, 8, 16, 24, 32, 48};
    int warmup   = 5;
    int repeat   = 30;
    bool profile = false;
    bool graph   = false;
    std::string csv_out;
};

struct Result {
    std::int32_t tokens               = 0;
    const char* execution             = "eager";
    std::size_t graph_nodes           = 0;
    double median_us                  = 0.0;
    double min_us                     = 0.0;
    double p95_us                     = 0.0;
    double weight_bytes               = 0.0;
    double logical_bytes              = 0.0;
    double projection_flops           = 0.0;
    double logical_gbs                = 0.0;
    double useful_tflops              = 0.0;
    std::size_t workspace_sweep_bytes = 0;
    std::size_t workspace_exact_bytes = 0;
};

std::vector<std::int32_t> parse_tokens(std::string_view raw) {
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
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto next = [&](const char* label) -> std::string_view {
            if (++index >= argc) { throw std::invalid_argument(std::string("missing ") + label); }
            return argv[index];
        };
        if (argument == "--t-sweep") {
            options.tokens = parse_tokens(next("--t-sweep value"));
        } else if (argument == "--warmup") {
            options.warmup = std::stoi(std::string(next("--warmup value")));
        } else if (argument == "--repeat") {
            options.repeat = std::stoi(std::string(next("--repeat value")));
        } else if (argument == "--execution") {
            const std::string_view mode = next("--execution value");
            if (mode != "eager" && mode != "graph") {
                throw std::invalid_argument("--execution must be eager or graph");
            }
            options.graph = mode == "graph";
        } else if (argument == "--csv-out") {
            options.csv_out = std::string(next("--csv-out value"));
        } else if (argument == "--profile") {
            options.profile = true;
        } else if (argument == "--help" || argument == "-h") {
            std::printf(
                "Usage: %s [--t-sweep 1,2,...] [--warmup N] [--repeat N] [--execution eager|graph]\n"
                "          [--csv-out PATH] [--profile]\n"
                "  --execution graph captures ONE complete public ops::linear_swiglu call per\n"
                "  graph; the reported graph_nodes is the captured count, not an assumption.\n"
                "  --profile runs a single eager or single graph launch for external profilers.\n",
                argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(argument));
        }
    }
    if (options.warmup < 0 || options.repeat <= 0) {
        throw std::invalid_argument("--warmup must be nonnegative and --repeat positive");
    }
    if (options.profile && options.tokens.size() != 1) {
        throw std::invalid_argument("--profile requires exactly one T");
    }
    if (options.profile && !options.csv_out.empty()) {
        throw std::invalid_argument("--csv-out is incompatible with --profile");
    }
    return options;
}

void write_csv(const std::string& path, const std::vector<Result>& results) {
    std::FILE* file = std::fopen(path.c_str(), "wb");
    if (file == nullptr) { throw std::runtime_error("cannot open csv output: " + path); }
    std::fprintf(file,
                 "t,execution,graph_nodes,median_us,min_us,p95_us,weight_bytes,logical_bytes,"
                 "projection_flops,effective_gbs,dram_spec_pct,useful_tflops,"
                 "workspace_sweep_bytes,workspace_exact_bytes\n");
    for (const Result& result : results) {
        const double seconds = result.median_us * 1.0e-6;
        std::fprintf(file,
                     "%d,%s,%zu,%.3f,%.3f,%.3f,%.0f,%.0f,%.0f,%.4f,%.4f,%.4f,%zu,%zu\n",
                     result.tokens, result.execution, result.graph_nodes, result.median_us,
                     result.min_us, result.p95_us, result.weight_bytes, result.logical_bytes,
                     result.projection_flops, result.logical_bytes / seconds / 1.0e9,
                     result.logical_bytes / seconds / 1.0e9 / 1792.0 * 100.0,
                     result.projection_flops / seconds / 1.0e12, result.workspace_sweep_bytes,
                     result.workspace_exact_bytes);
    }
    std::fclose(file);
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const auto [min_it, max_it] =
            std::minmax_element(options.tokens.begin(), options.tokens.end());
        const std::int32_t min_t = *min_it;
        const std::int32_t max_t = *max_it;

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        DeviceBuffer flush(kFlushBytes);
        DeviceBuffer input = bench::make_bf16(static_cast<std::size_t>(kHidden) * max_t);
        DeviceBuffer output(static_cast<std::size_t>(kOutputRows) * max_t * sizeof(std::uint16_t));
        bench::PackedQuantizedWeight packed = bench::make_row_split_weight(
            QType::Q4_G64_FP16, kGateUpRows, kHidden, kHidden, {0x31, 0xa5, 0x3c00});
        const std::size_t workspace_capacity = ops::linear_swiglu_workspace_capacity_bytes(
            QType::Q4_G64_FP16, kGateUpRows, kHidden, min_t, max_t);
        WorkspaceArena workspace(std::max<std::size_t>(workspace_capacity, 256));

        const auto launch = [&](std::int32_t tokens, cudaStream_t launch_stream) {
            Tensor x(input.p, DType::BF16, {kHidden, tokens});
            Tensor out(output.p, DType::BF16, {kOutputRows, tokens});
            ops::linear_swiglu(x, packed.weight, out, workspace, launch_stream);
        };

        if (options.profile) {
            bench::TimedGraph graph;
            if (options.graph) {
                graph.capture(stream, [&](cudaStream_t capture_stream) {
                    launch(options.tokens.front(), capture_stream);
                });
                graph.launch(stream);
            } else {
                launch(options.tokens.front(), stream);
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
            std::printf("PROFILE linear_swiglu Q4 T=%d execution=%s graph_nodes=%zu "
                        "workspace_sweep=%zu\n",
                        options.tokens.front(), options.graph ? "graph" : "eager", graph.nodes(),
                        workspace_capacity);
            CUDA_CHECK(cudaStreamDestroy(stream));
            return 0;
        }

        std::printf("# execution=%s warmup=%d repeat=%d flush_bytes=%zu "
                    "workspace_sweep_bytes=%zu\n",
                    options.graph ? "graph" : "eager", options.warmup, options.repeat, kFlushBytes,
                    workspace_capacity);
        std::vector<Result> results;
        results.reserve(options.tokens.size());
        for (const std::int32_t tokens : options.tokens) {
            bench::TimedGraph graph;
            if (options.graph) {
                graph.capture(stream, [&](cudaStream_t capture_stream) {
                    launch(tokens, capture_stream);
                });
            }
            const bench::ColdTiming timing =
                options.graph
                    ? bench::measure_cold_graph(graph, flush, stream, options.warmup, options.repeat)
                    : bench::measure_cold_launch(
                          [&](cudaStream_t launch_stream) { launch(tokens, launch_stream); }, flush,
                          stream, options.warmup, options.repeat);
            const double seconds = timing.median_us * 1.0e-6;
            const double flops   = 2.0 * static_cast<double>(kGateUpRows) * kHidden * tokens;
            const double bytes   = static_cast<double>(packed.model_weight_bytes()) +
                                 2.0 * static_cast<double>(kHidden + kOutputRows) * tokens;
            const std::size_t workspace_exact = ops::linear_swiglu_workspace_capacity_bytes(
                QType::Q4_G64_FP16, kGateUpRows, kHidden, tokens, tokens);
            std::printf("T=%-4d median=%9.3f us min=%9.3f p95=%9.3f %7.1f GB/s %7.2f TFLOP/s "
                        "workspace_sweep=%zu workspace_exact=%zu nodes=%zu\n",
                        tokens, timing.median_us, timing.min_us, timing.p95_us, bytes / seconds / 1.0e9,
                        flops / seconds / 1.0e12, workspace_capacity, workspace_exact,
                        graph.nodes());
            results.push_back({tokens,
                               options.graph ? "graph" : "eager",
                               graph.nodes(),
                               timing.median_us,
                               timing.min_us,
                               timing.p95_us,
                               static_cast<double>(packed.model_weight_bytes()),
                               bytes,
                               flops,
                               bytes / seconds / 1.0e9,
                               flops / seconds / 1.0e12,
                               workspace_capacity,
                               workspace_exact});
        }

        if (!options.csv_out.empty()) { write_csv(options.csv_out, results); }
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_q4_linear_swiglu_bench: %s\n", error.what());
        return 1;
    }
}
