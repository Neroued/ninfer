// Public-Op benchmark for variable-width DFlash2 pair and context-K RMSNorm+RoPE profiles.
#include "ninfer/ops/rmsnorm_rope.h"

#include "ninfer_bench_common.h"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;

namespace {

constexpr int kHeadDim    = 128;
constexpr int kQueryHeads = 32;
constexpr int kKeyHeads   = 8;

enum class Form : std::uint8_t { Pair, Single };
enum class Execution : std::uint8_t { Eager, Graph };

struct Options {
    std::vector<int> widths{2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    Form form           = Form::Pair;
    Execution execution = Execution::Graph;
    std::vector<int> batches{1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int> tokens{1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 128, 1024, 2048};
    int warmup   = 20;
    int repeat   = 200;
    bool profile = false;
};

[[noreturn]] void usage(const char* message) {
    std::fprintf(stderr,
                 "error: %s\n"
                 "usage: ninfer_rmsnorm_rope_bench --form pair|single "
                 "[--widths W,...] [--batches B,...] [--tokens T,...] [--execution eager|graph] "
                 "[--warmup N] [--repeat N] [--profile]\n",
                 message);
    std::exit(2);
}

int parse_i32(std::string_view text, int minimum, int maximum, const char* flag) {
    const std::string value(text);
    char* end         = nullptr;
    const long parsed = std::strtol(value.c_str(), &end, 10);
    if (end == value.c_str() || *end != '\0' || parsed < minimum || parsed > maximum) {
        usage(flag);
    }
    return static_cast<int>(parsed);
}

std::vector<int> parse_list(const char* text, int minimum, int maximum, const char* flag) {
    std::vector<int> values;
    std::string_view remaining(text);
    while (!remaining.empty()) {
        const std::size_t comma     = remaining.find(',');
        const std::string_view item = remaining.substr(0, comma);
        if (item.empty()) { usage(flag); }
        values.push_back(parse_i32(item, minimum, maximum, flag));
        if (comma == std::string_view::npos) { break; }
        remaining.remove_prefix(comma + 1);
    }
    if (values.empty()) { usage(flag); }
    return values;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto next = [&](const char* message) -> const char* {
            if (++index == argc) { usage(message); }
            return argv[index];
        };
        if (argument == "--form") {
            const std::string_view value(next("--form requires a value"));
            if (value == "pair")
                options.form = Form::Pair;
            else if (value == "single")
                options.form = Form::Single;
            else
                usage("--form expects pair or single");
        } else if (argument == "--widths") {
            options.widths = parse_list(next("--widths requires a value"), 2, 16, "--widths");
        } else if (argument == "--batches") {
            options.batches = parse_list(next("--batches requires a value"), 1, 8, "--batches");
        } else if (argument == "--tokens") {
            options.tokens = parse_list(next("--tokens requires a value"), 1, 2048, "--tokens");
        } else if (argument == "--execution") {
            const std::string_view value(next("--execution requires a value"));
            if (value == "eager")
                options.execution = Execution::Eager;
            else if (value == "graph")
                options.execution = Execution::Graph;
            else
                usage("--execution expects eager or graph");
        } else if (argument == "--warmup") {
            options.warmup = parse_i32(next("--warmup requires a value"), 0, 10000, "--warmup");
        } else if (argument == "--repeat") {
            options.repeat = parse_i32(next("--repeat requires a value"), 1, 10000, "--repeat");
        } else if (argument == "--profile") {
            options.profile = true;
        } else {
            usage("unknown argument");
        }
    }
    if (options.profile && ((options.form == Form::Pair &&
                             (options.batches.size() != 1 || options.widths.size() != 1)) ||
                            (options.form == Form::Single && options.tokens.size() != 1))) {
        usage("--profile requires one selected extent");
    }
    return options;
}

std::vector<std::int32_t> host_positions(int tokens) {
    std::vector<std::int32_t> values(static_cast<std::size_t>(tokens));
    for (int token = 0; token < tokens; ++token)
        values[static_cast<std::size_t>(token)] = 262'000 + token;
    return values;
}

bench::Result measure(const Options& options, const bench::launch_fn& launch, double bytes,
                      cudaStream_t stream) {
    if (options.execution == Execution::Eager) {
        return bench::bench_loop(launch, bytes, options.warmup, options.repeat, 100);
    }
    constexpr int repetitions = 32;
    bench::TimedGraph graph;
    graph.capture(stream, [&](cudaStream_t) {
        for (int i = 0; i < repetitions; ++i) launch(stream);
    });
    const auto timing = bench::measure_graph(graph, stream, options.warmup, options.repeat);
    bench::Result result;
    result.n_runs      = options.repeat;
    result.inner_iters = repetitions;
    result.median_us   = timing.median_us / repetitions;
    result.min_us      = timing.min_us / repetitions;
    result.p95_us      = timing.p95_us / repetitions;
    result.gbs         = bytes / result.median_us / 1.0e3;
    return result;
}

void run_pair(const Options& options, int width, int batch, cudaStream_t stream) {
    const int tokens          = width * batch;
    const auto positions_host = host_positions(tokens);
    DeviceBuffer positions(positions_host.size() * sizeof(std::int32_t));
    positions.copy_from_host(positions_host.data(), positions.bytes);
    DeviceBuffer q = bench::make_bf16(static_cast<std::size_t>(kHeadDim) * kQueryHeads * tokens);
    DeviceBuffer k = bench::make_bf16(static_cast<std::size_t>(kHeadDim) * kKeyHeads * tokens);
    DeviceBuffer q_weight = bench::make_bf16(kHeadDim);
    DeviceBuffer k_weight = bench::make_bf16(kHeadDim);
    Tensor t_positions(positions.p, DType::I32, {width, batch});
    Tensor t_q(q.p, DType::BF16, {kHeadDim, kQueryHeads, width, batch});
    Tensor t_k(k.p, DType::BF16, {kHeadDim, kKeyHeads, width, batch});
    Tensor t_q_weight(q_weight.p, DType::BF16, {kHeadDim});
    Tensor t_k_weight(k_weight.p, DType::BF16, {kHeadDim});
    const auto launch = [&](cudaStream_t launch_stream) {
        ops::rmsnorm_rope(t_positions, t_q_weight, t_k_weight, t_q, t_k, launch_stream);
    };
    if (options.profile) {
        for (int index = 0; index < options.warmup; ++index) launch(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaProfilerStart());
        launch(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaProfilerStop());
        return;
    }
    const double bytes =
        2.0 * static_cast<double>(kHeadDim) * (kQueryHeads + kKeyHeads) * tokens * 2.0;
    const bench::Result timing = measure(options, launch, bytes, stream);
    std::printf("form=pair W=%d B=%d T=%d heads_per_cta=8 execution=%s median=%.3f us min=%.3f us "
                "p95=%.3f us useful=%.1f GB/s graph_repetitions=%d cache=warm\n",
                width, batch, tokens, options.execution == Execution::Graph ? "graph" : "eager",
                timing.median_us, timing.min_us, timing.p95_us, timing.gbs,
                options.execution == Execution::Graph ? 32 : 1);
}

void run_single(const Options& options, int tokens, cudaStream_t stream) {
    const auto positions_host = host_positions(tokens);
    DeviceBuffer positions(positions_host.size() * sizeof(std::int32_t));
    positions.copy_from_host(positions_host.data(), positions.bytes);
    DeviceBuffer x      = bench::make_bf16(static_cast<std::size_t>(kHeadDim) * kKeyHeads * tokens);
    DeviceBuffer weight = bench::make_bf16(kHeadDim);
    Tensor t_positions(positions.p, DType::I32, {tokens});
    Tensor t_x(x.p, DType::BF16, {kHeadDim, kKeyHeads, tokens});
    Tensor t_weight(weight.p, DType::BF16, {kHeadDim});
    const auto launch = [&](cudaStream_t launch_stream) {
        ops::rmsnorm_rope(t_positions, t_weight, t_x, launch_stream);
    };
    if (options.profile) {
        for (int index = 0; index < options.warmup; ++index) launch(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaProfilerStart());
        launch(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaProfilerStop());
        return;
    }
    const double bytes         = 2.0 * static_cast<double>(kHeadDim) * kKeyHeads * tokens * 2.0;
    const bench::Result timing = measure(options, launch, bytes, stream);
    std::printf("form=single T=%d heads_per_cta=8 execution=%s median=%.3f us min=%.3f us "
                "p95=%.3f us useful=%.1f GB/s graph_repetitions=%d cache=warm\n",
                tokens, options.execution == Execution::Graph ? "graph" : "eager", timing.median_us,
                timing.min_us, timing.p95_us, timing.gbs,
                options.execution == Execution::Graph ? 32 : 1);
}

} // namespace

int main(int argc, char** argv) {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
            std::printf("SKIP: no usable CUDA device\n");
            return 0;
        }
        Options options     = parse_options(argc, argv);
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        if (options.form == Form::Pair) {
            for (int width : options.widths)
                for (int batch : options.batches) run_pair(options, width, batch, stream);
        } else {
            for (int tokens : options.tokens) run_single(options, tokens, stream);
        }
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_rmsnorm_rope_bench: %s\n", error.what());
        return 1;
    }
}
