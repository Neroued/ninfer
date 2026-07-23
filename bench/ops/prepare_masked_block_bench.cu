// Candidate and production qualification benchmark for prepare_masked_block.
// Every timed invocation is one CUDA Graph replay. A 256 MiB untimed write conditions GPU clocks;
// hot mode then primes this Op once, while cold mode measures immediately after the cache flush.
#include "core/device.h"
#include "ninfer/ops/prepare_masked_block.h"
#include "ninfer_bench_common.h"
#include "ops/launcher/prepare_masked_block.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

constexpr std::int32_t kMaskId = 248077;
constexpr std::size_t kFlush   = std::size_t{256} << 20;

enum class RouteChoice {
    Production,
    Warp32,
    Block64,
    Block128,
    Block256,
    All,
};

struct Options {
    std::vector<int> block_sizes{2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    RouteChoice route = RouteChoice::All;
    bool cold         = false;
    bool profile_once = false;
    int warmup        = 20;
    int repeat        = 101;
};

[[noreturn]] void usage(const char* message) {
    std::fprintf(stderr, "error: %s\n", message);
    std::fprintf(stderr, "usage: ninfer_prepare_masked_block_bench [--block-sizes 2,...] "
                         "[--route production|warp32|block64|block128|block256|all] "
                         "[--cold-cache] [--profile-once] [--warmup N] [--repeat N]\n");
    std::exit(2);
}

int parse_int(std::string_view text, int minimum, int maximum, const char* flag) {
    const std::string value(text);
    errno             = 0;
    char* end         = nullptr;
    const long parsed = std::strtol(value.c_str(), &end, 10);
    if (errno != 0 || end == value.c_str() || *end != '\0' || parsed < minimum ||
        parsed > maximum) {
        usage(flag);
    }
    return static_cast<int>(parsed);
}

std::vector<int> parse_list(const char* text, int minimum, int maximum, const char* flag) {
    std::vector<int> result;
    std::string_view remaining(text);
    while (!remaining.empty()) {
        const auto comma            = remaining.find(',');
        const std::string_view item = remaining.substr(0, comma);
        if (item.empty()) usage(flag);
        result.push_back(parse_int(item, minimum, maximum, flag));
        if (comma == std::string_view::npos) break;
        remaining.remove_prefix(comma + 1);
    }
    return result;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        const auto next = [&](const char* message) {
            if (++i == argc) usage(message);
            return argv[i];
        };
        if (arg == "--block-sizes") {
            options.block_sizes =
                parse_list(next("--block-sizes requires a value"), 2, 16, "--block-sizes");
        } else if (arg == "--route") {
            const std::string_view value(next("--route requires a value"));
            if (value == "production") {
                options.route = RouteChoice::Production;
            } else if (value == "warp32") {
                options.route = RouteChoice::Warp32;
            } else if (value == "block64") {
                options.route = RouteChoice::Block64;
            } else if (value == "block128") {
                options.route = RouteChoice::Block128;
            } else if (value == "block256") {
                options.route = RouteChoice::Block256;
            } else if (value == "all") {
                options.route = RouteChoice::All;
            } else {
                usage("--route expects production, warp32, block64, block128, block256, or all");
            }
        } else if (arg == "--cold-cache") {
            options.cold = true;
        } else if (arg == "--profile-once") {
            options.profile_once = true;
        } else if (arg == "--warmup") {
            options.warmup = parse_int(next("--warmup requires a value"), 0, 10000, "--warmup");
        } else if (arg == "--repeat") {
            options.repeat = parse_int(next("--repeat requires a value"), 1, 10000, "--repeat");
        } else {
            usage("unknown argument");
        }
    }
    return options;
}

struct Graph {
    cudaGraph_t graph          = nullptr;
    cudaGraphExec_t executable = nullptr;

    Graph() = default;

    Graph(const Graph&)            = delete;
    Graph& operator=(const Graph&) = delete;

    Graph(Graph&& other) noexcept : graph(other.graph), executable(other.executable) {
        other.graph      = nullptr;
        other.executable = nullptr;
    }

    ~Graph() {
        if (executable != nullptr) cudaGraphExecDestroy(executable);
        if (graph != nullptr) cudaGraphDestroy(graph);
    }
};

template <class Launch>
Graph capture_graph(Launch&& launch, cudaStream_t stream) {
    Graph result;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    launch(stream);
    CUDA_CHECK(cudaStreamEndCapture(stream, &result.graph));
    CUDA_CHECK(cudaGraphInstantiate(&result.executable, result.graph, nullptr, nullptr, 0));
    return result;
}

Result bench_graph(cudaGraphExec_t graph, DBuf& flush, bool cold, cudaStream_t stream, double bytes,
                   int warmup, int repeat) {
    cudaEvent_t begin = nullptr;
    cudaEvent_t end   = nullptr;
    CUDA_CHECK(cudaEventCreate(&begin));
    CUDA_CHECK(cudaEventCreate(&end));
    for (int i = 0; i < warmup; ++i) {
        CUDA_CHECK(cudaMemsetAsync(flush.p, 0xa5, flush.bytes, stream));
        if (!cold) CUDA_CHECK(cudaGraphLaunch(graph, stream));
        CUDA_CHECK(cudaGraphLaunch(graph, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repeat));
    for (int i = 0; i < repeat; ++i) {
        CUDA_CHECK(cudaMemsetAsync(flush.p, 0xa5, flush.bytes, stream));
        if (!cold) CUDA_CHECK(cudaGraphLaunch(graph, stream));
        CUDA_CHECK(cudaEventRecord(begin, stream));
        CUDA_CHECK(cudaGraphLaunch(graph, stream));
        CUDA_CHECK(cudaEventRecord(end, stream));
        CUDA_CHECK(cudaEventSynchronize(end));
        float milliseconds = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, begin, end));
        samples.push_back(static_cast<double>(milliseconds) * 1000.0);
    }
    CUDA_CHECK(cudaEventDestroy(begin));
    CUDA_CHECK(cudaEventDestroy(end));

    std::sort(samples.begin(), samples.end());
    Result result;
    result.n_runs    = repeat;
    result.median_us = samples[samples.size() / 2];
    result.min_us    = samples.front();
    result.p95_us = samples[std::min(samples.size() - 1, samples.size() * std::size_t{95} / 100)];
    result.gbs    = bytes / (result.median_us * 1.0e3);
    return result;
}

ops::detail::PrepareMaskedBlockRoute route_value(RouteChoice route) {
    switch (route) {
    case RouteChoice::Warp32:
        return ops::detail::PrepareMaskedBlockRoute::Warp32;
    case RouteChoice::Block64:
        return ops::detail::PrepareMaskedBlockRoute::Block64;
    case RouteChoice::Block128:
        return ops::detail::PrepareMaskedBlockRoute::Block128;
    case RouteChoice::Block256:
        return ops::detail::PrepareMaskedBlockRoute::Block256;
    case RouteChoice::Production:
    case RouteChoice::All:
        break;
    }
    return ops::detail::PrepareMaskedBlockRoute::Warp32;
}

struct Case {
    int block_size;
    DBuf anchor{sizeof(std::int32_t)};
    DBuf length{sizeof(std::int32_t)};
    DBuf ids;
    DBuf positions;
    Tensor anchor_tensor;
    Tensor length_tensor;
    Tensor ids_tensor;
    Tensor positions_tensor;

    explicit Case(int size)
        : block_size(size), ids(static_cast<std::size_t>(size) * sizeof(std::int32_t)),
          positions(static_cast<std::size_t>(size) * sizeof(std::int32_t)),
          anchor_tensor(anchor.p, DType::I32, {1}), length_tensor(length.p, DType::I32, {1}),
          ids_tensor(ids.p, DType::I32, {size}), positions_tensor(positions.p, DType::I32, {size}) {
        const std::int32_t anchor_value = 42;
        const std::int32_t length_value = 4096;
        CUDA_CHECK(
            cudaMemcpy(anchor.p, &anchor_value, sizeof(anchor_value), cudaMemcpyHostToDevice));
        CUDA_CHECK(
            cudaMemcpy(length.p, &length_value, sizeof(length_value), cudaMemcpyHostToDevice));
    }
};

void run_route(Case& data, RouteChoice choice, const Options& options, DBuf& flush,
               cudaStream_t stream) {
    auto plan = ops::detail::prepare_masked_block_resolve_plan(data.block_size);
    if (choice != RouteChoice::Production) plan.route = route_value(choice);
    const auto launch = [&](cudaStream_t launch_stream) {
        ops::detail::prepare_masked_block_launch(data.anchor_tensor, data.length_tensor, kMaskId,
                                                 data.ids_tensor, data.positions_tensor, plan,
                                                 launch_stream);
    };
    const char* route = ops::detail::prepare_masked_block_route_name(plan.route);
    if (options.profile_once) {
        launch(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::printf("PROFILE_ONCE route=%s B=%d threads=%d "
                    "kernel_regex='prepare_masked_block_kernel'\n",
                    route, data.block_size,
                    ops::detail::prepare_masked_block_route_threads(plan.route));
        return;
    }

    Graph graph          = capture_graph(launch, stream);
    const double bytes   = static_cast<double>(2 + 2 * data.block_size) * sizeof(std::int32_t);
    Result result        = bench_graph(graph.executable, flush, options.cold, stream, bytes,
                                       options.warmup, options.repeat);
    const double roof_us = bytes / (kRooflineGBs * 1.0e3);
    std::printf("selection=%-10s route=%-8s cache=%s B=%2d threads=%3d median_us=%7.3f "
                "min_us=%7.3f p95_us=%7.3f useful_GBs=%6.3f roof_us=%8.6f "
                "roof_eff_pct=%7.4f\n",
                choice == RouteChoice::Production ? "production" : "candidate", route,
                options.cold ? "cold" : "hot ", data.block_size,
                ops::detail::prepare_masked_block_route_threads(plan.route), result.median_us,
                result.min_us, result.p95_us, result.gbs, roof_us,
                result.median_us > 0.0 ? 100.0 * roof_us / result.median_us : 0.0);
}

} // namespace

int main(int argc, char** argv) {
    const Options options = parse_options(argc, argv);
    cudaStream_t stream   = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    DBuf flush(kFlush);
    std::printf("# RTX 5090 sm_120a; graph replay; exact I32 anchor/mask block; cache=%s\n",
                options.cold ? "cold" : "hot");

    for (const int block_size : options.block_sizes) {
        Case data(block_size);
        if (options.route == RouteChoice::All) {
            for (const auto route : {RouteChoice::Warp32, RouteChoice::Block64,
                                     RouteChoice::Block128, RouteChoice::Block256}) {
                run_route(data, route, options, flush, stream);
            }
        } else {
            run_route(data, options.route, options, flush, stream);
        }
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
