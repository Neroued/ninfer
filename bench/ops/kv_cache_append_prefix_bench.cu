// Candidate and production benchmark for device-count exact K/V prefix append.
// Every timed invocation is one CUDA Graph replay. Cold mode flushes 256 MiB before each replay
// and excludes the flush from the measured interval.
#include "core/device.h"
#include "core/kv_cache.h"
#include "ninfer/ops/kv_cache_append_prefix.h"
#include "ninfer_bench_common.h"
#include "ops/launcher/kv_cache_append_prefix.h"

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

constexpr int kD             = 128;
constexpr int kKVHeads       = 8;
constexpr int kCapacity      = 4096;
constexpr std::size_t kFlush = std::size_t{256} << 20;

enum class LayoutChoice {
    Linear,
    Cyclic,
    All,
};

enum class RouteChoice {
    Production,
    Flat16,
    Flat32,
    Persistent32,
    Token,
    All,
};

struct Options {
    std::vector<int> tokens{1, 2, 4, 8, 12, 16};
    std::vector<int> counts{0, 1, 4, 8, 12, 16};
    LayoutChoice layout = LayoutChoice::All;
    RouteChoice route   = RouteChoice::All;
    bool cold           = false;
    bool profile_once   = false;
    int warmup          = 10;
    int repeat          = 61;
};

[[noreturn]] void usage(const char* message) {
    std::fprintf(stderr, "error: %s\n", message);
    std::fprintf(stderr, "usage: ninfer_kv_cache_append_prefix_bench [--tokens 1,...] "
                         "[--counts 0,...] [--layout linear|cyclic|all] "
                         "[--route production|flat16|flat32|persistent32|token|all] "
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
        const auto comma           = remaining.find(',');
        const std::string_view one = remaining.substr(0, comma);
        if (one.empty()) usage(flag);
        result.push_back(parse_int(one, minimum, maximum, flag));
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
        if (arg == "--tokens") {
            options.tokens =
                parse_list(next("--tokens requires a value"), 1, kCapacity, "--tokens");
        } else if (arg == "--counts") {
            options.counts =
                parse_list(next("--counts requires a value"), 0, kCapacity, "--counts");
        } else if (arg == "--layout") {
            const std::string_view value(next("--layout requires a value"));
            if (value == "linear") {
                options.layout = LayoutChoice::Linear;
            } else if (value == "cyclic") {
                options.layout = LayoutChoice::Cyclic;
            } else if (value == "all") {
                options.layout = LayoutChoice::All;
            } else {
                usage("--layout expects linear, cyclic, or all");
            }
        } else if (arg == "--route") {
            const std::string_view value(next("--route requires a value"));
            if (value == "production") {
                options.route = RouteChoice::Production;
            } else if (value == "flat16") {
                options.route = RouteChoice::Flat16;
            } else if (value == "flat32") {
                options.route = RouteChoice::Flat32;
            } else if (value == "persistent32") {
                options.route = RouteChoice::Persistent32;
            } else if (value == "token") {
                options.route = RouteChoice::Token;
            } else if (value == "all") {
                options.route = RouteChoice::All;
            } else {
                usage("--route expects production, flat16, flat32, persistent32, token, or all");
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

Result bench_cold_graph(cudaGraphExec_t graph, DBuf& flush, cudaStream_t stream, double bytes,
                        int warmup, int repeat) {
    cudaEvent_t begin = nullptr;
    cudaEvent_t end   = nullptr;
    CUDA_CHECK(cudaEventCreate(&begin));
    CUDA_CHECK(cudaEventCreate(&end));
    for (int i = 0; i < warmup; ++i) {
        CUDA_CHECK(cudaMemsetAsync(flush.p, 0xa5, flush.bytes, stream));
        CUDA_CHECK(cudaGraphLaunch(graph, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repeat));
    for (int i = 0; i < repeat; ++i) {
        CUDA_CHECK(cudaMemsetAsync(flush.p, 0xa5, flush.bytes, stream));
        CUDA_CHECK(cudaEventRecord(begin, stream));
        CUDA_CHECK(cudaGraphLaunch(graph, stream));
        CUDA_CHECK(cudaEventRecord(end, stream));
        CUDA_CHECK(cudaEventSynchronize(end));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, begin, end));
        samples.push_back(static_cast<double>(ms) * 1000.0);
    }
    CUDA_CHECK(cudaEventDestroy(begin));
    CUDA_CHECK(cudaEventDestroy(end));

    std::sort(samples.begin(), samples.end());
    Result result;
    result.n_runs      = repeat;
    result.inner_iters = 1;
    result.median_us   = samples[samples.size() / 2];
    result.min_us      = samples.front();
    result.p95_us      = samples[std::min(samples.size() - 1, samples.size() * 95 / 100)];
    result.gbs         = bytes / (result.median_us * 1.0e3);
    return result;
}

struct Case {
    int tokens;
    bool cyclic;
    DBuf k;
    DBuf v;
    DBuf positions;
    DBuf count;
    DBuf cache_k;
    DBuf cache_v;
    Tensor tk;
    Tensor tv;
    Tensor tp;
    Tensor tc;

    Case(int t, int committed, bool is_cyclic)
        : tokens(t), cyclic(is_cyclic), k(make_bf16(static_cast<std::size_t>(kD) * kKVHeads * t)),
          v(make_bf16(static_cast<std::size_t>(kD) * kKVHeads * t)),
          positions(static_cast<std::size_t>(t) * sizeof(std::int32_t)),
          count(sizeof(std::int32_t)),
          cache_k(make_zeros(static_cast<std::size_t>(kD) * kCapacity * kKVHeads * 2)),
          cache_v(make_zeros(static_cast<std::size_t>(kD) * kCapacity * kKVHeads * 2)),
          tk(k.p, DType::BF16, {kD, kKVHeads, t}), tv(v.p, DType::BF16, {kD, kKVHeads, t}),
          tp(positions.p, DType::I32, {t}), tc(count.p, DType::I32, {1}) {
        const int start = cyclic ? 2 * kCapacity - 3 : 0;
        std::vector<std::int32_t> host_positions(static_cast<std::size_t>(tokens));
        for (int i = 0; i < tokens; ++i) host_positions[static_cast<std::size_t>(i)] = start + i;
        CUDA_CHECK(cudaMemcpy(positions.p, host_positions.data(), positions.bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(count.p, &committed, sizeof(committed), cudaMemcpyHostToDevice));
    }

    KVCacheLayerView linear_view() {
        return {
            .k              = Tensor(cache_k.p, DType::BF16, {kD, kCapacity, kKVHeads}),
            .v              = Tensor(cache_v.p, DType::BF16, {kD, kCapacity, kKVHeads}),
            .k_scale        = Tensor(),
            .v_scale        = Tensor(),
            .max_context    = kCapacity,
            .padded_context = kCapacity,
            .num_kv_heads   = kKVHeads,
            .head_dim       = kD,
            .dtype          = DType::BF16,
            .quant_group    = 0,
        };
    }

    CyclicKVCacheLayerView cyclic_view() {
        return {
            .k               = Tensor(cache_k.p, DType::BF16, {kD, kCapacity, kKVHeads}),
            .v               = Tensor(cache_v.p, DType::BF16, {kD, kCapacity, kKVHeads}),
            .k_scale         = Tensor(),
            .v_scale         = Tensor(),
            .capacity        = kCapacity,
            .padded_capacity = kCapacity,
            .num_kv_heads    = kKVHeads,
            .head_dim        = kD,
            .dtype           = DType::BF16,
            .quant_group     = 0,
        };
    }
};

ops::detail::KVCacheAppendPrefixRoute route_value(RouteChoice route) {
    switch (route) {
    case RouteChoice::Flat16:
        return ops::detail::KVCacheAppendPrefixRoute::Flat16;
    case RouteChoice::Flat32:
        return ops::detail::KVCacheAppendPrefixRoute::Flat32;
    case RouteChoice::Persistent32:
        return ops::detail::KVCacheAppendPrefixRoute::Persistent32;
    case RouteChoice::Token:
        return ops::detail::KVCacheAppendPrefixRoute::Token;
    case RouteChoice::Production:
    case RouteChoice::All:
        break;
    }
    return ops::detail::KVCacheAppendPrefixRoute::Flat32;
}

int route_grid(const ops::detail::KVCacheAppendPrefixPlan& plan) {
    switch (plan.route) {
    case ops::detail::KVCacheAppendPrefixRoute::Flat16:
        return (plan.max_count * 128 + 255) / 256;
    case ops::detail::KVCacheAppendPrefixRoute::Flat32:
        return (plan.max_count * 64 + 255) / 256;
    case ops::detail::KVCacheAppendPrefixRoute::Persistent32:
        return 1;
    case ops::detail::KVCacheAppendPrefixRoute::Token:
        return plan.max_count;
    }
    return 0;
}

void run_route(Case& data, int committed, RouteChoice choice, const Options& options, DBuf& flush,
               cudaStream_t stream) {
    const ops::KVCacheAppendPrefixExecutionEnvelope envelope{
        0,
        static_cast<std::uint32_t>(data.tokens),
    };
    auto plan = ops::detail::kv_cache_append_prefix_resolve_plan(data.tokens, envelope);
    if (choice != RouteChoice::Production) plan.route = route_value(choice);
    const auto launch = [&](cudaStream_t launch_stream) {
        if (data.cyclic) {
            ops::detail::kv_cache_append_prefix_launch(data.tk, data.tv, data.tp, data.tc,
                                                       data.cyclic_view(), plan, launch_stream);
        } else {
            ops::detail::kv_cache_append_prefix_launch(data.tk, data.tv, data.tp, data.tc,
                                                       data.linear_view(), plan, launch_stream);
        }
    };
    const char* route = ops::detail::kv_cache_append_prefix_route_name(plan.route);
    if (options.profile_once) {
        launch(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::printf("PROFILE_ONCE layout=%s route=%s T=%d C=%d grid=%d kernel_regex='%s'\n",
                    data.cyclic ? "cyclic" : "linear", route, data.tokens, committed,
                    route_grid(plan),
                    plan.route == ops::detail::KVCacheAppendPrefixRoute::Token
                        ? "kv_cache_append_prefix_token_kernel"
                        : "kv_cache_append_prefix_flat_kernel");
        return;
    }

    Graph graph        = capture_graph(launch, stream);
    const double bytes = static_cast<double>(committed) * 8192.0;
    Result result;
    if (options.cold) {
        result = bench_cold_graph(graph.executable, flush, stream, bytes, options.warmup,
                                  options.repeat);
    } else {
        result = bench_loop(
            [&](cudaStream_t launch_stream) {
                CUDA_CHECK(cudaGraphLaunch(graph.executable, launch_stream));
            },
            bytes, options.warmup, options.repeat, 20);
    }
    const double roof_us = bytes / (kRooflineGBs * 1.0e3);
    std::printf("layout=%-6s route=%-12s cache=%s T=%4d C=%4d grid=%4d median_us=%7.3f "
                "min_us=%7.3f p95_us=%7.3f useful_GBs=%7.1f roof_us=%6.3f roof_eff_pct=%5.1f\n",
                data.cyclic ? "cyclic" : "linear", route, options.cold ? "cold" : "hot ",
                data.tokens, committed, route_grid(plan), result.median_us, result.min_us,
                result.p95_us, result.gbs, roof_us,
                result.median_us > 0.0 ? 100.0 * roof_us / result.median_us : 0.0);
}

} // namespace

int main(int argc, char** argv) {
    const Options options = parse_options(argc, argv);
    cudaStream_t stream   = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    DBuf flush(kFlush);
    std::printf("# RTX 5090 sm_120a; graph replay; exact BF16 D128/KV8 prefix append; cache=%s\n",
                options.cold ? "cold" : "hot");

    for (const bool cyclic : {false, true}) {
        if ((cyclic && options.layout == LayoutChoice::Linear) ||
            (!cyclic && options.layout == LayoutChoice::Cyclic)) {
            continue;
        }
        for (const int tokens : options.tokens) {
            for (const int committed : options.counts) {
                if (committed > tokens) continue;
                Case data(tokens, committed, cyclic);
                if (options.route == RouteChoice::All) {
                    for (const auto route : {RouteChoice::Flat16, RouteChoice::Flat32,
                                             RouteChoice::Persistent32, RouteChoice::Token}) {
                        run_route(data, committed, route, options, flush, stream);
                    }
                } else {
                    run_route(data, committed, options.route, options, flush, stream);
                }
            }
        }
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
