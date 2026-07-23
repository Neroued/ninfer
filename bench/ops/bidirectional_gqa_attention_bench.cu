// Candidate and production benchmark for the fixed Qwen3.6 35B-A3B bidirectional GQA geometry.
// Every timed invocation is a CUDA Graph replay. Cold mode flushes 256 MiB immediately before
// each replay and excludes the flush from the timed interval.
#include "core/device.h"
#include "core/kv_cache.h"
#include "ninfer/ops/bidirectional_gqa_attention.h"
#include "ninfer_bench_common.h"
#include "ops/launcher/bidirectional_gqa_attention.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

constexpr int kD              = 128;
constexpr int kQHeads         = 32;
constexpr int kKVHeads        = 8;
constexpr float kScale        = 0.08838834764831844055f;
constexpr std::size_t kFlush  = std::size_t{256} << 20;
constexpr double kTcPeakTflop = 209.5;

enum class RouteChoice {
    Production,
    Direct,
    Split,
    All,
};

struct Options {
    std::vector<int> tokens{1, 2, 4, 8, 12, 16};
    std::vector<int> contexts{0, 32, 128, 512, 2048, 8192, 32768, 131072, 262144};
    RouteChoice route  = RouteChoice::All;
    bool cold          = false;
    bool profile_once  = false;
    int split_capacity = 0;
    int key_block      = 0;
    int warmup         = 5;
    int repeat         = 30;
};

[[noreturn]] void usage(const char* message) {
    std::fprintf(stderr, "error: %s\n", message);
    std::fprintf(stderr, "usage: ninfer_bidirectional_gqa_attention_bench "
                         "[--tokens 1,...,16] [--context 0,...,262144] "
                         "[--route production|direct|split|all] [--cold-cache] [--profile-once] "
                         "[--split-capacity 1..85] [--key-block 32|64] "
                         "[--warmup N] [--repeat N]\n");
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
        const std::size_t comma     = remaining.find(',');
        const std::string_view item = remaining.substr(0, comma);
        if (item.empty()) { usage(flag); }
        result.push_back(parse_int(item, minimum, maximum, flag));
        if (comma == std::string_view::npos) { break; }
        remaining.remove_prefix(comma + 1);
    }
    if (result.empty()) { usage(flag); }
    return result;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        const auto next = [&](const char* flag) -> const char* {
            if (++i == argc) { usage(flag); }
            return argv[i];
        };
        if (arg == "--tokens") {
            options.tokens = parse_list(next("--tokens requires a value"), 1, 16, "--tokens");
        } else if (arg == "--context") {
            options.contexts =
                parse_list(next("--context requires a value"), 0, 262144, "--context");
        } else if (arg == "--route") {
            const std::string_view route(next("--route requires a value"));
            if (route == "production") {
                options.route = RouteChoice::Production;
            } else if (route == "direct") {
                options.route = RouteChoice::Direct;
            } else if (route == "split") {
                options.route = RouteChoice::Split;
            } else if (route == "all") {
                options.route = RouteChoice::All;
            } else {
                usage("--route expects production, direct, split, or all");
            }
        } else if (arg == "--cold-cache") {
            options.cold = true;
        } else if (arg == "--profile-once") {
            options.profile_once = true;
        } else if (arg == "--split-capacity") {
            options.split_capacity =
                parse_int(next("--split-capacity requires a value"), 1, 85, "--split-capacity");
        } else if (arg == "--key-block") {
            options.key_block =
                parse_int(next("--key-block requires a value"), 32, 64, "--key-block");
            if (options.key_block != 32 && options.key_block != 64) {
                usage("--key-block expects 32 or 64");
            }
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

int align_context(int value) { return ((std::max(value, 1) + 127) / 128) * 128; }

KVCacheLayerView make_context(DBuf& k, DBuf& v, int maximum, int padded) {
    return {
        .k              = Tensor(k.p, DType::BF16, {kD, padded, kKVHeads}),
        .v              = Tensor(v.p, DType::BF16, {kD, padded, kKVHeads}),
        .k_scale        = Tensor(),
        .v_scale        = Tensor(),
        .max_context    = static_cast<std::uint32_t>(maximum),
        .padded_context = static_cast<std::uint32_t>(padded),
        .num_kv_heads   = kKVHeads,
        .head_dim       = kD,
        .dtype          = DType::BF16,
        .quant_group    = 0,
    };
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

    std::vector<double> sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    double sum = 0.0;
    for (double value : samples) sum += value;
    const auto percentile = [&](double fraction) {
        return sorted[std::min(sorted.size() - 1,
                               static_cast<std::size_t>(fraction * sorted.size()))];
    };
    Result result;
    result.n_runs      = repeat;
    result.inner_iters = 1;
    result.median_us   = percentile(0.50);
    result.min_us      = sorted.front();
    result.p95_us      = percentile(0.95);
    result.mean_us     = sum / static_cast<double>(samples.size());
    result.gbs         = bytes / (result.median_us * 1.0e3);
    return result;
}

struct Case {
    int tokens;
    int context_length;
    int padded_context;
    DBuf q;
    DBuf query_k;
    DBuf query_v;
    DBuf context_k;
    DBuf context_v;
    DBuf length;
    DBuf out;
    DBuf partial_acc;
    DBuf partial_m;
    DBuf partial_l;
    Tensor tq;
    Tensor tk;
    Tensor tv;
    Tensor tl;
    Tensor tout;
    Tensor tpartial_acc;
    Tensor tpartial_m;
    Tensor tpartial_l;
    KVCacheLayerView context;

    Case(int t, int l)
        : tokens(t), context_length(l), padded_context(align_context(l)),
          q(make_bf16(static_cast<std::size_t>(kD) * kQHeads * t)),
          query_k(make_bf16(static_cast<std::size_t>(kD) * kKVHeads * t)),
          query_v(make_bf16(static_cast<std::size_t>(kD) * kKVHeads * t)),
          context_k(make_zeros(static_cast<std::size_t>(kD) * padded_context * kKVHeads * 2)),
          context_v(make_zeros(static_cast<std::size_t>(kD) * padded_context * kKVHeads * 2)),
          length(sizeof(std::int32_t)),
          out(make_zeros(static_cast<std::size_t>(kD) * kQHeads * t * 2)),
          partial_acc(static_cast<std::size_t>(kD) * kQHeads * t * 85 * 2),
          partial_m(static_cast<std::size_t>(kQHeads) * t * 85 * sizeof(float)),
          partial_l(static_cast<std::size_t>(kQHeads) * t * 85 * sizeof(float)),
          tq(q.p, DType::BF16, {kD, kQHeads, t}), tk(query_k.p, DType::BF16, {kD, kKVHeads, t}),
          tv(query_v.p, DType::BF16, {kD, kKVHeads, t}), tl(length.p, DType::I32, {1}),
          tout(out.p, DType::BF16, {kD, kQHeads, t}),
          tpartial_acc(partial_acc.p, DType::BF16, {kD, kQHeads, t, 85}),
          tpartial_m(partial_m.p, DType::FP32, {kQHeads, t, 85}),
          tpartial_l(partial_l.p, DType::FP32, {kQHeads, t, 85}),
          context(make_context(context_k, context_v, std::max(l, 1), padded_context)) {
        CUDA_CHECK(
            cudaMemcpy(length.p, &context_length, sizeof(context_length), cudaMemcpyHostToDevice));
    }
};

ops::detail::BidirectionalGqaPlan candidate_plan(int tokens,
                                                 ops::detail::BidirectionalGqaRoute route,
                                                 int split_capacity = 0, int key_block = 0) {
    auto plan = ops::detail::bidirectional_gqa_resolve_plan(
        tokens, {0u, static_cast<std::uint32_t>(std::numeric_limits<int>::max())});
    plan.route = route;
    if (route == ops::detail::BidirectionalGqaRoute::Direct) {
        plan.split_capacity = 1;
        plan.key_block      = 32;
    } else if (split_capacity != 0) {
        plan.split_capacity = split_capacity;
    }
    if (key_block != 0) plan.key_block = key_block;
    return plan;
}

double useful_bytes(int tokens, int context) {
    return static_cast<double>(context) * 4096.0 + static_cast<double>(tokens) * 20480.0;
}

double useful_flops(int tokens, int context) {
    return 4.0 * static_cast<double>(tokens) * kQHeads * static_cast<double>(context + tokens) * kD;
}

void report(const char* route, int tokens, int context, const Result& result, bool cold) {
    const double bytes      = useful_bytes(tokens, context);
    const double flops      = useful_flops(tokens, context);
    const double bw_floor   = bytes / (kRooflineGBs * 1.0e3);
    const double tc_floor   = flops / (kTcPeakTflop * 1.0e6);
    const double roof_floor = std::max(bw_floor, tc_floor);
    const double tflops     = flops / (result.median_us * 1.0e6);
    std::printf("route=%-10s cache=%s T=%2d L=%6d median_us=%9.3f min_us=%9.3f p95_us=%9.3f "
                "useful_GBs=%8.1f useful_TF=%7.2f roof_us=%8.3f roof_eff_pct=%6.1f\n",
                route, cold ? "cold" : "hot ", tokens, context, result.median_us, result.min_us,
                result.p95_us, result.gbs, tflops, roof_floor,
                result.median_us > 0.0 ? 100.0 * roof_floor / result.median_us : 0.0);
}

void run_route(Case& data, const ops::detail::BidirectionalGqaPlan& plan, const Options& options,
               DBuf& flush, cudaStream_t stream) {
    const auto launch = [&](cudaStream_t launch_stream) {
        ops::detail::bidirectional_gqa_attention_launch(
            data.tq, data.tk, data.tv, data.tl, kScale, data.context, plan, data.tpartial_acc,
            data.tpartial_m, data.tpartial_l, data.tout, launch_stream);
    };
    if (options.profile_once) {
        if (options.cold) { CUDA_CHECK(cudaMemsetAsync(flush.p, 0xa5, flush.bytes, stream)); }
        launch(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::printf("PROFILE_ONCE route=%s T=%d L=%d key_block=%d split_capacity=%d "
                    "cache=%s partial_kernel_regex="
                    "'bidirectional_gqa_split_partial_kernel' reduce_kernel_regex="
                    "'bidirectional_gqa_reduce_kernel'\n",
                    ops::detail::bidirectional_gqa_route_name(plan.route), data.tokens,
                    data.context_length, plan.key_block, plan.split_capacity,
                    options.cold ? "cold" : "hot");
        return;
    }

    Graph graph        = capture_graph(launch, stream);
    const double bytes = useful_bytes(data.tokens, data.context_length);
    Result result;
    if (options.cold) {
        result = bench_cold_graph(graph.executable, flush, stream, bytes, options.warmup,
                                  options.repeat);
    } else {
        result = bench_loop(
            [&](cudaStream_t launch_stream) {
                CUDA_CHECK(cudaGraphLaunch(graph.executable, launch_stream));
            },
            bytes, options.warmup, options.repeat, 100);
    }
    report(ops::detail::bidirectional_gqa_route_name(plan.route), data.tokens, data.context_length,
           result, options.cold);
}

} // namespace

int main(int argc, char** argv) {
    const Options options = parse_options(argc, argv);
    cudaStream_t stream   = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    DBuf flush(kFlush);

    std::printf("# RTX 5090 sm_120a; graph replay; BF16 D=128 Hq=32 Hkv=8; "
                "DRAM=1792 GB/s TC=209.5 TF/s; cache=%s\n",
                options.cold ? "cold" : "hot");
    for (const int context : options.contexts) {
        for (const int tokens : options.tokens) {
            Case data(tokens, context);
            const auto production = ops::detail::bidirectional_gqa_resolve_plan(
                tokens, {static_cast<std::uint32_t>(context), static_cast<std::uint32_t>(context)});
            if (options.route == RouteChoice::Production) {
                run_route(data, production, options, flush, stream);
            } else if (options.route == RouteChoice::Direct) {
                run_route(data, candidate_plan(tokens, ops::detail::BidirectionalGqaRoute::Direct),
                          options, flush, stream);
            } else if (options.route == RouteChoice::Split) {
                run_route(data,
                          candidate_plan(tokens, ops::detail::BidirectionalGqaRoute::SplitKv,
                                         options.split_capacity, options.key_block),
                          options, flush, stream);
            } else {
                run_route(data, candidate_plan(tokens, ops::detail::BidirectionalGqaRoute::Direct),
                          options, flush, stream);
                run_route(data,
                          candidate_plan(tokens, ops::detail::BidirectionalGqaRoute::SplitKv,
                                         options.split_capacity, options.key_block),
                          options, flush, stream);
            }
        }
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
