// Temporary PR #324 candidate comparison; not a public Op benchmark.
#include "ninfer_bench_common.h"
#include "quantized_weight.cuh"
#include "ops/attn_input_proj/fp8/fp8_attn_input_plan.h"

#include <cuda_bf16.h>
#include <cuda_profiler_api.h>

#include <charconv>
#include <filesystem>
#include <fstream>
#include <string_view>

using namespace ninfer;

namespace {
struct Options {
    std::vector<int> tokens{192, 256, 512, 1024, 2048, 4096};
    std::string candidate = "both", cache = "cold", csv;
    bool graph = false, profile = false;
    int warmup = 5, repeat = 30;
};

void usage() {
    std::puts("fp8_tma_prefill_pilot [--tokens T,...] [--candidate baseline|tma|both]\n"
              "  [--cache cold|warm|both] [--execution eager|graph]\n"
              "  [--warmup N] [--repeat N] [--csv-out PATH] [--profile]\n"
              "Temporary private-launcher comparison, FP8/A8, T > 144 and T % 64 == 0.");
}

int integer(std::string_view value, int minimum) {
    int result        = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || result < minimum)
        throw std::invalid_argument("invalid integer argument");
    return result;
}

Options parse(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg == "--help" || arg == "-h") {
            usage();
            std::exit(0);
        }
        if (arg == "--profile") {
            options.profile = true;
            continue;
        }
        if (++i == argc) throw std::invalid_argument("missing argument value");
        const std::string_view value(argv[i]);
        if (arg == "--tokens") {
            options.tokens.clear();
            auto rest = value;
            for (;;) {
                const auto comma = rest.find(',');
                options.tokens.push_back(integer(rest.substr(0, comma), 1));
                if (comma == std::string_view::npos) break;
                rest.remove_prefix(comma + 1);
            }
        } else if (arg == "--candidate")
            options.candidate = value;
        else if (arg == "--cache")
            options.cache = value;
        else if (arg == "--csv-out")
            options.csv = value;
        else if (arg == "--warmup")
            options.warmup = integer(value, 0);
        else if (arg == "--repeat")
            options.repeat = integer(value, 1);
        else if (arg == "--execution") {
            if (value != "eager" && value != "graph")
                throw std::invalid_argument("--execution expects eager or graph");
            options.graph = value == "graph";
        } else
            throw std::invalid_argument("unknown argument");
    }
    if (options.candidate != "baseline" && options.candidate != "tma" &&
        options.candidate != "both")
        throw std::invalid_argument("--candidate expects baseline, tma, or both");
    if (options.cache != "cold" && options.cache != "warm" && options.cache != "both")
        throw std::invalid_argument("--cache expects cold, warm, or both");
    for (int t : options.tokens)
        if (t <= 144 || t % 64 || t > 64 * 65535)
            throw std::invalid_argument("pilot needs full 64-token tiles, 144 < T <= 4194240");
    if (options.profile &&
        (options.tokens.size() != 1 || options.candidate == "both" || options.cache == "both"))
        throw std::invalid_argument("--profile needs one candidate, one T, and one cache state");
    return options;
}

// Same varied fixtures as the public FP8 attention-input benchmark.
__device__ unsigned mix_bits(unsigned v) {
    v ^= v >> 16;
    v *= 0x7feb352dU;
    v ^= v >> 15;
    v *= 0x846ca68bU;
    return v ^ (v >> 16);
}

__global__ void fill_input(__nv_bfloat16* x, std::size_t n) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        x[i] = __float2bfloat16_rn(
            (float(mix_bits(static_cast<unsigned>(i) + 913U) >> 8) * (2.f / 16777216.f) - 1.f) *
            .01f);
}

__global__ void fill_fp8(std::uint8_t* codes, std::uint16_t* scales, std::size_t n, int rows) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        unsigned code = mix_bits(static_cast<unsigned>(i) + 7413U) & 255;
        if ((code & 127) == 127) --code;
        codes[i] = code;
    }
    if (i < rows) scales[i] = 0x3b80U + (mix_bits(static_cast<unsigned>(i) + 873U) & 255U);
}

void run(const Options& options) {
    constexpr int hidden = 5120, q_rows = 6144, kv_rows = 1024, parent_rows = 14336;
    const int max_t = *std::max_element(options.tokens.begin(), options.tokens.end());
    auto weight     = bench::make_fp8_weight(parent_rows, hidden);
    auto* data      = static_cast<std::uint8_t*>(weight.storage.p);
    fill_fp8<<<(weight.low_bytes + 255) / 256, 256>>>(
        data, reinterpret_cast<std::uint16_t*>(data + weight.scale_offset), weight.low_bytes,
        parent_rows);
    CUDA_CHECK(cudaGetLastError());
    DeviceBuffer input(std::size_t(hidden) * max_t * 2);
    fill_input<<<(std::size_t(hidden) * max_t + 255) / 256, 256>>>(
        static_cast<__nv_bfloat16*>(input.p), std::size_t(hidden) * max_t);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    DeviceBuffer q(std::size_t(q_rows) * max_t * 2), gate(std::size_t(q_rows) * max_t * 2);
    DeviceBuffer k(std::size_t(kv_rows) * max_t * 2), v(std::size_t(kv_rows) * max_t * 2);
    DeviceBuffer codes(std::size_t(hidden) * max_t), scales(std::size_t(max_t) * sizeof(float));
    DeviceBuffer flush(std::size_t{256} << 20);
    const ops::detail::Fp8A8Workspace workspace{static_cast<std::uint8_t*>(codes.p),
                                                static_cast<float*>(scales.p)};
    std::ofstream csv;
    if (!options.csv.empty()) {
        const std::filesystem::path path(options.csv);
        if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
        csv.open(path);
        if (!csv) throw std::runtime_error("cannot open CSV output");
        csv << "candidate,dispatch,tokens,cache,execution,graph_nodes,median_us,min_us,p95_us\n";
    }
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
    int runtime = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
    std::printf("# temporary FP8/A8 pilot gpu=%s cuda_runtime=%d warmup=%d repeat=%d\n",
                properties.name, runtime, options.warmup, options.repeat);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    for (int tokens : options.tokens) {
        Tensor x(input.p, DType::BF16, {hidden, tokens});
        Tensor tq(q.p, DType::BF16, {q_rows, tokens}), tg(gate.p, DType::BF16, {q_rows, tokens});
        Tensor tk(k.p, DType::BF16, {kv_rows, tokens}), tv(v.p, DType::BF16, {kv_rows, tokens});
        for (const char* cache : {"cold", "warm"}) {
            if (options.cache != "both" && options.cache != cache) continue;
            for (const char* candidate : {"baseline", "tma"}) {
                if (options.candidate != "both" && options.candidate != candidate) continue;
                const auto launch = [&](cudaStream_t s) {
                    if (std::string_view(candidate) == "baseline")
                        ops::detail::fp8_attn_input_a8_launch(x, weight.weight, tq, tg, tk, tv,
                                                              workspace, s);
                    else
                        ops::detail::fp8_attn_input_a8_tma_pilot_launch(x, weight.weight, tq, tg,
                                                                        tk, tv, workspace, s);
                };
                bench::TimedGraph graph;
                if (options.graph) graph.capture(stream, launch);
                const bool cold       = std::string_view(cache) == "cold";
                const char* execution = options.graph ? "graph" : "eager";
                if (options.profile) {
                    const auto invoke = [&] {
                        if (options.graph)
                            graph.launch(stream);
                        else
                            launch(stream);
                    };
                    for (int i = 0; i < options.warmup; ++i) invoke();
                    if (cold) bench::flush_l2(flush, stream);
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                    std::printf("PROFILE candidate=%s dispatch=private T=%d cache=%s execution=%s "
                                "graph_nodes=%zu\n",
                                candidate, tokens, cache, execution, graph.nodes());
                    std::fflush(stdout);
                    CUDA_CHECK(cudaProfilerStart());
                    invoke();
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                    CUDA_CHECK(cudaProfilerStop());
                    continue;
                }
                const auto timing =
                    options.graph
                        ? (cold ? bench::measure_cold_graph(graph, flush, stream, options.warmup,
                                                            options.repeat)
                                : bench::measure_graph(graph, stream, options.warmup,
                                                       options.repeat))
                        : (cold ? bench::measure_cold_launch(launch, flush, stream, options.warmup,
                                                             options.repeat)
                                : bench::measure_launch(launch, stream, options.warmup,
                                                        options.repeat));
                std::printf("candidate=%s dispatch=private T=%d cache=%s execution=%s "
                            "graph_nodes=%zu median=%.3f us min=%.3f us p95=%.3f us\n",
                            candidate, tokens, cache, execution, graph.nodes(), timing.median_us,
                            timing.min_us, timing.p95_us);
                if (csv.is_open())
                    csv << candidate << ",private," << tokens << ',' << cache << ',' << execution
                        << ',' << graph.nodes() << ',' << timing.median_us << ',' << timing.min_us
                        << ',' << timing.p95_us << '\n';
            }
        }
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    if (csv.is_open()) {
        csv.flush();
        if (!csv) throw std::runtime_error("cannot write CSV output");
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        run(parse(argc, argv));
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    return 0;
}
