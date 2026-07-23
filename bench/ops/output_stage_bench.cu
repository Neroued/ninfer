// Captured Qwen3.6-35B-A3B target-verification output-stage benchmark.
//
// The timed graph is final RMSNorm -> Q6 full-vocabulary head -> argmax. The
// production route is compared with the route table that preceded DV-06.

#include "ninfer/ops/argmax.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/rmsnorm.h"

#include "core/device.h"
#include "ninfer_bench_common.h"
#include "ops/linear/q6/q6_rowsplit_plan.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;

namespace {

constexpr std::int32_t kHidden          = 2048;
constexpr std::int32_t kVocab           = 248320;
constexpr std::int32_t kMaxTokens       = 16;
constexpr std::size_t kDefaultFlushSize = 256ULL << 20;
constexpr float kRmsEps                 = 1.0e-6F;

enum class Route {
    Production,
    Dv06Control,
};

struct Options {
    std::vector<std::int32_t> t_sweep{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    Route route            = Route::Production;
    int warmup             = 5;
    int repeat             = 50;
    std::size_t flush_size = kDefaultFlushSize;
};

struct Timing {
    double median_us = 0.0;
    double min_us    = 0.0;
    double p95_us    = 0.0;
};

struct Q6Weight {
    bench::DBuf storage;
    Weight weight{};
};

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

__global__ void fill_q6_scales(std::uint16_t* scales, std::uint64_t groups) {
    const std::uint64_t begin = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t step  = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t group = begin; group < groups; group += step) {
        scales[group] = 0x3c00u; // FP16 1.0
    }
}

Q6Weight make_q6_weight() {
    constexpr std::int32_t group = 64;
    const std::uint64_t groups =
        static_cast<std::uint64_t>(kVocab) * static_cast<std::uint64_t>(kHidden / group);
    const std::uint64_t code_bytes   = groups * 32;
    const std::uint64_t high_offset  = align_up(code_bytes, 256);
    const std::uint64_t high_bytes   = groups * 16;
    const std::uint64_t scale_offset = high_offset + align_up(high_bytes, 256);
    const std::uint64_t scale_bytes  = groups * sizeof(std::uint16_t);

    Q6Weight result{bench::DBuf(static_cast<std::size_t>(scale_offset + scale_bytes)), {}};
    CUDA_CHECK(cudaMemset(result.storage.p, 0x35, code_bytes));
    CUDA_CHECK(
        cudaMemset(static_cast<std::uint8_t*>(result.storage.p) + high_offset, 0x12, high_bytes));
    constexpr int block = 256;
    const int grid      = static_cast<int>(
        std::min<std::uint64_t>(65535, std::max<std::uint64_t>(1, (groups + block - 1) / block)));
    fill_q6_scales<<<grid, block>>>(
        reinterpret_cast<std::uint16_t*>(static_cast<std::uint8_t*>(result.storage.p) +
                                         scale_offset),
        groups);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    Weight& weight          = result.weight;
    weight.payload          = result.storage.p;
    weight.payload_bytes    = result.storage.bytes;
    weight.high_plane_bytes = high_bytes;
    weight.qtype            = QType::Q6G64_F16S;
    weight.layout           = QuantLayout::RowSplit;
    weight.scale_dtype      = DType::FP16;
    weight.group_size       = group;
    weight.shape[0]         = kVocab;
    weight.shape[1]         = kHidden;
    weight.padded_shape[0]  = kVocab;
    weight.padded_shape[1]  = kHidden;
    weight.ndim             = 2;
    weight.qdata            = result.storage.p;
    weight.qhigh            = static_cast<std::uint8_t*>(result.storage.p) + high_offset;
    weight.scales           = static_cast<std::uint8_t*>(result.storage.p) + scale_offset;
    weight.n                = kVocab;
    weight.k                = kHidden;
    weight.group            = group;
    return result;
}

std::vector<std::int32_t> parse_t_sweep(std::string_view raw) {
    std::vector<std::int32_t> result;
    for (std::size_t begin = 0; begin < raw.size();) {
        const std::size_t end = raw.find(',', begin);
        const std::string token(
            raw.substr(begin, end == std::string_view::npos ? raw.size() - begin : end - begin));
        const long value = std::stol(token);
        if (value < 1 || value > kMaxTokens) {
            throw std::invalid_argument("--t-sweep values must be in [1,16]");
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
        } else if (arg == "--route") {
            const std::string_view route = next("--route value");
            if (route == "production") {
                options.route = Route::Production;
            } else if (route == "dv06-control") {
                options.route = Route::Dv06Control;
            } else {
                throw std::invalid_argument("--route must be production or dv06-control");
            }
        } else if (arg == "--warmup") {
            options.warmup = std::stoi(std::string(next("--warmup value")));
        } else if (arg == "--repeat") {
            options.repeat = std::stoi(std::string(next("--repeat value")));
        } else if (arg == "--flush-mib") {
            const long mib = std::stol(std::string(next("--flush-mib value")));
            if (mib <= 0) { throw std::invalid_argument("--flush-mib must be positive"); }
            options.flush_size = static_cast<std::size_t>(mib) << 20;
        } else if (arg == "--help" || arg == "-h") {
            std::printf("Usage: %s [--t-sweep 1,2,...,16] "
                        "[--route production|dv06-control] [--warmup N] [--repeat N] "
                        "[--flush-mib N]\n",
                        argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(arg));
        }
    }
    if (options.warmup < 0 || options.repeat <= 0) {
        throw std::invalid_argument("--warmup must be nonnegative and --repeat positive");
    }
    return options;
}

class TimedGraph {
public:
    TimedGraph() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }

    ~TimedGraph() {
        if (exec_ != nullptr) cudaGraphExecDestroy(exec_);
        if (graph_ != nullptr) cudaGraphDestroy(graph_);
        if (start_ != nullptr) cudaEventDestroy(start_);
        if (stop_ != nullptr) cudaEventDestroy(stop_);
    }

    template <class Body>
    void capture(cudaStream_t stream, Body&& body) {
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        try {
            body(stream);
        } catch (...) {
            cudaGraph_t discard = nullptr;
            cudaStreamEndCapture(stream, &discard);
            if (discard != nullptr) cudaGraphDestroy(discard);
            throw;
        }
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph_));
        CUDA_CHECK(cudaGraphInstantiate(&exec_, graph_, 0));
        CUDA_CHECK(cudaGraphGetNodes(graph_, nullptr, &nodes_));
        if (nodes_ == 0) { throw std::runtime_error("captured output-stage graph is empty"); }
    }

    void launch(cudaStream_t stream) const { CUDA_CHECK(cudaGraphLaunch(exec_, stream)); }

    double launch_timed(cudaStream_t stream) const {
        CUDA_CHECK(cudaEventRecord(start_, stream));
        launch(stream);
        CUDA_CHECK(cudaEventRecord(stop_, stream));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float milliseconds = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_, stop_));
        return static_cast<double>(milliseconds) * 1000.0;
    }

    std::size_t nodes() const noexcept { return nodes_; }

private:
    cudaGraph_t graph_    = nullptr;
    cudaGraphExec_t exec_ = nullptr;
    cudaEvent_t start_    = nullptr;
    cudaEvent_t stop_     = nullptr;
    std::size_t nodes_    = 0;
};

Timing measure_cold(const TimedGraph& graph, bench::DBuf& flush, cudaStream_t stream, int warmup,
                    int repeat) {
    for (int i = 0; i < warmup; ++i) {
        CUDA_CHECK(cudaMemsetAsync(flush.p, 0xa5, flush.bytes, stream));
        graph.launch(stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repeat));
    for (int i = 0; i < repeat; ++i) {
        CUDA_CHECK(cudaMemsetAsync(flush.p, 0xa5, flush.bytes, stream));
        samples.push_back(graph.launch_timed(stream));
    }
    std::sort(samples.begin(), samples.end());
    const auto percentile = [&](double q) {
        const std::size_t index =
            std::min(samples.size() - 1,
                     static_cast<std::size_t>(q * static_cast<double>(samples.size() - 1)));
        return samples[index];
    };
    return {percentile(0.50), samples.front(), percentile(0.95)};
}

ops::detail::Q6Plan dv06_control_plan(std::int32_t tokens) {
    using S = ops::detail::Q6ScheduleId;
    using V = ops::detail::Q6KernelVariant;
    if (tokens <= 4) { return {S::SimtR8C4, V::None}; }
    if (tokens <= 6) { return {S::SimtR8C8, V::None}; }
    return {S::MmaR64C128, V::Predicated};
}

const char* route_name(Route route) {
    return route == Route::Production ? "production" : "dv06_control";
}

int run(const Options& options) {
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    Q6Weight head        = make_q6_weight();
    bench::DBuf residual = bench::make_bf16(static_cast<std::size_t>(kHidden) * kMaxTokens);
    std::vector<std::uint16_t> norm_host(kHidden, bench::f32_to_bf16(0.0F));
    bench::DBuf norm(norm_host.size() * sizeof(std::uint16_t));
    CUDA_CHECK(cudaMemcpy(norm.p, norm_host.data(), norm.bytes, cudaMemcpyHostToDevice));
    bench::DBuf hidden(static_cast<std::size_t>(kHidden) * kMaxTokens * sizeof(std::uint16_t));
    bench::DBuf logits(static_cast<std::size_t>(kVocab) * kMaxTokens * sizeof(std::uint16_t));
    bench::DBuf tokens(static_cast<std::size_t>(kMaxTokens) * sizeof(std::int32_t));
    bench::DBuf flush(options.flush_size);
    WorkspaceArena workspace(1);

    std::printf("# gpu=RTX_5090 cuda=13.1 sm=120a flush_mib=%zu warmup=%d repeat=%d\n",
                options.flush_size >> 20, options.warmup, options.repeat);
    std::printf("%4s %-14s %5s %10s %10s %10s %s\n", "T", "control", "nodes", "median_us", "min_us",
                "p95_us", "head_route");

    for (const std::int32_t t : options.t_sweep) {
        Tensor x(residual.p, DType::BF16, {kHidden, t});
        Tensor norm_weight(norm.p, DType::BF16, {kHidden});
        Tensor normalized(hidden.p, DType::BF16, {kHidden, t});
        Tensor output(logits.p, DType::BF16, {kVocab, t});
        Tensor selected(tokens.p, DType::I32, {t});
        const ops::detail::Q6Plan production =
            ops::detail::q6_rowsplit_resolve_plan({kVocab, kHidden, kHidden, t});
        const ops::detail::Q6Plan head_plan =
            options.route == Route::Production ? production : dv06_control_plan(t);

        const auto body = [&](cudaStream_t body_stream) {
            ops::rmsnorm(x, norm_weight, kRmsEps, true, normalized, body_stream);
            if (options.route == Route::Production) {
                ops::linear(normalized, head.weight, output, workspace, body_stream);
            } else {
                ops::detail::q6_rowsplit_execute_plan(head_plan, normalized, head.weight, output,
                                                      body_stream);
            }
            ops::argmax(output, selected, kVocab, body_stream);
        };

        body(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        TimedGraph graph;
        graph.capture(stream, body);
        const Timing timing = measure_cold(graph, flush, stream, options.warmup, options.repeat);
        std::printf("%4d %-14s %5zu %10.3f %10.3f %10.3f %s\n", t, route_name(options.route),
                    graph.nodes(), timing.median_us, timing.min_us, timing.p95_us,
                    ops::detail::q6_schedule_name(head_plan.schedule));
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        return run(parse_options(argc, argv));
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_output_stage_bench: %s\n", error.what());
        return 2;
    }
}
