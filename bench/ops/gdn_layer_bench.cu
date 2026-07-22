// Complete Qwen3.6-35B-A3B GDN mixer benchmark for the production small-T snapshot path.
//
// The measured layer is exactly the gdn_mix composition from hidden RMSNorm through the W8
// residual output projection. --route fused selects the production fused input/snapshot Op;
// --route composed retains the projection, convolution, and q/k/v extracts as a control.
// --qk-norm fused moves the two L2 normalizations into the recurrent kernel; composed retains the
// standalone kernels. Each T owns one captured CUDA Graph. Device timing events bracket
// each replay, so the reported interval excludes host graph-submission latency. A 256 MiB L2 flush
// precedes every measured replay and is outside the timed interval.

// Examples:
//   ./build/bench/ninfer_gdn_layer_bench
//   ./build/bench/ninfer_gdn_layer_bench --t-sweep 1,2,3,4,5,6 --repeat 50

#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/gated_delta_rule.h"
#include "ninfer/ops/gated_rmsnorm.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/l2norm.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/scatter.h"

#include "core/device.h"
#include "ninfer_bench_common.h"
#include "ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.h"
#include "ops/gdn_input_proj/w8/w8_gdn_input_plan.h"
#include "ops/linear_add/w8/w8_linear_add_plan.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace ninfer;

namespace {

constexpr std::int32_t kHidden           = 2048;
constexpr std::int32_t kHeadDim          = 128;
constexpr std::int32_t kQkHeads          = 16;
constexpr std::int32_t kValueHeads       = 32;
constexpr std::int32_t kKeyRows          = kHeadDim * kQkHeads;
constexpr std::int32_t kValueRows        = kHeadDim * kValueHeads;
constexpr std::int32_t kConvRows         = 2 * kKeyRows + kValueRows;
constexpr std::int32_t kInputRows        = kConvRows + kValueRows;
constexpr std::int32_t kConvStateRows    = 3;
constexpr std::int32_t kSnapshotSlots    = 7;
constexpr std::int32_t kInitialSlot      = 6;
constexpr float kEps                     = 1.0e-6F;
constexpr float kGdnScale                = 0.08838834764831845F; // 1 / sqrt(128)
constexpr std::size_t kDefaultFlushBytes = 256ULL << 20;

struct Options {
    std::vector<std::int32_t> t_sweep{1, 2, 3, 4, 5, 6};
    int warmup              = 5;
    int repeat              = 40;
    std::size_t flush_bytes = kDefaultFlushBytes;
    std::string route       = "fused";
    std::string qk_norm     = "fused";
    std::string csv_out;
};

struct Timing {
    double median_us = 0.0;
    double min_us    = 0.0;
    double p95_us    = 0.0;
};

struct Result {
    std::int32_t tokens     = 0;
    std::size_t graph_nodes = 0;
    std::string input_route;
    std::string control_route;
    std::string output_route;
    Timing timing;
};

struct DevicePackedWeight {
    explicit DevicePackedWeight(std::size_t bytes) : storage(bytes) {}

    bench::DBuf storage;
    Weight weight{};
};

__global__ void fill_w8_scales(std::uint8_t* scales, std::uint64_t groups) {
    const std::uint64_t begin = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t step  = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t group = begin; group < groups; group += step) {
        scales[group * 2]     = 0x00u;
        scales[group * 2 + 1] = 0x3cu; // FP16 1.0
    }
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

DevicePackedWeight make_w8_weight(std::int32_t rows, std::int32_t cols, std::uint8_t code_byte) {
    constexpr std::int32_t group = 32;
    if (cols % group != 0) { throw std::invalid_argument("W8 K must be divisible by 32"); }
    const std::uint64_t groups =
        static_cast<std::uint64_t>(rows) * static_cast<std::uint64_t>(cols / group);
    const std::uint64_t code_bytes   = groups * group;
    const std::uint64_t scale_offset = align_up(code_bytes, 256);
    const std::uint64_t scale_bytes  = groups * 2;
    DevicePackedWeight packed(static_cast<std::size_t>(scale_offset + scale_bytes));
    CUDA_CHECK(cudaMemset(packed.storage.p, code_byte, code_bytes));
    constexpr int block = 256;
    const int grid      = static_cast<int>(
        std::min<std::uint64_t>(65535, std::max<std::uint64_t>(1, (groups + block - 1) / block)));
    fill_w8_scales<<<grid, block>>>(static_cast<std::uint8_t*>(packed.storage.p) + scale_offset,
                                    groups);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    Weight& weight          = packed.weight;
    weight.payload          = packed.storage.p;
    weight.payload_bytes    = packed.storage.bytes;
    weight.high_plane_bytes = 0;
    weight.qtype            = QType::W8G32_F16S;
    weight.layout           = QuantLayout::RowSplit;
    weight.group_size       = group;
    weight.qdata            = packed.storage.p;
    weight.qhigh            = nullptr;
    weight.scales           = static_cast<std::uint8_t*>(packed.storage.p) + scale_offset;
    weight.n                = rows;
    weight.k                = cols;
    weight.group            = group;
    weight.scale_dtype      = DType::FP16;
    weight.ndim             = 2;
    weight.shape[0]         = rows;
    weight.shape[1]         = cols;
    weight.padded_shape[0]  = rows;
    weight.padded_shape[1]  = cols;
    return packed;
}

Weight make_bf16_weight(const bench::DBuf& storage, std::int32_t rows, std::int32_t cols) {
    Weight weight{};
    weight.payload          = storage.p;
    weight.payload_bytes    = static_cast<std::uint64_t>(rows) * cols * sizeof(std::uint16_t);
    weight.high_plane_bytes = 0;
    weight.qtype            = QType::BF16_CTRL;
    weight.layout           = QuantLayout::Contiguous;
    weight.qdata            = storage.p;
    weight.qhigh            = nullptr;
    weight.scales           = nullptr;
    weight.n                = rows;
    weight.k                = cols;
    weight.group            = 0;
    weight.group_size       = 0;
    weight.ndim             = 2;
    weight.shape[0]         = rows;
    weight.shape[1]         = cols;
    weight.padded_shape[0]  = rows;
    weight.padded_shape[1]  = cols;
    return weight;
}

bench::DBuf make_constant_bf16(std::size_t elements, float value) {
    std::vector<std::uint16_t> host(elements, bench::f32_to_bf16(value));
    bench::DBuf device(elements * sizeof(std::uint16_t));
    CUDA_CHECK(cudaMemcpy(device.p, host.data(), device.bytes, cudaMemcpyHostToDevice));
    return device;
}

bench::DBuf make_constant_f32(std::size_t elements, float value) {
    std::vector<float> host(elements, value);
    bench::DBuf device(elements * sizeof(float));
    CUDA_CHECK(cudaMemcpy(device.p, host.data(), device.bytes, cudaMemcpyHostToDevice));
    return device;
}

std::vector<std::int32_t> parse_t_sweep(std::string_view raw) {
    std::vector<std::int32_t> result;
    std::size_t begin = 0;
    while (begin < raw.size()) {
        const std::size_t end = raw.find(',', begin);
        const std::string token(
            raw.substr(begin, end == std::string_view::npos ? raw.size() - begin : end - begin));
        const long value = std::stol(token);
        if (value <= 0 || value > kSnapshotSlots - 1) {
            throw std::invalid_argument("--t-sweep values must be in [1,6]");
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
        } else if (arg == "--warmup") {
            options.warmup = std::stoi(std::string(next("--warmup value")));
        } else if (arg == "--repeat") {
            options.repeat = std::stoi(std::string(next("--repeat value")));
        } else if (arg == "--flush-mib") {
            const long mib = std::stol(std::string(next("--flush-mib value")));
            if (mib <= 0) { throw std::invalid_argument("--flush-mib must be positive"); }
            options.flush_bytes = static_cast<std::size_t>(mib) << 20;
        } else if (arg == "--csv-out") {
            options.csv_out = next("--csv-out path");
        } else if (arg == "--route") {
            options.route = next("--route value");
            if (options.route != "fused" && options.route != "composed") {
                throw std::invalid_argument("--route must be fused or composed");
            }
        } else if (arg == "--qk-norm") {
            options.qk_norm = next("--qk-norm value");
            if (options.qk_norm != "fused" && options.qk_norm != "composed") {
                throw std::invalid_argument("--qk-norm must be fused or composed");
            }
        } else if (arg == "--help" || arg == "-h") {
            std::printf("Usage: %s [--t-sweep 1,2,...,6] [--warmup N] [--repeat N] "
                        "[--flush-mib N] [--route fused|composed] "
                        "[--qk-norm fused|composed] [--csv-out PATH]\n",
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

    TimedGraph(const TimedGraph&)            = delete;
    TimedGraph& operator=(const TimedGraph&) = delete;

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
        std::size_t all_nodes = 0;
        CUDA_CHECK(cudaGraphGetNodes(graph_, nullptr, &all_nodes));
        if (all_nodes == 0) { throw std::runtime_error("captured layer graph is empty"); }
        body_nodes_ = all_nodes;
    }

    void launch(cudaStream_t stream) const { CUDA_CHECK(cudaGraphLaunch(exec_, stream)); }

    void launch_timed(cudaStream_t stream) const {
        CUDA_CHECK(cudaEventRecord(start_, stream));
        launch(stream);
        CUDA_CHECK(cudaEventRecord(stop_, stream));
    }

    double elapsed_us() const {
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float milliseconds = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_, stop_));
        return static_cast<double>(milliseconds) * 1000.0;
    }

    [[nodiscard]] std::size_t body_nodes() const noexcept { return body_nodes_; }

private:
    cudaGraph_t graph_      = nullptr;
    cudaGraphExec_t exec_   = nullptr;
    cudaEvent_t start_      = nullptr;
    cudaEvent_t stop_       = nullptr;
    std::size_t body_nodes_ = 0;
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
        graph.launch_timed(stream);
        samples.push_back(graph.elapsed_us());
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

struct Resources {
    explicit Resources(std::int32_t max_tokens)
        : input_weight(make_w8_weight(kInputRows, kHidden, 0x03u)),
          output_weight(make_w8_weight(kHidden, kValueRows, 0x00u)),
          control_storage(bench::make_bf16(static_cast<std::size_t>(2 * kValueHeads) * kHidden)),
          control_weight(make_bf16_weight(control_storage, 2 * kValueHeads, kHidden)),
          input_norm(make_constant_bf16(kHidden, 0.0F)),
          gdn_norm(make_constant_bf16(kHeadDim, 1.0F)),
          conv_weight(bench::make_bf16(static_cast<std::size_t>(kConvRows) * 4)),
          a_log(make_constant_f32(kValueHeads, -1.0F)),
          dt_bias(make_constant_f32(kValueHeads, 0.0F)),
          initial_slot(make_constant_i32(kInitialSlot)),
          residual(bench::make_bf16(static_cast<std::size_t>(kHidden) * max_tokens)),
          hidden(static_cast<std::size_t>(kHidden) * max_tokens * sizeof(std::uint16_t)),
          qkv(static_cast<std::size_t>(kConvRows) * max_tokens * sizeof(std::uint16_t)),
          z(static_cast<std::size_t>(kValueRows) * max_tokens * sizeof(std::uint16_t)),
          qkv_conv(static_cast<std::size_t>(kConvRows) * max_tokens * sizeof(std::uint16_t)),
          g(static_cast<std::size_t>(kValueHeads) * max_tokens * sizeof(float)),
          beta(static_cast<std::size_t>(kValueHeads) * max_tokens * sizeof(float)),
          q(static_cast<std::size_t>(kKeyRows) * max_tokens * sizeof(std::uint16_t)),
          k(static_cast<std::size_t>(kKeyRows) * max_tokens * sizeof(std::uint16_t)),
          v(static_cast<std::size_t>(kValueRows) * max_tokens * sizeof(std::uint16_t)),
          q_norm(static_cast<std::size_t>(kKeyRows) * max_tokens * sizeof(std::uint16_t)),
          k_norm(static_cast<std::size_t>(kKeyRows) * max_tokens * sizeof(std::uint16_t)),
          recurrent_out(static_cast<std::size_t>(kValueRows) * max_tokens * sizeof(std::uint16_t)),
          gated_out(static_cast<std::size_t>(kValueRows) * max_tokens * sizeof(std::uint16_t)),
          conv_states(bench::make_zeros(static_cast<std::size_t>(kConvRows) * kConvStateRows *
                                        kSnapshotSlots * sizeof(std::uint16_t))),
          ssm_states(bench::make_zeros(static_cast<std::size_t>(kHeadDim) * kHeadDim * kValueHeads *
                                       kSnapshotSlots * sizeof(float))),
          workspace(std::max<std::size_t>(1, ops::gdn_gating_proj_workspace_bytes(max_tokens))) {}

    static bench::DBuf make_constant_i32(std::int32_t value) {
        bench::DBuf device(sizeof(value));
        CUDA_CHECK(cudaMemcpy(device.p, &value, sizeof(value), cudaMemcpyHostToDevice));
        return device;
    }

    DevicePackedWeight input_weight;
    DevicePackedWeight output_weight;
    bench::DBuf control_storage;
    Weight control_weight;
    bench::DBuf input_norm;
    bench::DBuf gdn_norm;
    bench::DBuf conv_weight;
    bench::DBuf a_log;
    bench::DBuf dt_bias;
    bench::DBuf initial_slot;

    bench::DBuf residual;
    bench::DBuf hidden;
    bench::DBuf qkv;
    bench::DBuf z;
    bench::DBuf qkv_conv;
    bench::DBuf g;
    bench::DBuf beta;
    bench::DBuf q;
    bench::DBuf k;
    bench::DBuf v;
    bench::DBuf q_norm;
    bench::DBuf k_norm;
    bench::DBuf recurrent_out;
    bench::DBuf gated_out;
    bench::DBuf conv_states;
    bench::DBuf ssm_states;
    WorkspaceArena workspace;
};

Result run_case(Resources& resources, bench::DBuf& flush, cudaStream_t stream,
                const Options& options, std::int32_t tokens) {
    Tensor residual(resources.residual.p, DType::BF16, {kHidden, tokens});
    Tensor hidden(resources.hidden.p, DType::BF16, {kHidden, tokens});
    Tensor input_norm(resources.input_norm.p, DType::BF16, {kHidden});
    Tensor qkv(resources.qkv.p, DType::BF16, {kConvRows, tokens});
    Tensor z(resources.z.p, DType::BF16, {kValueRows, tokens});
    Tensor conv_weight(resources.conv_weight.p, DType::BF16, {kConvRows, 4});
    Tensor conv_states(resources.conv_states.p, DType::BF16,
                       {kConvRows, kConvStateRows, kSnapshotSlots});
    Tensor initial_slot(resources.initial_slot.p, DType::I32, {1});
    Tensor qkv_conv(resources.qkv_conv.p, DType::BF16, {kConvRows, tokens});
    Tensor a_log(resources.a_log.p, DType::FP32, {kValueHeads});
    Tensor dt_bias(resources.dt_bias.p, DType::FP32, {kValueHeads});
    Tensor g(resources.g.p, DType::FP32, {kValueHeads, tokens});
    Tensor beta(resources.beta.p, DType::FP32, {kValueHeads, tokens});
    Tensor q(resources.q.p, DType::BF16, {kKeyRows, tokens});
    Tensor k(resources.k.p, DType::BF16, {kKeyRows, tokens});
    Tensor v(resources.v.p, DType::BF16, {kValueRows, tokens});
    Tensor q_norm(resources.q_norm.p, DType::BF16, {kHeadDim, kQkHeads, tokens});
    Tensor k_norm(resources.k_norm.p, DType::BF16, {kHeadDim, kQkHeads, tokens});
    Tensor recurrent_out(resources.recurrent_out.p, DType::BF16, {kHeadDim, kValueHeads, tokens});
    Tensor ssm_states(resources.ssm_states.p, DType::FP32,
                      {kHeadDim, kHeadDim, kValueHeads, kSnapshotSlots});
    Tensor gdn_norm(resources.gdn_norm.p, DType::BF16, {kHeadDim});
    Tensor gated_out(resources.gated_out.p, DType::BF16, {kHeadDim, kValueHeads, tokens});

    const auto layer = [&](cudaStream_t s) {
        ops::rmsnorm(residual, input_norm, kEps, true, hidden, s);
        if (options.route == "fused") {
            ops::gdn_input_proj_conv_snapshot(hidden, resources.input_weight.weight, conv_weight,
                                              conv_states, initial_slot, q, k, v, z,
                                              resources.workspace, s);
        } else {
            ops::gdn_input_proj(hidden, resources.input_weight.weight, qkv, z, resources.workspace,
                                s);
            ops::causal_conv1d_silu_snapshot(qkv, conv_weight, conv_states, initial_slot, qkv_conv,
                                             s);
            ops::extract_bf16_columns(qkv_conv, 0, q, s);
            ops::extract_bf16_columns(qkv_conv, kKeyRows, k, s);
            ops::extract_bf16_columns(qkv_conv, 2 * kKeyRows, v, s);
        }
        ops::gdn_gating_proj(hidden, resources.control_weight, a_log, dt_bias, resources.workspace,
                             g, beta, s);
        Tensor q_recurrent       = q.view({kHeadDim, kQkHeads, tokens});
        Tensor k_recurrent       = k.view({kHeadDim, kQkHeads, tokens});
        const bool fused_qk_norm = options.qk_norm == "fused";
        if (!fused_qk_norm) {
            ops::l2norm(q_recurrent, kEps, q_norm, s);
            ops::l2norm(k_recurrent, kEps, k_norm, s);
            q_recurrent = q_norm;
            k_recurrent = k_norm;
        }
        ops::gated_delta_rule_snapshot(
            q_recurrent, k_recurrent, v.view({kHeadDim, kValueHeads, tokens}), g, beta, kGdnScale,
            fused_qk_norm, resources.workspace, ssm_states, initial_slot, recurrent_out, s);
        ops::gated_rmsnorm(recurrent_out, gdn_norm, z.view({kHeadDim, kValueHeads, tokens}), kEps,
                           gated_out, s);
        ops::linear_add(gated_out.view({kValueRows, tokens}), resources.output_weight.weight,
                        residual, resources.workspace, s);
    };

    // Resolve lazy function attributes and validate the complete production route before capture.
    layer(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    TimedGraph graph;
    graph.capture(stream, layer);
    const Timing timing = measure_cold(graph, flush, stream, options.warmup, options.repeat);

    const auto input_plan = ops::detail::w8_gdn_input_resolve_plan(
        {kHidden, kConvRows, kValueRows, kInputRows, kHidden, tokens});
    const auto control_plan =
        ops::detail::bf16_gdn_gating_resolve_plan({kValueHeads, kHidden, tokens});
    const auto output_plan =
        ops::detail::w8_linear_add_resolve_plan({kHidden, kValueRows, kValueRows, tokens});

    return {tokens,
            graph.body_nodes(),
            options.route + ":" + ops::detail::w8_gdn_input_schedule_name(input_plan.schedule) +
                "+qk-" + options.qk_norm,
            ops::detail::bf16_gdn_gating_schedule_name(control_plan.schedule),
            ops::detail::w8_schedule_name(output_plan.schedule),
            timing};
}

void write_csv(const Options& options, const std::vector<Result>& results) {
    if (options.csv_out.empty()) { return; }
    const std::filesystem::path path(options.csv_out);
    if (!path.parent_path().empty()) { std::filesystem::create_directories(path.parent_path()); }
    std::ofstream out(path);
    out << "T,graph_nodes,cold_median_us,cold_min_us,cold_p95_us,input_route,control_route,"
           "output_route,flush_bytes,warmup,repeat\n";
    for (const Result& result : results) {
        out << result.tokens << ',' << result.graph_nodes << ',' << result.timing.median_us << ','
            << result.timing.min_us << ',' << result.timing.p95_us << ',' << result.input_route
            << ',' << result.control_route << ',' << result.output_route << ','
            << options.flush_bytes << ',' << options.warmup << ',' << options.repeat << '\n';
    }
}

using RouteMember = std::string Result::*;

std::vector<std::string> unique_routes(const std::vector<Result>& results, RouteMember member) {
    std::vector<std::string> routes;
    for (const Result& result : results) {
        const std::string& route = result.*member;
        if (std::find(routes.begin(), routes.end(), route) == routes.end()) {
            routes.push_back(route);
        }
    }
    return routes;
}

std::size_t route_id(const std::vector<std::string>& routes, const std::string& route) {
    const auto found = std::find(routes.begin(), routes.end(), route);
    if (found == routes.end()) { throw std::logic_error("route missing from display index"); }
    return static_cast<std::size_t>(found - routes.begin()) + 1;
}

std::string route_tokens(const std::vector<Result>& results, RouteMember member,
                         const std::string& route) {
    std::vector<std::int32_t> tokens;
    for (const Result& result : results) {
        if (result.*member == route) { tokens.push_back(result.tokens); }
    }
    std::sort(tokens.begin(), tokens.end());
    tokens.erase(std::unique(tokens.begin(), tokens.end()), tokens.end());

    std::string text = "T=";
    for (std::size_t i = 0; i < tokens.size();) {
        const std::size_t begin = i;
        while (i + 1 < tokens.size() && tokens[i + 1] == tokens[i] + 1) { ++i; }
        if (begin != 0) { text += ','; }
        text += std::to_string(tokens[begin]);
        if (i != begin) {
            text += '-';
            text += std::to_string(tokens[i]);
        }
        ++i;
    }
    return text;
}

void print_route_legend(const std::vector<Result>& results, char id_prefix, const char* stage,
                        RouteMember member, const std::vector<std::string>& routes) {
    for (std::size_t i = 0; i < routes.size(); ++i) {
        const std::string tokens = route_tokens(results, member, routes[i]);
        std::printf("  %c%-2zu %-8s %-8s %s\n", id_prefix, i + 1, stage, tokens.c_str(),
                    routes[i].c_str());
    }
}

void print_results(const Options& options, const std::vector<Result>& results) {
    const auto input_routes   = unique_routes(results, &Result::input_route);
    const auto control_routes = unique_routes(results, &Result::control_route);
    const auto output_routes  = unique_routes(results, &Result::output_route);

    std::printf("Qwen3.6-35B-A3B GDN verify layer\n");
    std::printf("  route      %s\n", options.route.c_str());
    std::printf("  qk norm    %s\n", options.qk_norm.c_str());
    std::printf("  execution  CUDA Graph replay\n");
    std::printf("  cache      cold L2 (%zu MiB flush before each sample)\n",
                options.flush_bytes >> 20);
    std::printf("  samples    %d warmup + %d measured per T\n\n", options.warmup, options.repeat);

    std::printf("Latency (us)\n");
    std::printf("  T   nodes    median       min       p95   routes\n");
    std::printf("---  ------  --------  --------  --------  --------\n");
    for (const Result& result : results) {
        std::printf("%3d  %6zu  %8.3f  %8.3f  %8.3f  I%zu/C%zu/O%zu\n", result.tokens,
                    result.graph_nodes, result.timing.median_us, result.timing.min_us,
                    result.timing.p95_us, route_id(input_routes, result.input_route),
                    route_id(control_routes, result.control_route),
                    route_id(output_routes, result.output_route));
    }

    std::printf("\nRoutes\n");
    std::printf("  ID  stage    tokens   implementation\n");
    print_route_legend(results, 'I', "input", &Result::input_route, input_routes);
    print_route_legend(results, 'C', "control", &Result::control_route, control_routes);
    print_route_legend(results, 'O', "output", &Result::output_route, output_routes);
}

} // namespace

int main(int argc, char** argv) {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }
    try {
        const Options options = parse_options(argc, argv);
        const std::int32_t max_tokens =
            *std::max_element(options.t_sweep.begin(), options.t_sweep.end());
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        Resources resources(max_tokens);
        bench::DBuf flush(options.flush_bytes);
        std::vector<Result> results;
        results.reserve(options.t_sweep.size());

        for (const std::int32_t tokens : options.t_sweep) {
            Result result = run_case(resources, flush, stream, options, tokens);
            results.push_back(std::move(result));
        }
        print_results(options, results);
        write_csv(options, results);
        if (!options.csv_out.empty()) {
            std::printf("\nCSV written to %s\n", options.csv_out.c_str());
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_gdn_layer_bench: %s\n", error.what());
        return 1;
    }
}
