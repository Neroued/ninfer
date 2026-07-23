// Complete Qwen3.6 Text full-attention mixer benchmark at the two registered geometries.
//
// The measured composition matches TextContext::attn_mix: hidden RMSNorm, exact target input
// projections, Q/K RMSNorm, partial Text RoPE, append-and-attend GQA, sigmoid output gating, and
// output projection plus residual. The following post-attention norm and post-mixer are excluded.
// Each (target, KV dtype, context, T) point owns one captured CUDA Graph. Device events bracket
// graph replay, and a 256 MiB L2 flush precedes every sample outside the timed interval.
//
// Default production-representative matrix:
//   ./build/bench/ninfer_attention_layer_bench
// Focused examples:
//   ./build/bench/ninfer_attention_layer_bench --geometry 35b --kv-dtype bf16 \
//       --context 8192 --t-sweep 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16
//   ./build/bench/ninfer_attention_layer_bench --geometry 27b --kv-dtype int8 \
//       --context 128,8192 --repeat 80 --csv-out profiles/bench/attention-layer.csv

#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/gqa_attention.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/sigmoid_mul.h"

#include "core/device.h"
#include "core/kv_cache.h"
#include "ninfer_bench_common.h"
#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_plan.h"
#include "ops/attn_input_proj/w8/w8_attn_input_plan.h"
#include "ops/launcher/gqa_attention.h"
#include "ops/linear_add/q5/q5_linear_add_plan.h"
#include "ops/linear_add/w8/w8_linear_add_plan.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace ninfer;

namespace {

constexpr std::int32_t kHeadDim     = 256;
constexpr std::int32_t kRotaryDim   = 64;
constexpr float kRopeTheta          = 1.0e7F;
constexpr float kRmsEps             = 1.0e-6F;
constexpr float kAttentionScale     = 0.0625F;
constexpr std::size_t kDefaultFlush = 256ULL << 20;

enum class GeometrySelection { All, B27, B35 };
enum class KvSelection { TargetDefault, Bf16, Int8 };
enum class AttentionSelection { Auto, PromptControl };

struct Options {
    GeometrySelection geometry            = GeometrySelection::All;
    KvSelection kv                        = KvSelection::TargetDefault;
    AttentionSelection attention          = AttentionSelection::Auto;
    std::vector<std::int32_t> contexts    = {128, 8192};
    std::vector<std::int32_t> token_sweep = {1, 2, 3, 4, 5, 6};
    int warmup                            = 5;
    int repeat                            = 40;
    std::size_t flush_bytes               = kDefaultFlush;
    std::string csv_out;
};

struct Timing {
    double median_us = 0.0;
    double min_us    = 0.0;
    double p95_us    = 0.0;
};

struct Result {
    std::string geometry;
    std::string kv_dtype;
    std::int32_t context    = 0;
    std::int32_t tokens     = 0;
    std::size_t graph_nodes = 0;
    std::string input_route;
    std::string attention_route;
    std::string output_route;
    Timing timing;
};

struct DevicePackedWeight {
    explicit DevicePackedWeight(std::size_t bytes) : storage(bytes) {}

    bench::DBuf storage;
    Weight weight{};
};

__global__ void fill_fp16_one(std::uint8_t* scales, std::uint64_t groups) {
    const std::uint64_t begin = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t step  = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t group = begin; group < groups; group += step) {
        scales[2 * group]     = 0x00U;
        scales[2 * group + 1] = 0x3cU;
    }
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

DevicePackedWeight make_w8_weight(std::int32_t rows, std::int32_t cols, std::uint8_t code_byte) {
    constexpr std::int32_t group = 32;
    if (rows <= 0 || cols <= 0 || cols % group != 0) {
        throw std::invalid_argument("benchmark W8 shape is invalid");
    }
    const std::uint64_t groups =
        static_cast<std::uint64_t>(rows) * static_cast<std::uint64_t>(cols / group);
    const std::uint64_t code_bytes   = groups * group;
    const std::uint64_t scale_offset = align_up(code_bytes, 256);
    const std::uint64_t scale_bytes  = groups * 2;
    DevicePackedWeight packed(static_cast<std::size_t>(scale_offset + scale_bytes));
    CUDA_CHECK(cudaMemset(packed.storage.p, 0, packed.storage.bytes));
    CUDA_CHECK(cudaMemset(packed.storage.p, code_byte, code_bytes));

    constexpr int block = 256;
    const int grid      = static_cast<int>(
        std::min<std::uint64_t>(65535, std::max<std::uint64_t>(1, (groups + block - 1) / block)));
    fill_fp16_one<<<grid, block>>>(static_cast<std::uint8_t*>(packed.storage.p) + scale_offset,
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

DevicePackedWeight make_qx_weight(QType qtype, std::int32_t rows, std::int32_t cols,
                                  std::uint8_t code_byte) {
    constexpr std::int32_t group = 64;
    if ((qtype != QType::Q4G64_F16S && qtype != QType::Q5G64_F16S) || rows <= 0 || cols <= 0 ||
        cols % group != 0) {
        throw std::invalid_argument("benchmark Q4/Q5 shape is invalid");
    }
    const std::uint64_t groups =
        static_cast<std::uint64_t>(rows) * static_cast<std::uint64_t>(cols / group);
    const std::uint64_t low_bytes    = groups * 32;
    const std::uint64_t high_bytes   = qtype == QType::Q5G64_F16S ? groups * 8 : 0;
    const std::uint64_t high_offset  = align_up(low_bytes, 256);
    const std::uint64_t scale_offset = high_offset + align_up(high_bytes, 256);
    const std::uint64_t scale_bytes  = groups * 2;
    DevicePackedWeight packed(static_cast<std::size_t>(scale_offset + scale_bytes));
    CUDA_CHECK(cudaMemset(packed.storage.p, 0, packed.storage.bytes));
    CUDA_CHECK(cudaMemset(packed.storage.p, code_byte, low_bytes));

    constexpr int block = 256;
    const int grid      = static_cast<int>(
        std::min<std::uint64_t>(65535, std::max<std::uint64_t>(1, (groups + block - 1) / block)));
    fill_fp16_one<<<grid, block>>>(static_cast<std::uint8_t*>(packed.storage.p) + scale_offset,
                                   groups);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    Weight& weight          = packed.weight;
    weight.payload          = packed.storage.p;
    weight.payload_bytes    = packed.storage.bytes;
    weight.high_plane_bytes = high_bytes;
    weight.qtype            = qtype;
    weight.layout           = QuantLayout::RowSplit;
    weight.group_size       = group;
    weight.qdata            = packed.storage.p;
    weight.qhigh =
        high_bytes == 0 ? nullptr : static_cast<std::uint8_t*>(packed.storage.p) + high_offset;
    weight.scales          = static_cast<std::uint8_t*>(packed.storage.p) + scale_offset;
    weight.n               = rows;
    weight.k               = cols;
    weight.group           = group;
    weight.scale_dtype     = DType::FP16;
    weight.ndim            = 2;
    weight.shape[0]        = rows;
    weight.shape[1]        = cols;
    weight.padded_shape[0] = rows;
    weight.padded_shape[1] = cols;
    return packed;
}

bench::DBuf make_constant_bf16(std::size_t elements, float value) {
    std::vector<std::uint16_t> host(elements, bench::f32_to_bf16(value));
    bench::DBuf device(elements * sizeof(std::uint16_t));
    CUDA_CHECK(cudaMemcpy(device.p, host.data(), device.bytes, cudaMemcpyHostToDevice));
    return device;
}

bench::DBuf make_i32_sequence(std::int32_t start, std::int32_t count) {
    std::vector<std::int32_t> host(static_cast<std::size_t>(count));
    for (std::int32_t i = 0; i < count; ++i) { host[static_cast<std::size_t>(i)] = start + i; }
    bench::DBuf device(host.size() * sizeof(std::int32_t));
    CUDA_CHECK(cudaMemcpy(device.p, host.data(), device.bytes, cudaMemcpyHostToDevice));
    return device;
}

std::vector<std::int32_t> parse_i32_list(std::string_view raw, bool allow_zero, const char* label) {
    std::vector<std::int32_t> values;
    std::size_t begin = 0;
    while (begin < raw.size()) {
        const std::size_t end = raw.find(',', begin);
        const std::string token(
            raw.substr(begin, end == std::string_view::npos ? raw.size() - begin : end - begin));
        if (token.empty()) { throw std::invalid_argument(std::string("empty ") + label + " item"); }
        const long long value   = std::stoll(token);
        const long long minimum = allow_zero ? 0 : 1;
        if (value < minimum || value > std::numeric_limits<std::int32_t>::max()) {
            throw std::invalid_argument(std::string(label) + " item is out of range");
        }
        values.push_back(static_cast<std::int32_t>(value));
        if (end == std::string_view::npos) { break; }
        begin = end + 1;
    }
    if (values.empty()) { throw std::invalid_argument(std::string(label) + " must not be empty"); }
    return values;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        const auto next = [&](const char* label) -> std::string_view {
            if (++i >= argc) { throw std::invalid_argument(std::string("missing ") + label); }
            return argv[i];
        };
        if (arg == "--geometry") {
            const std::string_view value = next("--geometry value");
            if (value == "all")
                options.geometry = GeometrySelection::All;
            else if (value == "27b")
                options.geometry = GeometrySelection::B27;
            else if (value == "35b")
                options.geometry = GeometrySelection::B35;
            else
                throw std::invalid_argument("--geometry must be all, 27b, or 35b");
        } else if (arg == "--kv-dtype") {
            const std::string_view value = next("--kv-dtype value");
            if (value == "default")
                options.kv = KvSelection::TargetDefault;
            else if (value == "bf16")
                options.kv = KvSelection::Bf16;
            else if (value == "int8")
                options.kv = KvSelection::Int8;
            else
                throw std::invalid_argument("--kv-dtype must be default, bf16, or int8");
        } else if (arg == "--context") {
            options.contexts = parse_i32_list(next("--context value"), true, "--context");
        } else if (arg == "--attention-route") {
            const std::string_view value = next("--attention-route value");
            if (value == "auto")
                options.attention = AttentionSelection::Auto;
            else if (value == "prompt-control")
                options.attention = AttentionSelection::PromptControl;
            else
                throw std::invalid_argument("--attention-route must be auto or prompt-control");
        } else if (arg == "--t-sweep") {
            options.token_sweep = parse_i32_list(next("--t-sweep value"), false, "--t-sweep");
        } else if (arg == "--warmup") {
            options.warmup = std::stoi(std::string(next("--warmup value")));
        } else if (arg == "--repeat") {
            options.repeat = std::stoi(std::string(next("--repeat value")));
        } else if (arg == "--flush-mib") {
            const long long mib = std::stoll(std::string(next("--flush-mib value")));
            if (mib <= 0) { throw std::invalid_argument("--flush-mib must be positive"); }
            options.flush_bytes = static_cast<std::size_t>(mib) << 20;
        } else if (arg == "--csv-out") {
            options.csv_out = next("--csv-out path");
        } else if (arg == "--help" || arg == "-h") {
            std::printf("Usage: %s [--geometry all|27b|35b] [--kv-dtype default|bf16|int8] "
                        "[--attention-route auto|prompt-control] "
                        "[--context 128,8192] [--t-sweep 1,2,...,16] [--warmup N] [--repeat N] "
                        "[--flush-mib N] [--csv-out PATH]\n",
                        argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(arg));
        }
    }
    if (options.warmup < 0 || options.repeat <= 0) {
        throw std::invalid_argument("--warmup must be nonnegative and --repeat positive");
    }
    for (const std::int32_t tokens : options.token_sweep) {
        if (tokens > 16) { throw std::invalid_argument("--t-sweep values must be in [1,16]"); }
    }
    const std::int32_t max_tokens =
        *std::max_element(options.token_sweep.begin(), options.token_sweep.end());
    for (const std::int32_t context : options.contexts) {
        if (context > std::numeric_limits<std::int32_t>::max() - max_tokens) {
            throw std::invalid_argument("--context + max T exceeds int32");
        }
    }
    return options;
}

const char* kv_dtype_name(DType dtype) { return dtype == DType::I8 ? "int8" : "bf16"; }

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
        CUDA_CHECK(cudaGraphGetNodes(graph_, nullptr, &body_nodes_));
        if (body_nodes_ == 0) { throw std::runtime_error("captured attention graph is empty"); }
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

struct Geometry27 {
    static constexpr const char* name         = "27b";
    static constexpr std::int32_t hidden      = 5120;
    static constexpr std::int32_t query_heads = 24;
    static constexpr std::int32_t kv_heads    = 4;
    static constexpr std::int32_t query_rows  = query_heads * kHeadDim;
    static constexpr std::int32_t kv_rows     = kv_heads * kHeadDim;
    static constexpr std::int32_t parent_rows = query_rows + kv_rows;
    static constexpr DType default_kv_dtype   = DType::I8;

    static DevicePackedWeight make_input0() {
        return make_qx_weight(QType::Q4G64_F16S, parent_rows, hidden, 0x11U);
    }

    static std::optional<DevicePackedWeight> make_input1() {
        return make_qx_weight(QType::Q5G64_F16S, parent_rows, hidden, 0x11U);
    }

    static DevicePackedWeight make_output() {
        return make_qx_weight(QType::Q5G64_F16S, hidden, query_rows, 0x00U);
    }

    static void input_projection(const Tensor& hidden_tensor, const DevicePackedWeight& input0,
                                 const std::optional<DevicePackedWeight>& input1, Tensor& query,
                                 Tensor& gate, Tensor& key, Tensor& value,
                                 WorkspaceArena& workspace, cudaStream_t stream) {
        if (!input1.has_value()) { throw std::logic_error("27B gate/value weight is missing"); }
        ops::attn_input_proj(hidden_tensor, input0.weight, input1->weight, query, gate, key, value,
                             workspace, stream);
    }

    static std::string input_route(std::int32_t tokens) {
        const auto plan = ops::detail::q4_q5_attn_input_resolve_plan(
            {hidden, query_rows, kv_rows, hidden, tokens});
        return ops::detail::q4_q5_attn_input_schedule_name(plan.schedule);
    }

    static std::string output_route(std::int32_t tokens) {
        const auto plan =
            ops::detail::q5_linear_add_resolve_plan({hidden, query_rows, query_rows, tokens});
        return ops::detail::q5_linear_add_schedule_name(plan.schedule);
    }
};

struct Geometry35 {
    static constexpr const char* name         = "35b";
    static constexpr std::int32_t hidden      = 2048;
    static constexpr std::int32_t query_heads = 16;
    static constexpr std::int32_t kv_heads    = 2;
    static constexpr std::int32_t query_rows  = query_heads * kHeadDim;
    static constexpr std::int32_t kv_rows     = kv_heads * kHeadDim;
    static constexpr std::int32_t parent_rows = 2 * query_rows + 2 * kv_rows;
    static constexpr DType default_kv_dtype   = DType::BF16;

    static DevicePackedWeight make_input0() { return make_w8_weight(parent_rows, hidden, 0x01U); }

    static std::optional<DevicePackedWeight> make_input1() { return std::nullopt; }

    static DevicePackedWeight make_output() { return make_w8_weight(hidden, query_rows, 0x00U); }

    static void input_projection(const Tensor& hidden_tensor, const DevicePackedWeight& input0,
                                 const std::optional<DevicePackedWeight>&, Tensor& query,
                                 Tensor& gate, Tensor& key, Tensor& value,
                                 WorkspaceArena& workspace, cudaStream_t stream) {
        ops::attn_input_proj(hidden_tensor, input0.weight, query, gate, key, value, workspace,
                             stream);
    }

    static std::string input_route(std::int32_t tokens) {
        const auto plan = ops::detail::w8_attn_input_resolve_plan(
            {hidden, query_rows, kv_rows, parent_rows, hidden, tokens});
        return ops::detail::w8_attn_input_schedule_name(plan.schedule);
    }

    static std::string output_route(std::int32_t tokens) {
        const auto plan =
            ops::detail::w8_linear_add_resolve_plan({hidden, query_rows, query_rows, tokens});
        return ops::detail::w8_schedule_name(plan.schedule);
    }
};

template <class Geometry>
struct Resources {
    Resources(std::int32_t max_tokens, std::size_t workspace_bytes)
        : input0(Geometry::make_input0()), input1(Geometry::make_input1()),
          output(Geometry::make_output()), input_norm(make_constant_bf16(Geometry::hidden, 0.0F)),
          query_norm(make_constant_bf16(kHeadDim, 0.0F)),
          key_norm(make_constant_bf16(kHeadDim, 0.0F)),
          residual(bench::make_bf16(static_cast<std::size_t>(Geometry::hidden) * max_tokens)),
          hidden(static_cast<std::size_t>(Geometry::hidden) * max_tokens * sizeof(std::uint16_t)),
          query(static_cast<std::size_t>(Geometry::query_rows) * max_tokens *
                sizeof(std::uint16_t)),
          gate(static_cast<std::size_t>(Geometry::query_rows) * max_tokens * sizeof(std::uint16_t)),
          key(static_cast<std::size_t>(Geometry::kv_rows) * max_tokens * sizeof(std::uint16_t)),
          value(static_cast<std::size_t>(Geometry::kv_rows) * max_tokens * sizeof(std::uint16_t)),
          query_transformed(static_cast<std::size_t>(Geometry::query_rows) * max_tokens *
                            sizeof(std::uint16_t)),
          key_transformed(static_cast<std::size_t>(Geometry::kv_rows) * max_tokens *
                          sizeof(std::uint16_t)),
          attention(static_cast<std::size_t>(Geometry::query_rows) * max_tokens *
                    sizeof(std::uint16_t)),
          workspace(std::max<std::size_t>(1, workspace_bytes)) {}

    DevicePackedWeight input0;
    std::optional<DevicePackedWeight> input1;
    DevicePackedWeight output;
    bench::DBuf input_norm;
    bench::DBuf query_norm;
    bench::DBuf key_norm;
    bench::DBuf residual;
    bench::DBuf hidden;
    bench::DBuf query;
    bench::DBuf gate;
    bench::DBuf key;
    bench::DBuf value;
    bench::DBuf query_transformed;
    bench::DBuf key_transformed;
    bench::DBuf attention;
    WorkspaceArena workspace;
};

DType select_kv_dtype(KvSelection selection, DType target_default) {
    switch (selection) {
    case KvSelection::TargetDefault:
        return target_default;
    case KvSelection::Bf16:
        return DType::BF16;
    case KvSelection::Int8:
        return DType::I8;
    }
    throw std::logic_error("unknown KV selection");
}

template <class Geometry>
Result run_case(Resources<Geometry>& resources, KVCache& cache, bench::DBuf& flush,
                cudaStream_t stream, const Options& options, std::int32_t context,
                std::int32_t tokens) {
    bench::DBuf position_storage = make_i32_sequence(context, tokens);

    Tensor residual(resources.residual.p, DType::BF16, {Geometry::hidden, tokens});
    Tensor hidden(resources.hidden.p, DType::BF16, {Geometry::hidden, tokens});
    Tensor input_norm(resources.input_norm.p, DType::BF16, {Geometry::hidden});
    Tensor query(resources.query.p, DType::BF16, {kHeadDim, Geometry::query_heads, tokens});
    Tensor gate(resources.gate.p, DType::BF16, {kHeadDim, Geometry::query_heads, tokens});
    Tensor key(resources.key.p, DType::BF16, {kHeadDim, Geometry::kv_heads, tokens});
    Tensor value(resources.value.p, DType::BF16, {kHeadDim, Geometry::kv_heads, tokens});
    Tensor query_norm(resources.query_norm.p, DType::BF16, {kHeadDim});
    Tensor key_norm(resources.key_norm.p, DType::BF16, {kHeadDim});
    Tensor query_transformed(resources.query_transformed.p, DType::BF16,
                             {kHeadDim, Geometry::query_heads, tokens});
    Tensor key_transformed(resources.key_transformed.p, DType::BF16,
                           {kHeadDim, Geometry::kv_heads, tokens});
    Tensor attention(resources.attention.p, DType::BF16, {kHeadDim, Geometry::query_heads, tokens});
    Tensor positions(position_storage.p, DType::I32, {tokens});
    Tensor query_flat = query.view({Geometry::query_rows, tokens});
    Tensor gate_flat  = gate.view({Geometry::query_rows, tokens});
    Tensor key_flat   = key.view({Geometry::kv_rows, tokens});
    Tensor value_flat = value.view({Geometry::kv_rows, tokens});

    const auto layer = [&](cudaStream_t layer_stream) {
        ops::rmsnorm(residual, input_norm, kRmsEps, true, hidden, layer_stream);
        Geometry::input_projection(hidden, resources.input0, resources.input1, query_flat,
                                   gate_flat, key_flat, value_flat, resources.workspace,
                                   layer_stream);
        ops::rmsnorm(query, query_norm, kRmsEps, true, query_transformed, layer_stream);
        ops::rmsnorm(key, key_norm, kRmsEps, true, key_transformed, layer_stream);
        ops::rope(positions, kRotaryDim, kRopeTheta, query_transformed, key_transformed,
                  layer_stream);
        const auto visible = static_cast<std::uint32_t>(context + tokens);
        if (options.attention == AttentionSelection::PromptControl) {
            ops::detail::gqa_attention_prompt_launch(query_transformed, key_transformed, value,
                                                     positions, kAttentionScale,
                                                     cache.layer_view(0), attention, layer_stream);
        } else {
            ops::gqa_attention(query_transformed, key_transformed, value, positions,
                               kAttentionScale, cache.layer_view(0), {visible, visible},
                               resources.workspace, attention, layer_stream);
        }
        ops::sigmoid_mul(gate, attention, layer_stream);
        ops::linear_add(attention.view({Geometry::query_rows, tokens}), resources.output.weight,
                        residual, resources.workspace, layer_stream);
    };

    // Resolve route validation and lazy CUDA attributes before graph capture.
    layer(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    TimedGraph graph;
    graph.capture(stream, layer);
    const Timing timing = measure_cold(graph, flush, stream, options.warmup, options.repeat);
    const std::uint32_t visible = static_cast<std::uint32_t>(context + tokens);
    const auto selected_route =
        ops::detail::gqa_attention_resolve_route(Geometry::query_heads, tokens, {visible, visible});
    const char* route = options.attention == AttentionSelection::PromptControl
                            ? "prompt_control"
                            : ops::detail::gqa_attention_route_name(selected_route);
    const std::string attention_route =
        std::string("gqa.") + route + ".append." + kv_dtype_name(cache.dtype);
    return {Geometry::name,
            kv_dtype_name(cache.dtype),
            context,
            tokens,
            graph.body_nodes(),
            Geometry::input_route(tokens),
            attention_route,
            Geometry::output_route(tokens),
            timing};
}

template <class Geometry>
void run_geometry(const Options& options, cudaStream_t stream, bench::DBuf& flush,
                  std::vector<Result>& results) {
    const std::int32_t max_tokens =
        *std::max_element(options.token_sweep.begin(), options.token_sweep.end());
    const std::int32_t max_context =
        *std::max_element(options.contexts.begin(), options.contexts.end()) + max_tokens;
    const DType kv_dtype = select_kv_dtype(options.kv, Geometry::default_kv_dtype);

    LayoutBuilder cache_builder;
    const KVCacheLayout cache_layout =
        plan_kv_cache(cache_builder, 1, static_cast<std::uint32_t>(max_context), Geometry::kv_heads,
                      kHeadDim, kv_dtype, kv_dtype == DType::I8 ? kKvQuantGroup : 0);
    DeviceArena cache_arena(cache_builder.finish(256));
    KVCache cache(cache_arena.alloc_bytes(cache_arena.capacity()), cache_layout);
    CUDA_CHECK(cudaMemset(cache_arena.base(), 0, cache_arena.capacity()));

    const std::size_t workspace_bytes = std::max(
        ops::gqa_attention_workspace_bytes(Geometry::query_heads, max_tokens),
        ops::linear_add_workspace_bytes(Geometry::hidden, Geometry::query_rows, max_tokens));
    Resources<Geometry> resources(max_tokens, workspace_bytes);

    for (const std::int32_t context : options.contexts) {
        for (const std::int32_t tokens : options.token_sweep) {
            results.push_back(run_case(resources, cache, flush, stream, options, context, tokens));
        }
    }
}

void write_csv(const Options& options, const std::vector<Result>& results) {
    if (options.csv_out.empty()) { return; }
    const std::filesystem::path path(options.csv_out);
    if (!path.parent_path().empty()) { std::filesystem::create_directories(path.parent_path()); }
    std::ofstream out(path);
    out << "geometry,kv_dtype,context,T,graph_nodes,cold_median_us,cold_min_us,cold_p95_us,"
           "input_route,attention_route,output_route,flush_bytes,warmup,repeat\n";
    for (const Result& result : results) {
        out << result.geometry << ',' << result.kv_dtype << ',' << result.context << ','
            << result.tokens << ',' << result.graph_nodes << ',' << result.timing.median_us << ','
            << result.timing.min_us << ',' << result.timing.p95_us << ',' << result.input_route
            << ',' << result.attention_route << ',' << result.output_route << ','
            << options.flush_bytes << ',' << options.warmup << ',' << options.repeat << '\n';
    }
}

void print_results(const Options& options, const std::vector<Result>& results) {
    std::printf("Qwen3.6 Text full-attention mixer\n");
    std::printf("  boundary   input RMSNorm through output projection + residual\n");
    std::printf("  execution  CUDA Graph replay\n");
    std::printf("  cache      cold L2 (%zu MiB flush before each sample)\n",
                options.flush_bytes >> 20);
    std::printf("  samples    %d warmup + %d measured per point\n\n", options.warmup,
                options.repeat);
    std::printf("Latency (us)\n");
    std::printf(" target  kv       context   T  nodes    median       min       p95\n");
    std::printf("-------  -----  ----------  --  -----  --------  --------  --------\n");
    for (const Result& result : results) {
        std::printf("%7s  %5s  %10d  %2d  %5zu  %8.3f  %8.3f  %8.3f\n", result.geometry.c_str(),
                    result.kv_dtype.c_str(), result.context, result.tokens, result.graph_nodes,
                    result.timing.median_us, result.timing.min_us, result.timing.p95_us);
    }

    std::printf("\nRoutes\n");
    for (const Result& result : results) {
        std::printf("  %s/%s C=%d T=%d  input=%s  attention=%s  output=%s\n",
                    result.geometry.c_str(), result.kv_dtype.c_str(), result.context, result.tokens,
                    result.input_route.c_str(), result.attention_route.c_str(),
                    result.output_route.c_str());
    }
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
        cudaStream_t stream   = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        bench::DBuf flush(options.flush_bytes);
        std::vector<Result> results;
        results.reserve(options.contexts.size() * options.token_sweep.size() * 2);

        if (options.geometry != GeometrySelection::B35) {
            run_geometry<Geometry27>(options, stream, flush, results);
        }
        if (options.geometry != GeometrySelection::B27) {
            run_geometry<Geometry35>(options, stream, flush, results);
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
        std::fprintf(stderr, "ninfer_attention_layer_bench: %s\n", error.what());
        return 1;
    }
}
