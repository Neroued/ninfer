// Public GDN and production-stage timing. Private launchers also measure the prefill
// recurrent/chunked crossover, independently of decode/ReplaySSM.
// Each sample is a cold-L2 CUDA Graph replay with the flush outside its timer.
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/l2norm.h"
#include "ninfer_bench_common.h"
#include "ops/linear_attention/gated_delta_net/chunked/launch.h"
#include "ops/linear_attention/gated_delta_net/launch.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

namespace gated_delta_net_detail = ninfer::ops::detail::gated_delta_net;
namespace chunked_detail         = ninfer::ops::detail::gated_delta_net::chunked;

constexpr std::int32_t kDefaultQkHeads    = 16;
constexpr std::int32_t kDefaultValueHeads = 48;
constexpr std::int32_t kDefaultTokens     = 1024;
constexpr std::size_t kDefaultFlushBytes  = 256ULL << 20;
constexpr float kQkNormEpsilon            = 1.0e-6F;

enum class Mode {
    Running,
    BatchUpdate,
    ChunkedOnly,
    ForceChunked,
    RecurrentOnly,
};

struct Options {
    Mode mode                = Mode::Running;
    bool mode_explicit       = false;
    bool tokens_explicit     = false;
    bool sweep               = false;
    bool breakdown           = false;
    bool csv                 = false;
    bool help                = false;
    std::int32_t qk_heads    = kDefaultQkHeads;
    std::int32_t value_heads = kDefaultValueHeads;
    std::int32_t tokens      = kDefaultTokens;
    std::int32_t batch       = 1;
    int warmup               = 20;
    int repeat               = 100;
    std::size_t flush_bytes  = kDefaultFlushBytes;
    std::string qk_norm      = "fused";
};

struct Problem {
    std::int32_t qk_heads;
    std::int32_t value_heads;
    std::int32_t tokens;
    std::int32_t batch = 1;
};

struct GraphMeasurement {
    ColdTiming timing;
    std::size_t graph_nodes;
};

struct BenchRow {
    const char* state_form    = "";
    const char* normalization = "";
    std::string implementation;
    std::int32_t tokens               = 0;
    std::int32_t full_chunks          = 0;
    std::int32_t tail_tokens          = 0;
    std::size_t workspace_bytes       = 0;
    std::size_t graph_nodes           = 0;
    double logical_bytes              = 0.0;
    double traffic_bytes              = 0.0;
    double intermediate_traffic_bytes = 0.0;
    double stage_share_pct            = -1.0;
    double relative_to_e2e_pct        = -1.0;
    ColdTiming timing{};
};

struct TrafficBytes {
    double total        = 0.0;
    double intermediate = 0.0;
};

[[noreturn]] void fail(const std::string& message) { throw std::invalid_argument(message); }

std::int32_t parse_integer(const char* flag, const char* text, std::int32_t minimum) {
    errno            = 0;
    char* end        = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (errno != 0 || text == end || *end != '\0' || value < minimum ||
        value > static_cast<long>(INT32_MAX)) {
        fail(std::string("invalid value for ") + flag + ": " + text);
    }
    return static_cast<std::int32_t>(value);
}

void set_mode(Options& options, Mode mode, const char* flag) {
    if (options.mode_explicit && options.mode != mode) {
        fail(std::string(flag) + " cannot be combined with another benchmark mode");
    }
    options.mode          = mode;
    options.mode_explicit = true;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        const auto take = [&](const char* flag) -> const char* {
            if (++i >= argc) { fail(std::string("missing value for ") + flag); }
            return argv[i];
        };

        if (arg == "--running") {
            set_mode(options, Mode::Running, "--running");
        } else if (arg == "--batch-update") {
            set_mode(options, Mode::BatchUpdate, "--batch-update");
        } else if (arg == "--chunked-only") {
            set_mode(options, Mode::ChunkedOnly, "--chunked-only");
        } else if (arg == "--force-chunked") {
            set_mode(options, Mode::ForceChunked, "--force-chunked");
        } else if (arg == "--recurrent-only") {
            set_mode(options, Mode::RecurrentOnly, "--recurrent-only");
        } else if (arg == "--tokens") {
            options.tokens          = parse_integer("--tokens", take("--tokens"), 1);
            options.tokens_explicit = true;
        } else if (arg == "--sweep") {
            options.sweep = true;
        } else if (arg == "--qk-heads") {
            options.qk_heads = parse_integer("--qk-heads", take("--qk-heads"), 1);
        } else if (arg == "--value-heads") {
            options.value_heads = parse_integer("--value-heads", take("--value-heads"), 1);
        } else if (arg == "--batch") {
            options.batch = parse_integer("--batch", take("--batch"), 1);
        } else if (arg == "--qk-norm") {
            options.qk_norm = take("--qk-norm");
            if (options.qk_norm != "fused" && options.qk_norm != "composed") {
                fail("--qk-norm must be fused or composed");
            }
        } else if (arg == "--breakdown") {
            options.breakdown = true;
        } else if (arg == "--warmup") {
            options.warmup = parse_integer("--warmup", take("--warmup"), 0);
        } else if (arg == "--repeat") {
            options.repeat = parse_integer("--repeat", take("--repeat"), 1);
        } else if (arg == "--flush-mib") {
            const std::int32_t mib = parse_integer("--flush-mib", take("--flush-mib"), 1);
            options.flush_bytes    = static_cast<std::size_t>(mib) << 20;
        } else if (arg == "--csv") {
            options.csv = true;
        } else if (arg == "--help" || arg == "-h") {
            options.help = true;
        } else {
            fail("unknown argument: " + std::string(arg));
        }
    }

    if (options.sweep && options.tokens_explicit) {
        fail("--tokens and --sweep are mutually exclusive");
    }
    if (!gated_delta_net_detail::are_head_counts_valid(options.qk_heads, options.value_heads)) {
        fail("value heads must be at least q/k heads and divisible by them");
    }
    if (options.breakdown && options.mode == Mode::BatchUpdate) {
        fail("--breakdown requires a prefill mode");
    }
    if (options.qk_norm == "composed" && options.mode != Mode::BatchUpdate) {
        fail("--qk-norm composed is a batch-update comparison");
    }
    if (options.batch > 8 || (options.mode != Mode::BatchUpdate && options.batch != 1) ||
        (options.batch > 1 && options.qk_norm == "composed")) {
        fail("batch metadata is valid only for a fused-normalization batch update");
    }
    return options;
}

void print_help(const char* program) {
    std::printf(
        "Usage: %s [mode] [options]\n"
        "\n"
        "Modes (default: --running):\n"
        "  --running          public running-state Gated DeltaNet\n"
        "  --batch-update     public selected-slot Gated DeltaNet update at T=1\n"
        "  --chunked-only     forced chunked with pre-normalized BF16 inputs\n"
        "  --force-chunked    forced chunked with fused Q/K normalization\n"
        "  --recurrent-only   prefill recurrence with fused Q/K normalization\n"
        "  --breakdown        also time prepare and recurrence/output separately\n"
        "\n"
        "Workload:\n"
        "  --tokens N         exact token extent (running/chunked default: 1024)\n"
        "  --sweep            short prefill boundaries through T=4096\n"
        "  --qk-heads N       Q/K heads (default: 16)\n"
        "  --value-heads N    divisible value heads >= Q/K heads (default: 48)\n"
        "  --batch B          exact batch-update batch in [1,8] (default: 1)\n"
        "  --qk-norm MODE     batch-update normalization: fused or composed (default: fused)\n"
        "\n"
        "Measurement:\n"
        "  --warmup N         cold-L2 graph warmups per case (default: 20)\n"
        "  --repeat N         measured cold-L2 graph replays per case (default: 100)\n"
        "  --flush-mib N      L2 flush allocation in MiB (default: 256)\n"
        "  --csv              emit CSV instead of human-readable rows\n"
        "  -h, --help         show this help\n"
        "\n"
        "State/head dimension 128 is fixed; running/chunked use batch 1.\n",
        program);
}

std::vector<std::int32_t> token_values(const Options& options) {
    if (options.tokens_explicit) { return {options.tokens}; }
    if (options.mode == Mode::BatchUpdate) { return {1}; }
    if (!options.sweep) { return {kDefaultTokens}; }
    return {1, 8, 12, 15, 16, 17, 24, 31, 32, 33, 63, 64, 65, 128, 1024, 4096};
}

void validate_tokens(const Options& options, std::int32_t tokens) {
    if (options.mode == Mode::BatchUpdate && tokens != 1) {
        fail("batch-update benchmark requires T=1");
    }
}

float gated_delta_net_scale() {
    return 1.0F / std::sqrt(static_cast<float>(gated_delta_net_detail::kStateDim));
}

DeviceBuffer make_constant_f32(std::size_t elements, float value) {
    std::vector<float> host(elements, value);
    DeviceBuffer device(elements * sizeof(float));
    device.copy_from_host(host.data(), device.bytes);
    return device;
}

DeviceBuffer make_varied_bf16(std::size_t elements, std::uint32_t seed) {
    std::vector<std::uint16_t> host(elements);
    std::uint32_t state = seed;
    for (std::uint16_t& value : host) {
        state         = state * 1664525U + 1013904223U;
        const float u = static_cast<float>((state >> 8) & 0x00ffffffU) * (1.0F / 16777216.0F);
        value         = f32_to_bf16(2.0F * u - 1.0F);
    }
    DeviceBuffer device(elements * sizeof(std::uint16_t));
    device.copy_from_host(host.data(), device.bytes);
    return device;
}

DeviceBuffer make_normalized_bf16(std::size_t rows, std::uint32_t seed) {
    const std::size_t state_dim = static_cast<std::size_t>(gated_delta_net_detail::kStateDim);
    std::vector<float> values(rows * state_dim);
    std::uint32_t state = seed;
    for (float& value : values) {
        state         = state * 1664525U + 1013904223U;
        const float u = static_cast<float>((state >> 8) & 0x00ffffffU) * (1.0F / 16777216.0F);
        value         = 2.0F * u - 1.0F;
    }

    std::vector<std::uint16_t> host(values.size());
    for (std::size_t row = 0; row < rows; ++row) {
        double squared_sum = 0.0;
        for (std::size_t column = 0; column < state_dim; ++column) {
            const float value = values[row * state_dim + column];
            squared_sum += static_cast<double>(value) * value;
        }
        const float inverse = static_cast<float>(1.0 / std::sqrt(squared_sum + kQkNormEpsilon));
        for (std::size_t column = 0; column < state_dim; ++column) {
            host[row * state_dim + column] =
                f32_to_bf16(values[row * state_dim + column] * inverse);
        }
    }

    DeviceBuffer device(host.size() * sizeof(std::uint16_t));
    device.copy_from_host(host.data(), device.bytes);
    return device;
}

struct Operands {
    explicit Operands(Problem problem, bool normalized_qk)
        : problem(problem),
          q(normalized_qk
                ? make_normalized_bf16(static_cast<std::size_t>(problem.qk_heads) * problem.tokens *
                                           problem.batch,
                                       0x12345678U)
                : make_varied_bf16(static_cast<std::size_t>(gated_delta_net_detail::kStateDim) *
                                       problem.qk_heads * problem.tokens * problem.batch,
                                   0x12345678U)),
          k(normalized_qk
                ? make_normalized_bf16(static_cast<std::size_t>(problem.qk_heads) * problem.tokens *
                                           problem.batch,
                                       0x87654321U)
                : make_varied_bf16(static_cast<std::size_t>(gated_delta_net_detail::kStateDim) *
                                       problem.qk_heads * problem.tokens * problem.batch,
                                   0x87654321U)),
          v(make_varied_bf16(static_cast<std::size_t>(gated_delta_net_detail::kStateDim) *
                                 problem.value_heads * problem.tokens * problem.batch,
                             0x31415926U)),
          g(make_constant_f32(static_cast<std::size_t>(problem.value_heads) * problem.tokens *
                                  problem.batch,
                              -1.0F)),
          beta(make_constant_f32(static_cast<std::size_t>(problem.value_heads) * problem.tokens *
                                     problem.batch,
                                 0.5F)),
          out(make_zeros(static_cast<std::size_t>(gated_delta_net_detail::kStateDim) *
                         problem.value_heads * problem.tokens * problem.batch *
                         sizeof(std::uint16_t))) {}

    Tensor query() const {
        return Tensor(
            q.p, DType::BF16,
            {gated_delta_net_detail::kStateDim, problem.qk_heads, problem.tokens, problem.batch});
    }

    Tensor key() const {
        return Tensor(
            k.p, DType::BF16,
            {gated_delta_net_detail::kStateDim, problem.qk_heads, problem.tokens, problem.batch});
    }

    Tensor value() const {
        return Tensor(v.p, DType::BF16,
                      {gated_delta_net_detail::kStateDim, problem.value_heads, problem.tokens,
                       problem.batch});
    }

    Tensor gate() const {
        return Tensor(g.p, DType::FP32, {problem.value_heads, problem.tokens, problem.batch});
    }

    Tensor beta_tensor() const {
        return Tensor(beta.p, DType::FP32, {problem.value_heads, problem.tokens, problem.batch});
    }

    Tensor output() const {
        return Tensor(out.p, DType::BF16,
                      {gated_delta_net_detail::kStateDim, problem.value_heads, problem.tokens,
                       problem.batch});
    }

    Problem problem;
    DeviceBuffer q;
    DeviceBuffer k;
    DeviceBuffer v;
    DeviceBuffer g;
    DeviceBuffer beta;
    DeviceBuffer out;
};

template <class Launch>
GraphMeasurement measure_graph(Launch& launch, DeviceBuffer& flush, cudaStream_t stream,
                               const Options& options) {
    // Resolve lazy CUDA function attributes and reject an invalid case before capture.
    launch(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    TimedGraph graph;
    graph.capture(stream, launch);
    return {
        measure_cold_graph(graph, flush, stream, options.warmup, options.repeat),
        graph.nodes(),
    };
}

double running_logical_bytes(const Problem& problem) {
    const double tokens   = static_cast<double>(problem.tokens);
    const double batch    = static_cast<double>(problem.batch);
    const double qk_bytes = static_cast<double>(gated_delta_net_detail::kStateDim) *
                            problem.qk_heads * tokens * batch * sizeof(std::uint16_t);
    const double value_bytes = static_cast<double>(gated_delta_net_detail::kStateDim) *
                               problem.value_heads * tokens * batch * sizeof(std::uint16_t);
    const double gate_bytes =
        static_cast<double>(problem.value_heads) * tokens * batch * sizeof(float);
    const double state_bytes = static_cast<double>(gated_delta_net_detail::kStateDim) *
                               gated_delta_net_detail::kStateDim * problem.value_heads * batch *
                               sizeof(float);
    return 2.0 * qk_bytes + 2.0 * value_bytes + 2.0 * gate_bytes + 2.0 * state_bytes;
}

double batch_update_logical_bytes(const Problem& problem) {
    const double tokens   = static_cast<double>(problem.tokens);
    const double batch    = static_cast<double>(problem.batch);
    const double qk_bytes = static_cast<double>(gated_delta_net_detail::kStateDim) *
                            problem.qk_heads * tokens * batch * sizeof(std::uint16_t);
    const double value_bytes = static_cast<double>(gated_delta_net_detail::kStateDim) *
                               problem.value_heads * tokens * batch * sizeof(std::uint16_t);
    const double gate_bytes =
        static_cast<double>(problem.value_heads) * tokens * batch * sizeof(float);
    const double state_bytes = static_cast<double>(gated_delta_net_detail::kStateDim) *
                               gated_delta_net_detail::kStateDim * problem.value_heads * batch *
                               sizeof(float);
    return 2.0 * qk_bytes + 2.0 * value_bytes + 2.0 * gate_bytes + 2.0 * state_bytes +
           batch * sizeof(std::int32_t);
}

double qk_tensor_bytes(const Problem& problem) {
    return static_cast<double>(gated_delta_net_detail::kStateDim) * problem.qk_heads *
           problem.tokens * problem.batch * sizeof(std::uint16_t);
}

double value_tensor_bytes(const Problem& problem) {
    return static_cast<double>(gated_delta_net_detail::kStateDim) * problem.value_heads *
           problem.tokens * problem.batch * sizeof(std::uint16_t);
}

double gate_tensor_bytes(const Problem& problem) {
    return static_cast<double>(problem.value_heads) * problem.tokens * problem.batch *
           sizeof(float);
}

double state_tensor_bytes(const Problem& problem) {
    return static_cast<double>(gated_delta_net_detail::kStateDim) *
           gated_delta_net_detail::kStateDim * problem.value_heads * problem.batch * sizeof(float);
}

TrafficBytes prepare_traffic(const Problem& p, std::size_t workspace) {
    return {2.0 * qk_tensor_bytes(p) * (p.value_heads / p.qk_heads) + 2.0 * gate_tensor_bytes(p) +
                workspace,
            double(workspace)};
}

TrafficBytes recurrence_traffic(const Problem& p, int slices) {
    const double packets =
        double(chunked_detail::chunk_count(p.tokens)) * p.value_heads *
        (sizeof(chunked_detail::QkChunk) + sizeof(chunked_detail::ControlChunk)) * slices;
    return {packets + 2.0 * value_tensor_bytes(p) + 2.0 * state_tensor_bytes(p), packets};
}

TrafficBytes batch_update_traffic(const Problem& problem, bool composed) {
    const double logical = batch_update_logical_bytes(problem);
    if (!composed) { return {logical, 0.0}; }
    const double normalized_qk_round_trip = 4.0 * qk_tensor_bytes(problem);
    return {
        logical + normalized_qk_round_trip,
        normalized_qk_round_trip,
    };
}

BenchRow run_batch_update(const Options& options, std::int32_t tokens, DeviceBuffer& flush,
                          cudaStream_t stream) {
    const Problem problem{options.qk_heads, options.value_heads, tokens, options.batch};
    Operands operands(problem, false);

    const std::size_t qk_elements = static_cast<std::size_t>(gated_delta_net_detail::kStateDim) *
                                    problem.qk_heads * tokens * problem.batch;
    const std::size_t state_elements = static_cast<std::size_t>(gated_delta_net_detail::kStateDim) *
                                       gated_delta_net_detail::kStateDim * problem.value_heads;
    const std::int32_t slots = problem.batch;
    DeviceBuffer states      = make_zeros(state_elements * slots * sizeof(float));
    std::vector<std::int32_t> state_slots_host(static_cast<std::size_t>(problem.batch));
    for (std::int32_t row = 0; row < problem.batch; ++row) {
        state_slots_host[static_cast<std::size_t>(row)] = row;
    }
    DeviceBuffer state_slots(state_slots_host.size() * sizeof(std::int32_t));
    state_slots.copy_from_host(state_slots_host.data(), state_slots.bytes);
    DeviceBuffer q_normalized;
    DeviceBuffer k_normalized;
    if (options.qk_norm == "composed") {
        q_normalized = make_zeros(qk_elements * sizeof(std::uint16_t));
        k_normalized = make_zeros(qk_elements * sizeof(std::uint16_t));
    }

    Tensor q    = operands.query();
    Tensor k    = operands.key();
    Tensor v    = operands.value();
    Tensor g    = operands.gate();
    Tensor beta = operands.beta_tensor();
    Tensor out  = operands.output();
    Tensor ssm_states(states.p, DType::FP32,
                      {gated_delta_net_detail::kStateDim, gated_delta_net_detail::kStateDim,
                       problem.value_heads, slots});
    Tensor selected_slots(state_slots.p, DType::I32, {problem.batch});
    Tensor q_norm;
    Tensor k_norm;
    if (options.qk_norm == "composed") {
        q_norm =
            Tensor(q_normalized.p, DType::BF16,
                   {gated_delta_net_detail::kStateDim, problem.qk_heads, tokens, problem.batch});
        k_norm =
            Tensor(k_normalized.p, DType::BF16,
                   {gated_delta_net_detail::kStateDim, problem.qk_heads, tokens, problem.batch});
    }

    const bool composed = options.qk_norm == "composed";
    auto launch         = [&](cudaStream_t launch_stream) {
        if (composed) {
            ops::l2norm(q, kQkNormEpsilon, q_norm, launch_stream);
            ops::l2norm(k, kQkNormEpsilon, k_norm, launch_stream);
        }
        const Tensor& q_input = composed ? q_norm : q;
        const Tensor& k_input = composed ? k_norm : k;
        ops::gated_delta_net_batch_update(q_input, k_input, v, g, beta, gated_delta_net_scale(),
                                                  !composed, ssm_states, selected_slots, selected_slots,
                                                  out, launch_stream);
    };
    const GraphMeasurement measurement = measure_graph(launch, flush, stream, options);
    const TrafficBytes traffic         = batch_update_traffic(problem, composed);

    return {
        "batch_update",
        composed ? "composed" : "fused",
        composed ? "l2norm_x2+public.batch_update.qk_pre_normalized"
                 : "public.batch_update.qk_fused",
        tokens,
        0,
        0,
        0,
        measurement.graph_nodes,
        batch_update_logical_bytes(problem),
        traffic.total,
        traffic.intermediate,
        -1.0,
        -1.0,
        measurement.timing,
    };
}

std::vector<BenchRow> run_prefill(const Options& options, std::int32_t tokens, DeviceBuffer& flush,
                                  DeviceExecutionView execution) {
    constexpr auto state_dim = gated_delta_net_detail::kStateDim;
    const bool normalize     = options.mode != Mode::ChunkedOnly;
    const bool force_chunked =
        options.mode == Mode::ChunkedOnly || options.mode == Mode::ForceChunked;
    const bool force_recurrent = options.mode == Mode::RecurrentOnly;
    const bool chunked =
        force_chunked || (!force_recurrent && tokens >= chunked_detail::kMinTokens);
    const Problem problem{options.qk_heads, options.value_heads, tokens};
    Operands operands(problem, !normalize);
    const std::size_t state_elements = std::size_t(state_dim) * state_dim * problem.value_heads;
    DeviceBuffer state_in            = make_zeros(state_elements * sizeof(float));
    DeviceBuffer state_out           = make_zeros(state_elements * sizeof(float));
    Tensor q = operands.query(), k = operands.key(), v = operands.value();
    Tensor g = operands.gate(), beta = operands.beta_tensor(), out = operands.output();
    Tensor si(state_in.p, DType::FP32, {state_dim, state_dim, problem.value_heads});
    Tensor so(state_out.p, DType::FP32, {state_dim, state_dim, problem.value_heads});
    const auto layout =
        chunked_detail::workspace_layout(problem.qk_heads, problem.value_heads, tokens);
    std::size_t bytes = 0;
    if (force_chunked) {
        bytes = layout.total_bytes;
    } else if (!force_recurrent) {
        bytes = ops::gated_delta_net_workspace_capacity_bytes(problem.qk_heads, problem.value_heads,
                                                              tokens, tokens);
    }
    WorkspaceArena workspace(std::max<std::size_t>(256, bytes));
    const DeviceSpan backing{workspace.base(), bytes};
    auto* qk =
        chunked ? static_cast<chunked_detail::QkChunk*>(layout.qk.bind(backing).data) : nullptr;
    auto* control =
        chunked ? static_cast<chunked_detail::ControlChunk*>(layout.control.bind(backing).data)
                : nullptr;
    const chunked_detail::Arguments args{static_cast<const __nv_bfloat16*>(q.data),
                                         static_cast<const __nv_bfloat16*>(k.data),
                                         static_cast<const __nv_bfloat16*>(v.data),
                                         static_cast<const float*>(g.data),
                                         static_cast<const float*>(beta.data),
                                         static_cast<const float*>(si.data),
                                         static_cast<float*>(so.data),
                                         static_cast<__nv_bfloat16*>(out.data),
                                         problem.qk_heads,
                                         problem.value_heads,
                                         tokens,
                                         gated_delta_net_scale()};
    const int tile =
        chunked ? chunked_detail::value_tile(problem.value_heads, execution.multiprocessor_count)
                : 0;
    const auto prep_io          = prepare_traffic(problem, bytes);
    const auto rec_io           = recurrence_traffic(problem, tile ? state_dim / tile : 0);
    const TrafficBytes total_io = chunked ? TrafficBytes{prep_io.total + rec_io.total,
                                                         prep_io.intermediate + rec_io.intermediate}
                                          : TrafficBytes{running_logical_bytes(problem), 0};
    auto prepare                = [&](cudaStream_t stream) {
        chunked_detail::launch_prepare(args, qk, control, normalize, stream);
    };
    auto recurrence = [&](cudaStream_t stream) {
        chunked_detail::launch_recurrence(args, qk, control,
                                          {stream, execution.multiprocessor_count});
    };
    auto pipeline = [&](cudaStream_t stream) {
        if (force_chunked) {
            prepare(stream);
            recurrence(stream);
        } else if (force_recurrent)
            gated_delta_net_detail::launch_recurrent_inout(
                q, k, v, g, beta, gated_delta_net_scale(), normalize, si, so, out, stream);
        else
            ops::gated_delta_net(q, k, v, g, beta, gated_delta_net_scale(), normalize, workspace,
                                 si, so, out, {stream, execution.multiprocessor_count});
    };
    std::vector<BenchRow> rows;
    auto measure = [&](const char* name, TrafficBytes io, auto& body, double logical) {
        const auto measured = measure_graph(body, flush, execution.stream, options);
        rows.push_back({"running", normalize ? "fused" : "pre_normalized", name, tokens,
                        chunked ? tokens / chunked_detail::kChunkSize : 0,
                        chunked ? tokens % chunked_detail::kChunkSize : 0, bytes,
                        measured.graph_nodes, logical, io.total, io.intermediate, -1.0, -1.0,
                        measured.timing});
    };
    measure(force_chunked     ? "forced.chunked"
            : force_recurrent ? "forced.recurrent"
                              : "public.total",
            total_io, pipeline, running_logical_bytes(problem));
    if (options.breakdown && chunked) {
        measure("chunked.prepare", prep_io, prepare, 0);
        measure("chunked.recurrence", rec_io, recurrence, 0);
        const double sum = rows[1].timing.median_us + rows[2].timing.median_us;
        for (int i = 1; i < 3; ++i) {
            rows[i].stage_share_pct = 100.0 * rows[i].timing.median_us / sum;
            rows[i].relative_to_e2e_pct =
                100.0 * rows[i].timing.median_us / rows[0].timing.median_us;
        }
    }
    return rows;
}

double logical_gbps(const BenchRow& row) {
    if (row.logical_bytes == 0.0 || row.timing.median_us <= 0.0) { return 0.0; }
    return row.logical_bytes / (row.timing.median_us * 1.0e3);
}

double traffic_gbps(const BenchRow& row) {
    if (row.traffic_bytes == 0.0 || row.timing.median_us <= 0.0) { return 0.0; }
    return row.traffic_bytes / (row.timing.median_us * 1.0e3);
}

void print_csv_header() {
    std::printf(
        "state_form,normalization,implementation,dtype,state_dim,qk_heads,value_heads,tokens,"
        "batch,full_chunks,tail_tokens,workspace_bytes,logical_bytes,tensor_io_request_bytes,"
        "workspace_request_bytes,graph_nodes,cache,execution,warmup,repeat,"
        "median_us,min_us,p95_us,logical_gbps,tensor_io_request_gbps,stage_share_pct,"
        "relative_to_e2e_pct\n");
}

void print_row(const BenchRow& row, const Options& options) {
    if (options.csv) {
        std::printf("%s,%s,%s,BF16,%d,%d,%d,%d,%d,%d,%d,%zu,%.0f,%.0f,%.0f,%zu,"
                    "cold_l2,cuda_graph,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,",
                    row.state_form, row.normalization, row.implementation.c_str(),
                    gated_delta_net_detail::kStateDim, options.qk_heads, options.value_heads,
                    row.tokens, options.batch, row.full_chunks, row.tail_tokens,
                    row.workspace_bytes, row.logical_bytes, row.traffic_bytes,
                    row.intermediate_traffic_bytes, row.graph_nodes, options.warmup, options.repeat,
                    row.timing.median_us, row.timing.min_us, row.timing.p95_us, logical_gbps(row),
                    traffic_gbps(row));
        if (row.stage_share_pct >= 0.0) { std::printf("%.2f", row.stage_share_pct); }
        std::printf(",");
        if (row.relative_to_e2e_pct >= 0.0) { std::printf("%.2f", row.relative_to_e2e_pct); }
        std::printf("\n");
        return;
    }

    std::printf("%-8s T=%-4d B=%-2d full_chunks=%-2d tail=%-2d nodes=%zu ws=%7.2f MiB "
                "median=%8.3f us min=%8.3f us p95=%8.3f us",
                row.state_form, row.tokens, options.batch, row.full_chunks, row.tail_tokens,
                row.graph_nodes,
                static_cast<double>(row.workspace_bytes) / static_cast<double>(1ULL << 20),
                row.timing.median_us, row.timing.min_us, row.timing.p95_us);
    if (row.stage_share_pct >= 0.0) { std::printf(" stage_share=%6.2f%%", row.stage_share_pct); }
    if (row.relative_to_e2e_pct >= 0.0) {
        std::printf(" e2e_ratio=%6.2f%%", row.relative_to_e2e_pct);
    }
    std::printf("\n  bytes logical=%7.2f MiB traffic=%7.2f MiB intermediate=%7.2f MiB"
                " | bandwidth logical=%8.1f GB/s traffic=%8.1f GB/s"
                "\n  %s\n",
                row.logical_bytes / static_cast<double>(1ULL << 20),
                row.traffic_bytes / static_cast<double>(1ULL << 20),
                row.intermediate_traffic_bytes / static_cast<double>(1ULL << 20), logical_gbps(row),
                traffic_gbps(row), row.implementation.c_str());
}

void print_banner(const Options& options, const cudaDeviceProp& device) {
    if (options.csv) {
        print_csv_header();
        return;
    }
    std::printf("Gated DeltaNet benchmark\n");
    std::printf("  device      %s (sm_%d%d)\n", device.name, device.major, device.minor);
    std::printf("  geometry    state_dim=128 qk_heads=%d value_heads=%d batch=%d\n",
                options.qk_heads, options.value_heads, options.batch);
    std::printf("  execution   CUDA Graph replay\n");
    std::printf("  cache       cold L2 (%zu MiB flush before each sample)\n",
                options.flush_bytes >> 20);
    std::printf(
        "  traffic     CTA tensor I/O requests including replicated packets (not measured DRAM)\n");
    std::printf("  samples     %d warmup + %d measured per case\n\n", options.warmup,
                options.repeat);
}

} // namespace

int main(int argc, char** argv) {
    cudaStream_t stream = nullptr;
    try {
        const Options options = parse_options(argc, argv);
        if (options.help) {
            print_help(argv[0]);
            return 0;
        }

        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
            std::printf("SKIP: no usable CUDA device\n");
            return 0;
        }
        cudaDeviceProp device{};
        CUDA_CHECK(cudaGetDeviceProperties(&device, 0));
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        DeviceBuffer flush(options.flush_bytes);
        print_banner(options, device);

        for (const std::int32_t tokens : token_values(options)) {
            validate_tokens(options, tokens);
            if (options.mode == Mode::BatchUpdate) {
                print_row(run_batch_update(options, tokens, flush, stream), options);
            } else {
                for (const BenchRow& row :
                     run_prefill(options, tokens, flush, {stream, device.multiProcessorCount}))
                    print_row(row, options);
            }
        }

        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        if (stream != nullptr) { cudaStreamDestroy(stream); }
        std::fprintf(stderr, "ninfer_gated_delta_net_bench: %s\n", error.what());
        return 1;
    }
}
