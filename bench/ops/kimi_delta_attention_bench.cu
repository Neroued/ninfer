// Cold-L2 CUDA Graph benchmark for the public recurrent Kimi Delta Attention Direct and selected
// state-pool batch-update paths. Kimi Linear uses H=32 and Kimi K3 uses H=96; both use K=V=128
// and enter runtime-H production kernels.
#include "ninfer/ops/kimi_delta_attention.h"

#include "ninfer_bench_common.h"

#include <cuda_runtime.h>

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

constexpr std::int32_t kStateDim      = 128;
constexpr std::int32_t kDefaultTokens = 1;
constexpr std::size_t kDefaultFlush   = 256ULL << 20;
constexpr float kLowerBound           = -5.0F;
constexpr float kScale                = 1.0F / 11.313708498984761F;

struct Options {
    std::string profile     = "kimi-k3";
    std::int32_t heads      = 96;
    std::int32_t tokens     = kDefaultTokens;
    std::int32_t batch      = 1;
    bool heads_explicit     = false;
    bool tokens_explicit    = false;
    bool batch_explicit     = false;
    bool batch_update       = false;
    bool sweep              = false;
    bool batch_sweep        = false;
    bool csv                = false;
    bool help               = false;
    int warmup              = 20;
    int repeat              = 100;
    std::size_t flush_bytes = kDefaultFlush;
};

struct Problem {
    const char* profile;
    std::int32_t heads;
    std::int32_t tokens;
    std::int32_t batch;
    bool batch_update;
};

[[noreturn]] void fail(const std::string& message) { throw std::invalid_argument(message); }

std::int32_t parse_integer(const char* flag, const char* text, std::int32_t minimum) {
    errno            = 0;
    char* end        = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (errno != 0 || text == end || *end != '\0' || value < minimum || value > INT32_MAX) {
        fail(std::string("invalid value for ") + flag + ": " + text);
    }
    return static_cast<std::int32_t>(value);
}

void select_profile(Options& options, std::string_view profile) {
    if (profile == "kimi-linear") {
        options.profile = "kimi-linear";
        if (!options.heads_explicit) { options.heads = 32; }
    } else if (profile == "kimi-k3") {
        options.profile = "kimi-k3";
        if (!options.heads_explicit) { options.heads = 96; }
    } else if (profile == "all") {
        options.profile = "all";
        if (options.heads_explicit) { fail("--profile all cannot be combined with --heads"); }
    } else {
        fail("--profile must be kimi-linear, kimi-k3, or all");
    }
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto take = [&](const char* flag) -> const char* {
            if (++index >= argc) { fail(std::string("missing value for ") + flag); }
            return argv[index];
        };

        if (argument == "--profile") {
            select_profile(options, take("--profile"));
        } else if (argument == "--heads") {
            options.heads          = parse_integer("--heads", take("--heads"), 1);
            options.heads_explicit = true;
            options.profile        = "custom";
        } else if (argument == "--tokens") {
            options.tokens          = parse_integer("--tokens", take("--tokens"), 1);
            options.tokens_explicit = true;
        } else if (argument == "--batch-update") {
            options.batch_update = true;
        } else if (argument == "--batch") {
            options.batch          = parse_integer("--batch", take("--batch"), 1);
            options.batch_explicit = true;
        } else if (argument == "--sweep") {
            options.sweep = true;
        } else if (argument == "--batch-sweep") {
            options.batch_sweep = true;
        } else if (argument == "--warmup") {
            options.warmup = parse_integer("--warmup", take("--warmup"), 0);
        } else if (argument == "--repeat") {
            options.repeat = parse_integer("--repeat", take("--repeat"), 1);
        } else if (argument == "--flush-mib") {
            options.flush_bytes =
                static_cast<std::size_t>(parse_integer("--flush-mib", take("--flush-mib"), 1))
                << 20;
        } else if (argument == "--csv") {
            options.csv = true;
        } else if (argument == "--help" || argument == "-h") {
            options.help = true;
        } else {
            fail("unknown argument: " + std::string(argument));
        }
    }
    if (options.sweep && options.tokens_explicit) {
        fail("--tokens and --sweep are mutually exclusive");
    }
    if (options.batch_sweep && options.batch_explicit) {
        fail("--batch and --batch-sweep are mutually exclusive");
    }
    if (options.batch_update) {
        if (options.tokens_explicit || options.sweep) {
            fail("--batch-update has a fixed token extent of 1");
        }
        if (options.batch > 8) { fail("--batch-update requires B in [1,8]"); }
    } else if (options.batch_explicit || options.batch_sweep) {
        fail("--batch and --batch-sweep require --batch-update");
    }
    return options;
}

void print_help(const char* program) {
    std::printf("Usage: %s [options]\n"
                "\n"
                "Mode (default: Direct):\n"
                "  --batch-update  selected-slot one-token update\n"
                "\n"
                "Workload:\n"
                "  --profile NAME   kimi-linear (H=32), kimi-k3 (H=96), or all (default: kimi-k3)\n"
                "  --heads H        custom positive runtime head count\n"
                "  --tokens T       Direct token extent (default: 1)\n"
                "  --sweep          Direct T in {1,2,4,8,16,32,64}\n"
                "  --batch B        batch-update B in [1,8] (default: 1)\n"
                "  --batch-sweep    batch-update B in {1,2,4,8}\n"
                "\n"
                "Measurement:\n"
                "  --warmup N       cold-L2 graph warmups (default: 20)\n"
                "  --repeat N       measured cold-L2 graph replays (default: 100)\n"
                "  --flush-mib N    L2 flush allocation in MiB (default: 256)\n"
                "  --csv            emit CSV\n"
                "  -h, --help       show this help\n",
                program);
}

DeviceBuffer make_f32(std::size_t count, float base, float step) {
    std::vector<float> host(count);
    for (std::size_t index = 0; index < count; ++index) {
        host[index] = base + step * static_cast<float>(static_cast<int>(index % 127) - 63);
    }
    DeviceBuffer device(count * sizeof(float));
    device.copy_from_host(host.data(), device.bytes);
    return device;
}

double traffic_bytes(const Problem& problem) {
    const double batch   = static_cast<double>(problem.batch);
    const double vectors = static_cast<double>(kStateDim) * problem.heads * problem.tokens * batch;
    const double state   = static_cast<double>(kStateDim) * kStateDim * problem.heads *
                         (problem.batch_update ? batch : 1.0);
    const double controls   = static_cast<double>(problem.heads) * problem.tokens * batch;
    const double parameters = static_cast<double>(problem.heads) * (kStateDim + 1);
    // q/k/v/g reads plus out write, beta read, A_log/dt_bias reads, and state read+write.
    return 5.0 * vectors * sizeof(std::uint16_t) + controls * sizeof(std::uint16_t) +
           parameters * sizeof(float) + 2.0 * state * sizeof(float) +
           (problem.batch_update ? batch * sizeof(std::int32_t) : 0.0);
}

ColdTiming measure_problem(const Problem& problem, const Options& options,
                           std::size_t& graph_nodes) {
    const std::size_t vector_elements =
        static_cast<std::size_t>(kStateDim) * problem.heads * problem.tokens * problem.batch;
    const std::size_t state_elements = static_cast<std::size_t>(kStateDim) * kStateDim *
                                       problem.heads * (problem.batch_update ? problem.batch : 1);
    const std::size_t beta_elements =
        static_cast<std::size_t>(problem.heads) * problem.tokens * problem.batch;

    DeviceBuffer q     = make_bf16(vector_elements);
    DeviceBuffer k     = make_bf16(vector_elements);
    DeviceBuffer v     = make_bf16(vector_elements);
    DeviceBuffer g     = make_bf16(vector_elements);
    DeviceBuffer beta  = make_bf16(beta_elements);
    DeviceBuffer a_log = make_f32(static_cast<std::size_t>(problem.heads), -0.15F, 0.002F);
    DeviceBuffer dt_bias =
        make_f32(static_cast<std::size_t>(kStateDim) * problem.heads, 0.0F, 0.01F);
    DeviceBuffer state = make_zeros(state_elements * sizeof(float));
    DeviceBuffer out   = make_zeros(vector_elements * sizeof(std::uint16_t));
    DeviceBuffer flush(options.flush_bytes);
    std::vector<std::int32_t> host_state_slots(static_cast<std::size_t>(problem.batch));
    for (std::int32_t row = 0; row < problem.batch; ++row) {
        host_state_slots[static_cast<std::size_t>(row)] = row;
    }
    DeviceBuffer state_slots(host_state_slots.size() * sizeof(std::int32_t));
    state_slots.copy_from_host(host_state_slots.data(), state_slots.bytes);

    Tensor q_tensor(q.p, DType::BF16, {kStateDim, problem.heads, problem.tokens, problem.batch});
    Tensor k_tensor(k.p, DType::BF16, {kStateDim, problem.heads, problem.tokens, problem.batch});
    Tensor v_tensor(v.p, DType::BF16, {kStateDim, problem.heads, problem.tokens, problem.batch});
    Tensor g_tensor(g.p, DType::BF16, {kStateDim, problem.heads, problem.tokens, problem.batch});
    Tensor beta_tensor(beta.p, DType::BF16, {problem.heads, problem.tokens, problem.batch});
    Tensor a_log_tensor(a_log.p, DType::FP32, {problem.heads});
    Tensor dt_bias_tensor(dt_bias.p, DType::FP32, {kStateDim, problem.heads});
    Tensor state_tensor(
        state.p, DType::FP32,
        {kStateDim, kStateDim, problem.heads, problem.batch_update ? problem.batch : 1});
    Tensor state_slots_tensor(state_slots.p, DType::I32, {problem.batch});
    Tensor out_tensor(out.p, DType::BF16,
                      {kStateDim, problem.heads, problem.tokens, problem.batch});

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    ColdTiming timing;
    {
        TimedGraph graph;
        graph.capture(stream, [&](cudaStream_t capture_stream) {
            if (problem.batch_update) {
                ops::kimi_delta_attention_batch_update(
                    q_tensor, k_tensor, v_tensor, g_tensor, beta_tensor, a_log_tensor,
                    dt_bias_tensor, kLowerBound, kScale, state_tensor, state_slots_tensor,
                    out_tensor, capture_stream);
            } else {
                ops::kimi_delta_attention(q_tensor, k_tensor, v_tensor, g_tensor, beta_tensor,
                                          a_log_tensor, dt_bias_tensor, kLowerBound, kScale,
                                          state_tensor, out_tensor, capture_stream);
            }
        });
        graph_nodes = graph.nodes();
        timing      = measure_cold_graph(graph, flush, stream, options.warmup, options.repeat);
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    return timing;
}

void emit(const Problem& problem, const ColdTiming& timing, std::size_t graph_nodes, bool csv) {
    const double bytes = traffic_bytes(problem);
    const double gbs   = bytes / timing.median_us / 1000.0;
    if (csv) {
        std::printf("%s,%s,%d,%d,%d,%zu,%.6f,%.6f,%.6f,%.3f,%.0f\n",
                    problem.batch_update ? "batch_update" : "direct", problem.profile,
                    problem.heads, problem.tokens, problem.batch, graph_nodes, timing.median_us,
                    timing.min_us, timing.p95_us, gbs, bytes);
        return;
    }
    std::printf("mode=%-12s profile=%-11s H=%3d T=%3d B=%d graph_nodes=%zu median=%8.3f us "
                "min=%8.3f us p95=%8.3f us traffic=%7.1f GB/s\n",
                problem.batch_update ? "batch_update" : "direct", problem.profile, problem.heads,
                problem.tokens, problem.batch, graph_nodes, timing.median_us, timing.min_us,
                timing.p95_us, gbs);
}

std::vector<std::int32_t> token_extents(const Options& options) {
    if (options.sweep) { return {1, 2, 4, 8, 16, 32, 64}; }
    return {options.tokens};
}

std::vector<Problem> problems(const Options& options) {
    std::vector<Problem> result;
    const std::vector<std::int32_t> batches = options.batch_sweep
                                                  ? std::vector<std::int32_t>{1, 2, 4, 8}
                                                  : std::vector<std::int32_t>{options.batch};
    const std::vector<std::int32_t> tokens =
        options.batch_update ? std::vector<std::int32_t>{1} : token_extents(options);
    for (const std::int32_t token_count : tokens) {
        for (const std::int32_t batch : batches) {
            if (options.profile == "all") {
                result.push_back({"kimi-linear", 32, token_count, batch, options.batch_update});
                result.push_back({"kimi-k3", 96, token_count, batch, options.batch_update});
            } else {
                result.push_back({options.profile.c_str(), options.heads, token_count, batch,
                                  options.batch_update});
            }
        }
    }
    return result;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        if (options.help) {
            print_help(argv[0]);
            return 0;
        }
        if (options.csv) {
            std::printf("mode,profile,heads,tokens,batch,graph_nodes,median_us,min_us,p95_us,"
                        "traffic_gbs,traffic_bytes\n");
        }
        for (const Problem& problem : problems(options)) {
            std::size_t graph_nodes = 0;
            const ColdTiming timing = measure_problem(problem, options, graph_nodes);
            emit(problem, timing, graph_nodes, options.csv);
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_kimi_delta_attention_bench: %s\n", error.what());
        return 1;
    }
}
