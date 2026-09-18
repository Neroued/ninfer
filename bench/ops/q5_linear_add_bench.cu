// Cold-cache public Op benchmark for Q5_G64_FP16 RowSplit LinearAdd [5120,6144] and [5120,17408].
//
// LinearAdd updates the residual in place, so every warmup launch and every measured sample is
// preceded by restoring the residual to the same finite non-trivial initial value and then
// evicting L2; both run outside the timed interval. One complete public ops::linear_add call is
// captured per graph, and the reported node count is the captured graph's real node count.

#include "core/weight.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"

#include "core/device.h"
#include "ninfer_bench_common.h"
#include "quantized_weight.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;

namespace {

constexpr std::int32_t kRows           = 5120;
constexpr std::size_t kDefaultFlushMiB = 256;
constexpr int kDefaultWarmup           = 5;
constexpr int kDefaultRepeat           = 50;
constexpr double kDramSpecGBs          = 1792.0;
// NVIDIA's GB202 table reports dense/sparse pairs at boost clock; keep the accumulator precision
// explicit. SIMT and split-K routes report no Tensor Core ratio.
constexpr double kBf16Fp32AccumulateTFLOPs = 209.5;

struct Options {
    std::int32_t hidden = 0;
    std::vector<std::int32_t> tokens{1, 2, 4, 8, 16, 24, 25, 32, 48, 49, 64, 192, 193};
    int warmup                = kDefaultWarmup;
    int repeat                = kDefaultRepeat;
    bool graph                = false;
    bool profile              = false;
    std::uint64_t flush_bytes = kDefaultFlushMiB << 20;
    std::string csv_out;
};

std::vector<std::int32_t> parse_tokens(std::string_view raw) {
    std::vector<std::int32_t> result;
    std::size_t begin = 0;
    while (begin < raw.size()) {
        const std::size_t end = raw.find(',', begin);
        const std::string token(
            raw.substr(begin, end == std::string_view::npos ? raw.size() - begin : end - begin));
        const long value = std::stol(token);
        if (value <= 0 || value > std::numeric_limits<std::int32_t>::max()) {
            throw std::invalid_argument("--t-sweep values must be positive int32");
        }
        result.push_back(static_cast<std::int32_t>(value));
        if (end == std::string_view::npos) { break; }
        begin = end + 1;
    }
    if (result.empty()) { throw std::invalid_argument("--t-sweep must not be empty"); }
    return result;
}

void usage(const char* argv0) {
    std::printf("Usage:\n"
                "  %s --k 6144|17408 --t-sweep 1,2,... [options]\n"
                "Options:\n"
                "  --warmup N         Warmup launches per point (default %d).\n"
                "  --repeat N         Measured samples per point (default %d).\n"
                "  --execution MODE   eager (default) or graph; time one complete public Op.\n"
                "  --flush-mib N      L2 eviction buffer size (default %zu MiB).\n"
                "  --csv-out PATH     Write all measurement rows as CSV.\n"
                "  --profile          Report the public shape and workspace query for one T.\n",
                argv0, kDefaultWarmup, kDefaultRepeat, kDefaultFlushMiB);
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto next = [&](const char* label) -> std::string_view {
            if (++index >= argc) { throw std::invalid_argument(std::string("missing ") + label); }
            return argv[index];
        };
        if (argument == "--k") {
            options.hidden = std::stoi(std::string(next("--k value")));
        } else if (argument == "--t-sweep") {
            options.tokens = parse_tokens(next("--t-sweep value"));
        } else if (argument == "--warmup") {
            options.warmup = std::stoi(std::string(next("--warmup value")));
        } else if (argument == "--repeat") {
            options.repeat = std::stoi(std::string(next("--repeat value")));
        } else if (argument == "--execution") {
            const std::string mode(next("--execution value"));
            if (mode == "graph") {
                options.graph = true;
            } else if (mode == "eager") {
                options.graph = false;
            } else {
                throw std::invalid_argument("--execution must be eager or graph");
            }
        } else if (argument == "--flush-mib") {
            options.flush_bytes =
                static_cast<std::uint64_t>(std::stoll(std::string(next("--flush-mib value"))))
                << 20;
        } else if (argument == "--csv-out") {
            options.csv_out = std::string(next("--csv-out value"));
        } else if (argument == "--profile") {
            options.profile = true;
        } else if (argument == "--help" || argument == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(argument));
        }
    }
    if (options.hidden != 6144 && options.hidden != 17408) {
        throw std::invalid_argument("--k must be 6144 or 17408");
    }
    if (options.warmup < 0 || options.repeat <= 0) {
        throw std::invalid_argument("--warmup must be nonnegative and --repeat positive");
    }
    if (options.flush_bytes == 0) { throw std::invalid_argument("--flush-mib must be positive"); }
    if (options.profile && options.tokens.size() != 1) {
        throw std::invalid_argument("--profile requires exactly one T");
    }
    if (options.profile && !options.csv_out.empty()) {
        throw std::invalid_argument("--profile and --csv-out are mutually exclusive");
    }
    return options;
}

struct Row {
    std::int32_t tokens     = 0;
    const char* execution   = "eager";
    std::size_t graph_nodes = 0;
    bench::ColdTiming timing;
    std::uint64_t weight_bytes        = 0;
    std::uint64_t logical_bytes       = 0;
    double projection_flops           = 0.0;
    std::size_t workspace_sweep_bytes = 0;
    std::size_t workspace_exact_bytes = 0;
    std::size_t workspace_used_bytes  = 0;
};

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const auto [min_it, max_it] =
            std::minmax_element(options.tokens.begin(), options.tokens.end());
        const std::int32_t min_t = *min_it;
        const std::int32_t max_t = *max_it;

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        DeviceBuffer flush(options.flush_bytes);
        DeviceBuffer input    = bench::make_bf16(static_cast<std::size_t>(options.hidden) * max_t);
        DeviceBuffer residual = bench::make_bf16(static_cast<std::size_t>(kRows) * max_t);
        // The same deterministic ramp, kept aside so every sample starts from identical residual
        // contents: LinearAdd accumulates, and an accumulated operand is not the operand the
        // benchmark claims to measure.
        DeviceBuffer residual_init = bench::make_bf16(static_cast<std::size_t>(kRows) * max_t);
        bench::PackedQuantizedWeight packed = bench::make_row_split_weight(
            QType::Q5_G64_FP16, kRows, options.hidden, options.hidden, {0x31, 0xa5, 0x3c00});

        const std::size_t workspace_capacity = ops::linear_add_workspace_capacity_bytes(
            QType::Q5_G64_FP16, kRows, options.hidden, min_t, max_t);
        // The A16 routes report a zero-capacity profile, and the arena type rejects a zero-byte
        // allocation, so the owning arena is non-empty here. Every row still reports how many of
        // those bytes the public Op actually consumes.
        WorkspaceArena workspace(std::max<std::size_t>(workspace_capacity, 256));

        if (options.profile) {
            const std::int32_t tokens = options.tokens.front();
            Tensor x(input.p, DType::BF16, {options.hidden, tokens});
            Tensor out(residual.p, DType::BF16, {kRows, tokens});
            workspace.reset();
            ops::linear_add(x, packed.weight, out, ops::LinearPolicy::A16Only, workspace, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            std::printf("profile op=linear_add qtype=q5_g64_fp16 rows=%d k=%d T=%d "
                        "workspace_sweep=%zu workspace_exact=%zu workspace_used=%zu\n",
                        kRows, options.hidden, tokens, workspace_capacity,
                        ops::linear_add_workspace_capacity_bytes(
                            QType::Q5_G64_FP16, kRows, options.hidden, ops::LinearPolicy::A16Only,
                            tokens, tokens),
                        workspace.used());
            CUDA_CHECK(cudaStreamDestroy(stream));
            return 0;
        }

        std::printf("# execution=%s warmup=%d repeat=%d flush_bytes=%llu workspace_sweep_bytes=%zu "
                    "op=linear_add qtype=q5_g64_fp16 rows=%d k=%d\n",
                    options.graph ? "graph" : "eager", options.warmup, options.repeat,
                    static_cast<unsigned long long>(options.flush_bytes), workspace_capacity, kRows,
                    options.hidden);

        std::vector<Row> rows;
        rows.reserve(options.tokens.size());

        for (const std::int32_t tokens : options.tokens) {
            Tensor x(input.p, DType::BF16, {options.hidden, tokens});
            Tensor out(residual.p, DType::BF16, {kRows, tokens});
            const std::size_t residual_bytes = static_cast<std::size_t>(kRows) * tokens * 2;
            const auto restore               = [&](cudaStream_t prepare_stream) {
                CUDA_CHECK(cudaMemcpyAsync(residual.p, residual_init.p, residual_bytes,
                                                         cudaMemcpyDeviceToDevice, prepare_stream));
            };
            const auto body = [&](cudaStream_t launch_stream) {
                ops::linear_add(x, packed.weight, out, ops::LinearPolicy::A16Only, workspace,
                                launch_stream);
            };

            bench::TimedGraph graph;
            if (options.graph) graph.capture(stream, body);

            const bench::ColdTiming timing =
                options.graph ? bench::measure_cold_graph_prepared(restore, graph, flush, stream,
                                                                   options.warmup, options.repeat)
                              : bench::measure_cold_launch_prepared(restore, body, flush, stream,
                                                                    options.warmup, options.repeat);

            // Evidence for the public zero-extra-workspace contract: consume from a freshly reset
            // arena in a single untimed call and report what the Op actually took.
            workspace.reset();
            body(stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            const std::size_t workspace_used = workspace.used();

            Row row;
            row.tokens       = tokens;
            row.execution    = options.graph ? "graph" : "eager";
            row.graph_nodes  = graph.nodes();
            row.timing       = timing;
            row.weight_bytes = packed.model_weight_bytes();
            // x is read (2 bytes/element), the residual is read and written (4 bytes/element).
            row.logical_bytes =
                row.weight_bytes + 2ull * options.hidden * tokens + 4ull * kRows * tokens;
            row.projection_flops      = 2.0 * static_cast<double>(kRows) * options.hidden * tokens;
            row.workspace_sweep_bytes = workspace_capacity;
            row.workspace_exact_bytes = ops::linear_add_workspace_capacity_bytes(
                QType::Q5_G64_FP16, kRows, options.hidden, ops::LinearPolicy::A16Only, tokens,
                tokens);
            row.workspace_used_bytes = workspace_used;
            const double seconds     = timing.median_us * 1.0e-6;
            std::printf("K=%-5d T=%-4d %-5s nodes=%zu median=%9.3f us min=%9.3f p95=%9.3f "
                        "%7.1f GB/s %7.2f TFLOP/s workspace_exact=%zu workspace_used=%zu\n",
                        options.hidden, tokens, row.execution, row.graph_nodes, timing.median_us,
                        timing.min_us, timing.p95_us, row.logical_bytes / seconds / 1.0e9,
                        row.projection_flops / seconds / 1.0e12, row.workspace_exact_bytes,
                        row.workspace_used_bytes);
            rows.push_back(row);
        }

        if (!options.csv_out.empty()) {
            std::FILE* file = std::fopen(options.csv_out.c_str(), "w");
            if (file == nullptr) {
                throw std::runtime_error("cannot open --csv-out path: " + options.csv_out);
            }
            std::fprintf(file,
                         "op,rows,k,t,execution,graph_nodes,median_us,min_us,p95_us,weight_bytes,"
                         "logical_bytes,projection_flops,effective_gbs,dram_spec_pct,useful_tflops,"
                         "workspace_sweep_bytes,workspace_exact_bytes,workspace_used_bytes\n");
            for (const Row& row : rows) {
                const double seconds = row.timing.median_us * 1.0e-6;
                const double gbs     = row.logical_bytes / seconds / 1.0e9;
                std::fprintf(file,
                             "linear_add,%d,%d,%d,%s,%zu,%.3f,%.3f,%.3f,%llu,%llu,%.0f,%.2f,%.2f,"
                             "%.3f,%zu,%zu,%zu\n",
                             kRows, options.hidden, row.tokens, row.execution, row.graph_nodes,
                             row.timing.median_us, row.timing.min_us, row.timing.p95_us,
                             static_cast<unsigned long long>(row.weight_bytes),
                             static_cast<unsigned long long>(row.logical_bytes),
                             row.projection_flops, gbs, gbs / kDramSpecGBs * 100.0,
                             row.projection_flops / seconds / 1.0e12, row.workspace_sweep_bytes,
                             row.workspace_exact_bytes, row.workspace_used_bytes);
            }
            std::fclose(file);
        }

        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_q5_linear_add_bench: %s\n", error.what());
        return 1;
    }
}
