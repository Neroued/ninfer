// Cold-cache benchmark for the two registered public Q8 LinearPair geometries.

#include "common/roofline.h"
#include "core/weight.h"
#include "core/buffer.h"
#include "ninfer/ops/linear_pair.h"

#include "core/device.h"
#include "op_measurement.cuh"
#include "quantized_weight.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
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

constexpr std::int32_t kParentRows = 6144;
constexpr std::int32_t kFirstRow   = 4096;
constexpr std::int32_t kSecondRow  = 5120;
constexpr std::int32_t kRows       = 1024;

enum class Execution : std::uint8_t { Eager, Graph };

struct Options {
    std::vector<std::int32_t> tokens{1, 2, 4, 8, 16, 32, 48, 64, 96, 128};
    std::int32_t k            = 0;
    Execution execution       = Execution::Graph;
    int warmup                = 5;
    int repeat                = 30;
    std::uint64_t flush_bytes = 256ULL << 20;
};

Options parse_options(int argc, char** argv) {
    Options options;
    bool have_tokens = false;
    bool have_sweep  = false;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto next = [&](const char* label) -> std::string_view {
            if (++index >= argc) { throw std::invalid_argument(std::string("missing ") + label); }
            return argv[index];
        };
        if (argument == "--tokens") {
            options.tokens = bench::measurement::integers(next("--tokens value"));
            std::sort(options.tokens.begin(), options.tokens.end());
            options.tokens.erase(std::unique(options.tokens.begin(), options.tokens.end()),
                                 options.tokens.end());
            have_tokens = true;
        } else if (argument == "--sweep") {
            options.tokens = bench::measurement::sweep(next("--sweep value"));
            have_sweep     = true;
        } else if (argument == "--k") {
            options.k = bench::measurement::integer(next("--k value"));
            if (options.k != 2048 && options.k != 5120) {
                throw std::invalid_argument("--k must be 2048 or 5120");
            }
        } else if (argument == "--execution") {
            const std::string_view value = next("--execution value");
            if (value == "eager")
                options.execution = Execution::Eager;
            else if (value == "graph")
                options.execution = Execution::Graph;
            else
                throw std::invalid_argument("--execution must be eager or graph");
        } else if (argument == "--warmup") {
            options.warmup = bench::measurement::integer(next("--warmup value"), 0);
        } else if (argument == "--repeat") {
            options.repeat = bench::measurement::integer(next("--repeat value"));
        } else if (argument == "--flush-mib") {
            const std::uint64_t mib =
                bench::measurement::integer<std::uint64_t>(next("--flush-mib value"));
            if (mib > (std::numeric_limits<std::uint64_t>::max() >> 20)) {
                throw std::overflow_error("flush size overflows");
            }
            options.flush_bytes = mib << 20;
        } else if (argument == "--help" || argument == "-h") {
            std::printf("Usage: %s --k 2048|5120 "
                        "[--tokens 1,2,... | --sweep START:END[:STEP]] "
                        "[--execution eager|graph] [--warmup N] [--repeat N] [--flush-mib N]\n",
                        argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(argument));
        }
    }
    if (have_tokens && have_sweep) {
        throw std::invalid_argument("--tokens and --sweep are mutually exclusive");
    }
    if (options.k == 0) { throw std::invalid_argument("--k is required"); }
    if (options.repeat <= 0) { throw std::invalid_argument("--repeat must be positive"); }
    return options;
}

std::uint64_t pair_weight_bytes(std::int32_t k) {
    const std::uint64_t codes_per_weight = static_cast<std::uint64_t>(kRows) * k;
    const std::uint64_t scales_per_weight =
        static_cast<std::uint64_t>(kRows) * (k / 32) * sizeof(std::uint16_t);
    return 2 * (codes_per_weight + scales_per_weight);
}

double useful_bytes(std::int32_t k, std::int32_t tokens) {
    return static_cast<double>(pair_weight_bytes(k)) +
           2.0 * static_cast<double>((k + 2 * kRows) * tokens);
}

double useful_flops(std::int32_t k, std::int32_t tokens) {
    return 4.0 * static_cast<double>(kRows) * k * tokens;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const std::int32_t maximum_tokens =
            *std::max_element(options.tokens.begin(), options.tokens.end());

        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        const auto cuda = ninfer::bench::device_context().execution_view(stream);

        bench::L2FlushBuffer flush(options.flush_bytes);
        DeviceBuffer input =
            bench::make_bf16(static_cast<std::size_t>(options.k) * maximum_tokens, 101U);
        DeviceBuffer first_output(ninfer::bench::device_context().device(),
                                  static_cast<std::size_t>(kRows) * maximum_tokens * 2);
        DeviceBuffer second_output(ninfer::bench::device_context().device(),
                                   static_cast<std::size_t>(kRows) * maximum_tokens * 2);
        bench::PackedQuantizedWeight parent = bench::make_row_split_weight(
            QType::Q8_G32_FP16, kParentRows, options.k, options.k, 501U);
        const Weight first_weight  = bench::row_view(parent.weight, kFirstRow, kRows);
        const Weight second_weight = bench::row_view(parent.weight, kSecondRow, kRows);

        std::printf("# gpu=%s public=linear_pair shape=two_adjacent_Q8[1024,%d] "
                    "execution=%s cache=cold flush_bytes=%llu read_reference=%s_GB/s "
                    "bf16_tc_reference=%s_TFLOP/s\n",
                    properties.name, options.k,
                    options.execution == Execution::Graph ? "graph_replay" : "eager",
                    static_cast<unsigned long long>(options.flush_bytes),
                    bench::format_metric(bench::kRoofline.sustained_read_gbs, 1).c_str(),
                    bench::format_metric(bench::kRoofline.bf16_f32_tflops, 1).c_str());

        double t1_median = std::numeric_limits<double>::quiet_NaN();
        for (const std::int32_t tokens : options.tokens) {
            Tensor x(input.span(), DType::BF16, {options.k, tokens});
            Tensor first(first_output.span(), DType::BF16, {kRows, tokens});
            Tensor second(second_output.span(), DType::BF16, {kRows, tokens});
            const auto launch = [&](CudaExecutionView cuda) {
                ops::linear_pair(x, first_weight, second_weight, first, second, cuda);
            };
            bench::ColdTiming timing;
            if (options.execution == Execution::Graph) {
                bench::TimedGraph graph;
                graph.capture(cuda, launch);
                graph.launch(cuda);
                CUDA_CHECK(cudaStreamSynchronize(stream));
                timing =
                    bench::measure_cold_graph(graph, flush, cuda, options.warmup, options.repeat);
            } else {
                timing =
                    bench::measure_cold_launch(launch, flush, cuda, options.warmup, options.repeat);
            }
            if (tokens == 1) { t1_median = timing.median_us; }

            const double seconds       = timing.median_us * 1.0e-6;
            const double gbs           = useful_bytes(options.k, tokens) / seconds / 1.0e9;
            const double tflops        = useful_flops(options.k, tokens) / seconds / 1.0e12;
            const double extrapolation = std::isnan(t1_median)
                                             ? std::numeric_limits<double>::quiet_NaN()
                                             : tokens * t1_median / timing.median_us;
            std::printf(
                "T=%-3d median=%8.3f us min=%8.3f us p95=%8.3f us "
                "effective=%7.1f GB/s READ=%5s%% logical=%6.2f TFLOP/s ",
                tokens, timing.median_us, timing.min_us, timing.p95_us, gbs,
                bench::format_metric(100.0 * gbs / bench::kRoofline.sustained_read_gbs, 1).c_str(),
                tflops);
            const double tc_pct = tokens > 1 ? 100.0 * tflops / bench::kRoofline.bf16_f32_tflops
                                             : std::numeric_limits<double>::quiet_NaN();
            std::printf("TC=%5s%% ", bench::format_metric(tc_pct, 1).c_str());
            if (std::isnan(extrapolation)) {
                std::printf("T1_linear=n/a\n");
            } else {
                std::printf("T1_linear=%5.2fx\n", extrapolation);
            }
        }

        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_linear_pair_bench: %s\n", error.what());
        return 1;
    }
}
