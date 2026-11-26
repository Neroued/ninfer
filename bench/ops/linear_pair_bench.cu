#include "common/roofline.h"
#include "core/weight.h"
#include "core/buffer.h"
#include "ninfer/ops/linear_pair.h"

#include "core/device.h"
#include "op_measurement.cuh"
#include "quantized_weight.cuh"
#include "direct_bf16_weight.cuh"
#include <optional>

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
    std::optional<QType> format;
    std::int32_t n            = 0;
    std::int32_t k            = 0;
    Execution execution       = Execution::Graph;
    int warmup                = 5;
    int repeat                = 30;
    std::uint64_t flush_bytes = 256ULL << 20;
    std::string csv_out;
    bool profile = false;
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
        } else if (argument == "--format") {
            const auto value = next("--format value");
            if (value == "bf16")
                options.format = QType::BF16;
            else if (value == "q8")
                options.format = QType::Q8_G32_FP16;
            else
                throw std::invalid_argument("--format must be bf16 or q8");
        } else if (argument == "--n") {
            options.n = bench::measurement::integer(next("--n value"));
        } else if (argument == "--k") {
            options.k = bench::measurement::integer(next("--k value"));
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
        } else if (argument == "--csv-out") {
            options.csv_out = next("--csv-out value");
        } else if (argument == "--profile") {
            options.profile = true;
        } else if (argument == "--help" || argument == "-h") {
            std::printf(
                "Usage: %s --k K [--format bf16|q8] [--n N] "
                "[--tokens 1,2,... | --sweep START:END[:STEP]] "
                "[--execution eager|graph] [--warmup N] [--repeat N] [--flush-mib N] "
                "[--csv-out PATH] [--profile]\n"
                "Defaults by K are BF16 N=4096 for K=128 and Q8 N=1024 for K=2048 or 5120.\n"
                "BF16 also accepts N=8192 for K=128 and uses distinct inputs. Q8 uses one shared "
                "input.\n"
#if NINFER_TARGET_RTX5090
                "For BF16 K=2304, N defaults to 1024.\n"
#endif
                "Default timing is cold graph body. --profile selects one post-warmup call.\n",
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
    if (!options.format) {
        options.format = options.k == 128 || options.k == 2304 ? QType::BF16 : QType::Q8_G32_FP16;
    }
    if (options.n == 0) { options.n = options.k == 128 ? 4096 : kRows; }
    if (options.repeat <= 0) { throw std::invalid_argument("--repeat must be positive"); }
    if (options.profile && options.tokens.size() != 1)
        throw std::invalid_argument("--profile requires exactly one token count");
    return options;
}

std::uint64_t useful_bytes(const ops::LinearProfile& shape, std::int32_t tokens) {
    const auto nk = std::uint64_t(shape.n) * shape.k;
    const auto kt = std::uint64_t(shape.k) * tokens;
    const auto nt = std::uint64_t(shape.n) * tokens;
    if (shape.qtype == QType::BF16) return 2 * (2 * nk + 2 * kt + 2 * nt);
    return 2 * (nk + nk / 32 * sizeof(std::uint16_t)) + 2 * (kt + 2 * nt);
}

std::uint64_t useful_flops(const ops::LinearProfile& shape, std::int32_t tokens) {
    return 4ULL * shape.n * shape.k * tokens;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const auto [minimum, maximum] =
            std::minmax_element(options.tokens.begin(), options.tokens.end());
        const int maximum_tokens = *maximum;
        const ops::LinearProfile shape{*options.format, options.n, options.k};
        (void)ops::linear_pair_workspace_capacity_bytes(shape, shape, *minimum, maximum_tokens);
        const bool bf16 = shape.qtype == QType::BF16;

        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        const auto cuda = ninfer::bench::device_context().execution_view(stream);

        bench::L2FlushBuffer flush(options.flush_bytes);
        const int rows = options.n;
        DeviceBuffer second_input =
            bf16 ? bench::make_bf16(std::size_t(options.k) * maximum_tokens, 397U) : DeviceBuffer{};
        DeviceBuffer input =
            bench::make_bf16(static_cast<std::size_t>(options.k) * maximum_tokens, 101U);
        DeviceBuffer first_output(ninfer::bench::device_context().device(),
                                  static_cast<std::size_t>(rows) * maximum_tokens * 2);
        DeviceBuffer second_output(ninfer::bench::device_context().device(),
                                   static_cast<std::size_t>(rows) * maximum_tokens * 2);
        std::optional<bench::PackedQuantizedWeight> parent;
        std::optional<bench::DirectBf16Weight> w0, w1;
        Weight first_weight, second_weight;
        if (bf16) {
            w0.emplace(bench::make_direct_bf16_weight(rows, options.k, 501U));
            w1.emplace(bench::make_direct_bf16_weight(rows, options.k, 723U));
            first_weight  = w0->weight;
            second_weight = w1->weight;
        } else {
            parent.emplace(bench::make_row_split_weight(QType::Q8_G32_FP16, kParentRows, options.k,
                                                        options.k, 501U));
            first_weight  = bench::row_view(parent->weight, kFirstRow, kRows);
            second_weight = bench::row_view(parent->weight, kSecondRow, kRows);
        }

        std::printf("# gpu=%s public=linear_pair format=%s shape=two[%d,%d] "
                    "execution=%s cache=cold flush_bytes=%llu read_reference=%s_GB/s "
                    "bf16_tc_reference=%s_TFLOP/s\n",
                    properties.name, bf16 ? "bf16" : "q8", rows, options.k,
                    options.execution == Execution::Graph ? "graph_replay" : "eager",
                    static_cast<unsigned long long>(options.flush_bytes),
                    bench::format_metric(bench::kRoofline.sustained_read_gbs, 1).c_str(),
                    bench::format_metric(bench::kRoofline.bf16_f32_tflops, 1).c_str());

        double t1_median = std::numeric_limits<double>::quiet_NaN();
        bench::measurement::Options timing_options;
        timing_options.warmup  = options.warmup;
        timing_options.repeat  = options.repeat;
        timing_options.profile = options.profile;
        std::optional<bench::measurement::Csv> csv;
        if (!options.csv_out.empty())
            csv.emplace(options.csv_out, "logical_bytes,useful_flops,effective_gb_s,useful_tflops,"
                                         "tensor_peak_tflops");
        for (const std::int32_t tokens : options.tokens) {
            Tensor x(input.span(), DType::BF16, {options.k, tokens});
            const Tensor second_x =
                bf16 ? Tensor(second_input.span(), DType::BF16, {options.k, tokens}) : x;
            Tensor first(first_output.span(), DType::BF16, {rows, tokens});
            Tensor second(second_output.span(), DType::BF16, {rows, tokens});
            const auto launch = [&](CudaExecutionView cuda) {
                ops::linear_pair(x, second_x, first_weight, second_weight, first, second, cuda);
            };
            const std::string execution = options.execution == Execution::Graph ? "graph" : "eager";
            const auto result           = bench::measurement::measure(
                launch, [](auto) {}, flush, cuda, timing_options, execution, "cold");
            const auto& timing = result.gpu;
            if (tokens == 1) { t1_median = timing.median_us; }

            const double seconds       = timing.median_us * 1.0e-6;
            const auto bytes           = useful_bytes(shape, tokens);
            const auto flops           = useful_flops(shape, tokens);
            const double gbs           = bytes / seconds / 1.0e9;
            const double tflops        = flops / seconds / 1.0e12;
            const double extrapolation = std::isnan(t1_median)
                                             ? std::numeric_limits<double>::quiet_NaN()
                                             : tokens * t1_median / timing.median_us;
            std::printf(
                "T=%-3d median=%8.3f us min=%8.3f us p95=%8.3f us "
                "effective=%7.1f GB/s READ=%5s%% logical=%6.2f TFLOP/s ",
                tokens, timing.median_us, timing.min_us, timing.p95_us, gbs,
                bench::format_metric(100.0 * gbs / bench::kRoofline.sustained_read_gbs, 1).c_str(),
                tflops);
            // The K=2304 decode range can include SIMT routes.
            const bool report_tc =
                options.k == 2304 ? tokens >= 128 : options.k == 128 || tokens > 1;
            const double tc_pct = report_tc ? 100.0 * tflops / bench::kRoofline.bf16_f32_tflops
                                            : std::numeric_limits<double>::quiet_NaN();
            std::printf("TC=%5s%% ", bench::format_metric(tc_pct, 1).c_str());
            if (std::isnan(extrapolation)) {
                std::printf("T1_linear=n/a\n");
            } else {
                std::printf("T1_linear=%5.2fx\n", extrapolation);
            }
            bench::measurement::Row row;
            row.op     = "linear_pair";
            row.format = bf16 ? "bf16" : "q8";
            row.policy = "a16";
            row.n      = rows;
            row.k      = options.k;
            row.tokens = tokens;
            if (csv)
                csv->write(row, result, execution, "cold",
                           std::to_string(bytes) + ',' + std::to_string(flops) + ',' +
                               bench::format_metric(gbs, 3, "") + ',' +
                               bench::format_metric(tflops, 3, "") + ',' +
                               (report_tc
                                    ? bench::format_metric(bench::kRoofline.bf16_f32_tflops, 3, "")
                                    : ""));
        }

        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_linear_pair_bench: %s\n", error.what());
        return 1;
    }
}
