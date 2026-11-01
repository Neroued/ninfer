// Cold-cache public-Op benchmark for BF16 LinearAdd [5120,6144].

#include "common/roofline.h"
#include "core/weight.h"
#include "core/buffer.h"
#include "ninfer/ops/linear_add.h"

#include "core/device.h"
#include "direct_bf16_weight.cuh"
#include "op_measurement.cuh"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;

namespace {

constexpr std::int32_t kRows               = 5120;
constexpr std::int32_t kHidden             = 6144;
constexpr std::uint64_t kDefaultFlushBytes = 256ULL << 20;
constexpr int kDefaultWarmup               = 3;
constexpr int kDefaultRepeat               = 20;

struct Options {
    std::vector<std::int32_t> tokens{
        1,  2,  3,  4,  5,   6,   7,   8,   9,   10,   11,   12,   13,   14,   15, 16,
        17, 18, 19, 20, 21,  22,  23,  24,  25,  26,   27,   28,   29,   30,   31, 32,
        40, 48, 64, 96, 127, 128, 129, 256, 512, 1023, 1024, 1025, 1536, 2048,
    };
    int warmup                = kDefaultWarmup;
    int repeat                = kDefaultRepeat;
    std::uint64_t flush_bytes = kDefaultFlushBytes;
    bool profile              = false;
    std::string csv_out;
};

struct Result {
    std::int32_t tokens = 0;
    bench::ColdTiming timing{};
    std::uint64_t logical_bytes = 0;
    double useful_flops         = 0.0;
    double effective_gbs        = 0.0;
    double read_pct             = 0.0;
    double useful_tflops        = 0.0;
    double tc_pct               = 0.0;
    double memory_floor_us      = 0.0;
    double compute_floor_us     = 0.0;
    double roofline_pct         = 0.0;
};

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto next = [&](const char* label) -> std::string_view {
            if (++index >= argc) { throw std::invalid_argument(std::string("missing ") + label); }
            return argv[index];
        };
        if (argument == "--t-sweep") {
            options.tokens = bench::measurement::integers(next("--t-sweep value"));
        } else if (argument == "--sweep") {
            const auto range = next("--sweep value");
            if (std::count(range.begin(), range.end(), ':') != 2)
                throw std::invalid_argument("--sweep must be BEGIN:END:STEP");
            options.tokens = bench::measurement::sweep(range);
        } else if (argument == "--warmup") {
            options.warmup = bench::measurement::integer(next("--warmup value"), 0);
        } else if (argument == "--repeat") {
            options.repeat = bench::measurement::integer(next("--repeat value"));
        } else if (argument == "--flush-bytes") {
            options.flush_bytes =
                bench::measurement::integer<std::uint64_t>(next("--flush-bytes value"));
        } else if (argument == "--csv-out") {
            options.csv_out = next("--csv-out path");
        } else if (argument == "--profile") {
            options.profile = true;
        } else if (argument == "--help" || argument == "-h") {
            std::printf("Usage: %s [--t-sweep 1,4,... | --sweep BEGIN:END:STEP] "
                        "[--warmup N] [--repeat N] "
                        "[--flush-bytes N] [--csv-out PATH] [--profile]\n",
                        argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(argument));
        }
    }
    if (options.tokens.empty() || options.warmup < 0 || options.repeat <= 0 ||
        options.flush_bytes == 0) {
        throw std::invalid_argument("invalid empty sweep, timing count, or flush size");
    }
    std::sort(options.tokens.begin(), options.tokens.end());
    options.tokens.erase(std::unique(options.tokens.begin(), options.tokens.end()),
                         options.tokens.end());
    if (options.profile && options.tokens.size() != 1) {
        throw std::invalid_argument("--profile requires one T");
    }
    return options;
}

Result make_result(std::int32_t tokens, const bench::ColdTiming& timing,
                   std::uint64_t weight_bytes) {
    const std::uint64_t x_bytes        = 2ULL * static_cast<std::uint64_t>(kHidden) * tokens;
    const std::uint64_t residual_bytes = 2ULL * static_cast<std::uint64_t>(kRows) * tokens;
    const std::uint64_t logical_bytes  = weight_bytes + x_bytes + 2ULL * residual_bytes;
    const double useful_flops =
        2.0 * static_cast<double>(kRows) * static_cast<double>(kHidden) * tokens;
    const double seconds = timing.median_us * 1.0e-6;
    const double memory_floor_us =
        static_cast<double>(logical_bytes) / (bench::kRoofline.memory_gbs * 1.0e9) * 1.0e6;
    // T=1..4 use SIMT; only the larger-T routes execute BF16 MMA.
    const double tensor_peak =
        tokens >= 5 ? bench::kRoofline.bf16_f32_tflops : std::numeric_limits<double>::quiet_NaN();
    const double compute_floor_us  = useful_flops / (tensor_peak * 1.0e6);
    const double roofline_floor_us = std::isfinite(compute_floor_us)
                                         ? std::max(memory_floor_us, compute_floor_us)
                                         : std::numeric_limits<double>::quiet_NaN();

    return {
        tokens,
        timing,
        logical_bytes,
        useful_flops,
        static_cast<double>(logical_bytes) / seconds / 1.0e9,
        (static_cast<double>(logical_bytes) / seconds / 1.0e9) /
            bench::kRoofline.sustained_read_gbs * 100.0,
        useful_flops / seconds / 1.0e12,
        (useful_flops / seconds / 1.0e12) / tensor_peak * 100.0,
        memory_floor_us,
        compute_floor_us,
        roofline_floor_us / timing.median_us * 100.0,
    };
}

void print_result(const Result& result) {
    const char* bound =
        !std::isfinite(result.compute_floor_us)
            ? "-"
            : (result.memory_floor_us >= result.compute_floor_us ? "memory" : "compute");
    std::printf("T=%-4d median=%8.3f us  %7.1f GB/s READ=%6s%%  "
                "%7.2f TFLOP/s TC=%6s%%  mem_floor=%7.3f us %-7s roofline=%6s%%\n",
                result.tokens, result.timing.median_us, result.effective_gbs,
                bench::format_metric(result.read_pct).c_str(), result.useful_tflops,
                bench::format_metric(result.tc_pct).c_str(), result.memory_floor_us, bound,
                bench::format_metric(result.roofline_pct).c_str());
}

void write_csv(const Options& options, const std::vector<Result>& results) {
    if (options.csv_out.empty()) { return; }
    const std::filesystem::path path(options.csv_out);
    if (!path.parent_path().empty()) { std::filesystem::create_directories(path.parent_path()); }
    std::ofstream out(path);
    out << "T,median_us,min_us,p95_us,logical_bytes,useful_flops,effective_gbs,"
           "sustained_read_pct,useful_tflops,bf16_tc_pct,memory_floor_us,compute_floor_us,"
           "roofline_pct,warmup,repeat,flush_bytes\n";
    for (const Result& result : results) {
        out << result.tokens << ',' << result.timing.median_us << ',' << result.timing.min_us << ','
            << result.timing.p95_us << ',' << result.logical_bytes << ',' << result.useful_flops
            << ',' << result.effective_gbs << ',' << bench::format_metric(result.read_pct, 6, "")
            << ',' << result.useful_tflops << ',' << bench::format_metric(result.tc_pct, 6, "")
            << ',' << result.memory_floor_us << ','
            << bench::format_metric(result.compute_floor_us, 6, "") << ','
            << bench::format_metric(result.roofline_pct, 6, "") << ',' << options.warmup << ','
            << options.repeat << ',' << options.flush_bytes << '\n';
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options         = parse_options(argc, argv);
        const std::int32_t max_tokens = options.tokens.back();
        cudaStream_t stream           = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        const auto cuda = ninfer::bench::device_context().execution_view(stream);

        bench::L2FlushBuffer flush(options.flush_bytes);
        DeviceBuffer input = bench::make_bf16(static_cast<std::size_t>(kHidden) * max_tokens, 101U);
        DeviceBuffer residual =
            bench::make_bf16(static_cast<std::size_t>(kRows) * max_tokens, 103U);
        bench::SavedBuffer residual_initial(residual);
        const auto restore = [&](CudaExecutionView cuda) { residual_initial.restore(cuda); };
        bench::DirectBf16Weight weight = bench::make_direct_bf16_weight(kRows, kHidden, 0x61U);
        ninfer::DeviceBuffer workspace_storage(ninfer::bench::device_context().device(), 1);
        MemoryArena workspace(workspace_storage.span());

        if (options.profile) {
            const std::int32_t tokens = options.tokens.front();
            Tensor x(input.span(), DType::BF16, {kHidden, tokens});
            Tensor out(residual.span(), DType::BF16, {kRows, tokens});
            for (int index = 0; index < options.warmup; ++index) {
                restore(cuda);
                bench::flush_l2(flush, cuda);
                ops::linear_add(x, weight.weight, out, workspace, cuda);
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
            restore(cuda);
            bench::flush_l2(flush, cuda);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            std::printf("PROFILE op=linear_add T=%d\n", tokens);
            CUDA_CHECK(cudaProfilerStart());
            ops::linear_add(x, weight.weight, out, workspace, cuda);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaProfilerStop());
            CUDA_CHECK(cudaStreamDestroy(stream));
            return 0;
        }

        std::vector<Result> results;
        for (const std::int32_t tokens : options.tokens) {
            Tensor x(input.span(), DType::BF16, {kHidden, tokens});
            Tensor out(residual.span(), DType::BF16, {kRows, tokens});
            const auto launch = [&](CudaExecutionView cuda) {
                ops::linear_add(x, weight.weight, out, workspace, cuda);
            };
            Result result =
                make_result(tokens,
                            bench::measure_cold_launch_prepared(restore, launch, flush, cuda,
                                                                options.warmup, options.repeat),
                            weight.model_weight_bytes());
            print_result(result);
            results.push_back(std::move(result));
        }
        write_csv(options, results);
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_bf16_linear_add_bench: %s\n", error.what());
        return 1;
    }
}
