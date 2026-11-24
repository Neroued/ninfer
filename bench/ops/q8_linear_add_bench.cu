// Cold-cache public-Op benchmark for Q8 LinearAdd.

#include "core/weight.h"
#include "core/buffer.h"
#include "ninfer/ops/linear_add.h"

#include "core/device.h"
#include "op_measurement.cuh"
#include "quantized_weight.cuh"

#include <cuda_runtime.h>

#include <algorithm>
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

constexpr std::int32_t kRows      = 2048;
constexpr std::size_t kFlushBytes = 256ULL << 20;

struct Options {
    std::int32_t hidden = 6144;
    std::vector<std::int32_t> t_sweep{
        1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16,  17,
        24,  31,  32,  33,  48,  63,  64,  65,  80,  95,  96,  97,  112, 127, 128, 129, 160,
        192, 224, 256, 320, 384, 448, 512, 576, 640, 641, 704, 768, 832, 896, 960, 1024};
    int warmup = 5;
    int repeat = 30;
    std::string csv_out;
    bool profile = false;
};

struct Result {
    std::int32_t t;
    bench::ColdTiming timing;
};

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        const auto next = [&](const char* label) -> std::string_view {
            if (++i >= argc) { throw std::invalid_argument(std::string("missing ") + label); }
            return argv[i];
        };
        if (arg == "--t-sweep") {
            options.t_sweep = bench::measurement::integers(next("--t-sweep value"));
        } else if (arg == "--k") {
            options.hidden = bench::measurement::integer(next("--k value"));
        } else if (arg == "--warmup") {
            options.warmup = bench::measurement::integer(next("--warmup value"), 0);
        } else if (arg == "--repeat") {
            options.repeat = bench::measurement::integer(next("--repeat value"));
        } else if (arg == "--csv-out") {
            options.csv_out = next("--csv-out path");
        } else if (arg == "--profile") {
            options.profile = true;
        } else if (arg == "--help" || arg == "-h") {
            std::printf("Usage: %s [--k 4096|6144] [--t-sweep 1,2,...] [--warmup N] [--repeat N] "
                        "[--csv-out PATH] [--profile]\n",
                        argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(arg));
        }
    }
    if (options.warmup < 0 || options.repeat <= 0) {
        throw std::invalid_argument("--warmup must be nonnegative and --repeat positive");
    }
    if (options.hidden != 4096 && options.hidden != 6144) {
        throw std::invalid_argument("--k must be 4096 or 6144");
    }
    if (options.profile && options.t_sweep.size() != 1) {
        throw std::invalid_argument("--profile requires exactly one T");
    }
    return options;
}

void append(std::vector<Result>& results, std::int32_t t, std::int32_t hidden,
            bench::ColdTiming timing, std::size_t weight_bytes) {
    const double useful_flops = 2.0 * kRows * hidden * static_cast<double>(t);
    const double useful_bytes =
        static_cast<double>(weight_bytes) + 2.0 * static_cast<double>((hidden + 2 * kRows) * t);
    const double tflops = useful_flops / timing.median_us / 1.0e6;
    const double gbs    = useful_bytes / timing.median_us / 1.0e3;
    std::printf("T=%-4d median=%8.3f us  useful=%7.2f TFLOP/s %7.1f GB/s\n", t, timing.median_us,
                tflops, gbs);
    results.push_back({t, timing});
}

void write_csv(const Options& options, const std::vector<Result>& results) {
    if (options.csv_out.empty()) { return; }
    const std::filesystem::path path(options.csv_out);
    if (!path.parent_path().empty()) { std::filesystem::create_directories(path.parent_path()); }
    std::ofstream out(path);
    out << "N,K,T,median_us,min_us,p95_us,warmup,repeat\n";
    for (const Result& result : results) {
        out << kRows << ',' << options.hidden << ',' << result.t << ',' << result.timing.median_us
            << ',' << result.timing.min_us << ',' << result.timing.p95_us << ',' << options.warmup
            << ',' << options.repeat << '\n';
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const std::int32_t max_t =
            *std::max_element(options.t_sweep.begin(), options.t_sweep.end());
        const std::int32_t min_t =
            *std::min_element(options.t_sweep.begin(), options.t_sweep.end());

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        const auto cuda = ninfer::bench::device_context().execution_view(stream);
        ninfer::bench::L2FlushBuffer flush(kFlushBytes);
        ninfer::DeviceBuffer input =
            bench::make_bf16(static_cast<std::size_t>(options.hidden) * max_t, 101U);
        ninfer::DeviceBuffer residual =
            bench::make_bf16(static_cast<std::size_t>(kRows) * max_t, 103U);
        bench::SavedBuffer residual_initial(residual);
        const auto restore = [&](CudaExecutionView cuda) { residual_initial.restore(cuda); };
        bench::PackedQuantizedWeight packed = bench::make_row_split_weight(
            QType::Q8_G32_FP16, kRows, options.hidden, options.hidden, 501U);
        const std::size_t workspace_capacity = ops::linear_add_workspace_capacity_bytes(
            QType::Q8_G32_FP16, kRows, options.hidden, min_t, max_t);
        ninfer::DeviceBuffer workspace_storage(ninfer::bench::device_context().device(),
                                               std::max<std::size_t>(workspace_capacity, 256));
        MemoryArena workspace(workspace_storage.span());
        std::vector<Result> results;

        for (const std::int32_t t : options.t_sweep) {
            Tensor x(input.span(), DType::BF16, {options.hidden, t});
            Tensor out(residual.span(), DType::BF16, {kRows, t});

            if (options.profile) {
                ops::linear_add(x, packed.weight, out, workspace, cuda);
                CUDA_CHECK(cudaStreamSynchronize(stream));
                std::printf("profile op=linear_add T=%d\n", t);
                continue;
            }

            const auto launch = [&](CudaExecutionView cuda) {
                ops::linear_add(x, packed.weight, out, workspace, cuda);
            };
            append(results, t, options.hidden,
                   bench::measure_cold_launch_prepared(restore, launch, flush, cuda, options.warmup,
                                                       options.repeat),
                   packed.storage.size());
        }

        CUDA_CHECK(cudaStreamSynchronize(stream));
        write_csv(options, results);
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_q8_linear_add_bench: %s\n", error.what());
        return 1;
    }
}
