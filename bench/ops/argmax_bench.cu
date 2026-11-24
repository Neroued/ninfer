#include "ninfer/ops/argmax.h"
#include "core/buffer.h"
#include "op_measurement.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

constexpr std::int32_t kFullPhysicalRows = 248320;
constexpr std::int32_t kFullValidRows    = 248077;
constexpr std::int32_t kShortlistRows    = 131072;

struct Options {
    measurement::Options timing;
    std::string shape;
    int cols = 0;
};

void usage(const char* argv0) {
    std::printf("usage: %s [--shape full|shortlist --cols N]\n", argv0);
    measurement::help();
}

int parse_int(std::string_view value, const char* name) {
    try {
        std::size_t parsed = 0;
        const int out      = std::stoi(std::string(value), &parsed);
        if (parsed != value.size()) { throw std::invalid_argument("trailing"); }
        return out;
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string(name) + " expects an integer");
    }
}

Options parse_args(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        auto need_value = [&](const char* name) -> std::string_view {
            if (i + 1 >= argc) {
                throw std::invalid_argument(std::string(name) + " needs a value");
            }
            return argv[++i];
        };
        if (arg == "--shape") {
            options.shape = need_value("--shape");
        } else if (arg == "--cols") {
            options.cols = parse_int(need_value("--cols"), "--cols");
        } else if (arg == "-h" || arg == "--help") {
            usage(argc > 0 ? argv[0] : "ninfer_argmax_bench");
            std::exit(0);
        } else if (!measurement::parse(options.timing, arg, need_value("option"))) {
            throw std::invalid_argument("unknown argument: " + std::string(arg));
        }
    }
    if (options.shape.empty()) {
        if (options.cols != 0) { throw std::invalid_argument("--cols requires --shape"); }
        return options;
    }
    if (options.shape != "full" && options.shape != "shortlist") {
        throw std::invalid_argument("--shape must be full or shortlist");
    }
    if (options.cols == 0) { options.cols = 1; }
    if (options.cols < 1) throw std::invalid_argument("columns must be positive");
    return options;
}

void run_shape(std::int32_t physical_rows, std::int32_t valid_rows, int cols,
               measurement::LogicalBenchmark& benchmark) {
    DeviceBuffer logits = make_bf16(static_cast<std::size_t>(physical_rows) * cols, 101U);
    DeviceBuffer out    = make_zeros(static_cast<std::size_t>(cols) * sizeof(std::int32_t));
    Tensor tout(out.span(), DType::I32, {cols});

    Tensor tlogits(logits.span(), DType::BF16, {physical_rows, cols});
    benchmark.run(
        {.op = "argmax", .format = "bf16", .n = physical_rows, .k = valid_rows, .tokens = cols},
        std::size_t(cols) * (valid_rows * 2 + 4),
        [&](CudaExecutionView cuda) { ops::argmax(tlogits, tout, valid_rows, cuda); },
        [](CudaExecutionView) {});
}

} // namespace

int main(int argc, char** argv) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }

    try {
        const Options options = parse_args(argc, argv);
        measurement::hardware();
        measurement::LogicalBenchmark benchmark(options.timing);
        if (options.shape.empty()) {
            for (int cols = 1; cols <= 16; ++cols) {
                run_shape(kFullPhysicalRows, kFullValidRows, cols, benchmark);
            }
            run_shape(kFullPhysicalRows, kFullValidRows, 128, benchmark);
            for (const int cols : {1, 120}) {
                run_shape(kShortlistRows, kShortlistRows, cols, benchmark);
            }
        } else if (options.shape == "full") {
            run_shape(kFullPhysicalRows, kFullValidRows, options.cols, benchmark);
        } else {
            run_shape(kShortlistRows, kShortlistRows, options.cols, benchmark);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ninfer_argmax_bench: %s\n", e.what());
        return 2;
    }
    return 0;
}
