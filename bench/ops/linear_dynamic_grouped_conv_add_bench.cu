#include "ninfer/ops/dynamic_grouped_conv.h"

#include "ninfer_bench_common.h"
#include "ops/dynamic_grouped_conv/w8/w8_dynamic_grouped_conv_add_plan.h"
#include "quantized_weight.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

constexpr std::int32_t kHidden           = 5120;
constexpr std::int32_t kMaximumWidth     = 16;
constexpr std::int32_t kGroups           = 320;
constexpr std::int32_t kTaps             = 2;
constexpr std::int32_t kSides            = 2;
constexpr std::size_t kDefaultFlushBytes = 256ULL << 20;

struct Options {
    int width               = 0;
    std::int32_t input_rows = 0;
    std::int32_t batch_size = 0;
    int warmup              = 8;
    int repeat              = 40;
    std::size_t flush_bytes = kDefaultFlushBytes;
};

Options parse_args(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const auto next = [&](const char* label) -> const char* {
            if (index + 1 >= argc) { throw std::invalid_argument(std::string("missing ") + label); }
            return argv[++index];
        };
        if (!std::strcmp(argv[index], "--width")) {
            options.width = std::atoi(next("width"));

        } else if (!std::strcmp(argv[index], "--warmup")) {
            options.warmup = std::atoi(next("warmup"));
        } else if (!std::strcmp(argv[index], "--repeat")) {
            options.repeat = std::atoi(next("repeat"));
        } else if (!std::strcmp(argv[index], "--k")) {
            options.input_rows = std::atoi(next("K"));
        } else if (!std::strcmp(argv[index], "--batch")) {
            options.batch_size = std::atoi(next("batch"));
        } else if (!std::strcmp(argv[index], "--flush-mib")) {
            const long mib = std::strtol(next("flush MiB"), nullptr, 10);
            if (mib <= 0) { throw std::invalid_argument("flush MiB must be positive"); }
            options.flush_bytes = static_cast<std::size_t>(mib) << 20;
        } else if (!std::strcmp(argv[index], "--help") || !std::strcmp(argv[index], "-h")) {
            std::printf("usage: %s [--width 2..16] [--k 4096|17408] [--batch 1..8] [--warmup N] "
                        "[--repeat N] "
                        "[--flush-mib N]\n",
                        argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(argv[index]));
        }
    }
    if (options.warmup < 0 || options.repeat <= 0) {
        throw std::invalid_argument("warmup must be nonnegative and repeat positive");
    }
    if (options.input_rows != 0 && options.input_rows != 4096 && options.input_rows != 17408) {
        throw std::invalid_argument("K must be 4096 or 17408");
    }
    if ((options.width && (options.width < 2 || options.width > 16)) || options.batch_size < 0 ||
        options.batch_size > 8) {
        throw std::invalid_argument("batch must be in [1,8]");
    }
    return options;
}

void run_profile(std::int32_t input_rows, const Options& options, DeviceBuffer& flush,
                 cudaStream_t stream) {
    DeviceBuffer input = make_bf16(static_cast<std::size_t>(input_rows) * kMaximumWidth * 8);
    DeviceBuffer base  = make_bf16(static_cast<std::size_t>(kHidden) * kTaps * kSides);
    DeviceBuffer delta = make_bf16(static_cast<std::size_t>(kGroups) * kTaps * kMaximumWidth * 8);
    DeviceBuffer residual        = make_bf16(static_cast<std::size_t>(kHidden) * kMaximumWidth * 8);
    PackedQuantizedWeight packed = make_row_split_weight(QType::W8G32_F16S, kHidden, input_rows,
                                                         input_rows, {0x31U, 0x00U, 0x1800U});
    const std::size_t capacity =
        ops::linear_dynamic_grouped_conv_add_workspace_capacity_bytes(input_rows, 2, 16, 1, 8);
    WorkspaceArena workspace(std::max<std::size_t>(capacity, 256));
    Tensor base_kernel(base.p, DType::BF16, {kHidden, kTaps, kSides});

    for (int width = 2; width <= 16; ++width) {
        if (options.width && options.width != width) continue;
        for (std::int32_t batch_size = 1; batch_size <= 8; ++batch_size) {
            if (options.batch_size != 0 && batch_size != options.batch_size) { continue; }
            const std::int32_t cols = width * batch_size;
            Tensor x(input.p, DType::BF16, {input_rows, width, batch_size});
            Tensor finish_delta(delta.p, DType::BF16, {kGroups, kTaps, width, batch_size});
            Tensor residual_view(residual.p, DType::BF16, {kHidden, width, batch_size});
            const double flops = 2.0 * static_cast<double>(kHidden) * input_rows * cols;
            const double bytes = static_cast<double>(packed.model_weight_bytes()) +
                                 2.0 * static_cast<double>(input_rows) * cols +
                                 2.0 * static_cast<double>(kHidden) * kTaps * kSides +
                                 2.0 * static_cast<double>(kGroups) * kTaps * cols +
                                 4.0 * static_cast<double>(kHidden) * cols;
            const auto measure = [&](const char* route, auto&& launch) {
                TimedGraph graph;
                workspace.reset();
                workspace.reset_peak();
                graph.capture(stream, launch);
                const ColdTiming timing =
                    measure_cold_graph(graph, flush, stream, options.warmup, options.repeat);
                const double tflops    = flops / timing.median_us / 1.0e6;
                const double bandwidth = bytes / timing.median_us / 1.0e3;
                std::printf("%d,%d,%d,%d,%s,%.3f,%.3f,%.3f,%.2f,%.1f,%zu,%zu\n", input_rows, width,
                            batch_size, cols, route, timing.median_us, timing.min_us, timing.p95_us,
                            tflops, bandwidth, graph.nodes(), workspace.peak_used());
            };
            const char* route = ops::detail::w8_linear_dynamic_grouped_conv_add_route_name(
                input_rows, width, batch_size);
            measure(route, [&](cudaStream_t capture_stream) {
                ops::linear_dynamic_grouped_conv_add(x, packed.weight, base_kernel, finish_delta,
                                                     residual_view, workspace, capture_stream);
            });
        }
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
        const Options options = parse_args(argc, argv);
        DeviceBuffer flush(options.flush_bytes);
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        std::printf("C,W,B,T,route,median_us,min_us,p95_us,effective_tflops,effective_gbs,"
                    "graph_nodes,workspace_bytes\n");
        for (const std::int32_t input_rows : std::array<std::int32_t, 2>{4096, 17408}) {
            if (options.input_rows != 0 && input_rows != options.input_rows) { continue; }
            run_profile(input_rows, options, flush, stream);
        }
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_linear_dynamic_grouped_conv_add_bench: %s\n", error.what());
        return 1;
    }
}
