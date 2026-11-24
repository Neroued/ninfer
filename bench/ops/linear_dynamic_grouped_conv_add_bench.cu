#include "core/weight.h"
#include "core/buffer.h"
#include "ninfer/ops/dynamic_grouped_conv.h"

#include "op_measurement.cuh"
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
            options.width = measurement::integer(next("width"), 0);
        } else if (!std::strcmp(argv[index], "--warmup")) {
            options.warmup = measurement::integer(next("warmup"), 0);
        } else if (!std::strcmp(argv[index], "--repeat")) {
            options.repeat = measurement::integer(next("repeat"));
        } else if (!std::strcmp(argv[index], "--k")) {
            options.input_rows = measurement::integer(next("K"), 0);
        } else if (!std::strcmp(argv[index], "--batch")) {
            options.batch_size = measurement::integer(next("batch"), 0);
        } else if (!std::strcmp(argv[index], "--flush-mib")) {
            const auto mib = measurement::integer<std::size_t>(
                next("flush MiB"), 1, std::numeric_limits<std::size_t>::max() >> 20);
            options.flush_bytes = mib << 20;
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

void run_profile(std::int32_t input_rows, const Options& options, bench::L2FlushBuffer& flush,
                 CudaExecutionView cuda) {
    DeviceBuffer input = make_bf16(static_cast<std::size_t>(input_rows) * kMaximumWidth * 8, 101U);
    DeviceBuffer base  = make_bf16(static_cast<std::size_t>(kHidden) * kTaps * kSides, 103U);
    DeviceBuffer delta =
        make_bf16(static_cast<std::size_t>(kGroups) * kTaps * kMaximumWidth * 8, 105U);
    DeviceBuffer residual = make_bf16(static_cast<std::size_t>(kHidden) * kMaximumWidth * 8, 107U);
    SavedBuffer residual_initial(residual);
    const auto restore = [&](CudaExecutionView cuda) { residual_initial.restore(cuda); };
    PackedQuantizedWeight packed =
        make_row_split_weight(QType::Q8_G32_FP16, kHidden, input_rows, input_rows, 501U);
    const std::size_t capacity =
        ops::linear_dynamic_grouped_conv_add_workspace_capacity_bytes(input_rows, 2, 16, 1, 8);
    ninfer::DeviceBuffer workspace_storage(ninfer::bench::device_context().device(),
                                           std::max<std::size_t>(capacity, 256));
    MemoryArena workspace(workspace_storage.span());
    Tensor base_kernel(base.span(), DType::BF16, {kHidden, kTaps, kSides});

    for (int width = 2; width <= 16; ++width) {
        if (options.width && options.width != width) continue;
        for (std::int32_t batch_size = 1; batch_size <= 8; ++batch_size) {
            if (options.batch_size != 0 && batch_size != options.batch_size) { continue; }
            const std::int32_t cols = width * batch_size;
            Tensor x(input.span(), DType::BF16, {input_rows, width, batch_size});
            Tensor finish_delta(delta.span(), DType::BF16, {kGroups, kTaps, width, batch_size});
            Tensor residual_view(residual.span(), DType::BF16, {kHidden, width, batch_size});
            const double flops = 2.0 * static_cast<double>(kHidden) * input_rows * cols;
            const double bytes = static_cast<double>(packed.model_weight_bytes()) +
                                 2.0 * static_cast<double>(input_rows) * cols +
                                 2.0 * static_cast<double>(kHidden) * kTaps * kSides +
                                 2.0 * static_cast<double>(kGroups) * kTaps * cols +
                                 4.0 * static_cast<double>(kHidden) * cols;
            const auto measure = [&](auto&& launch) {
                TimedGraph graph;
                workspace.reset();
                workspace.reset_peak();
                graph.capture(cuda, launch);
                const ColdTiming timing = measure_cold_graph_prepared(
                    restore, graph, flush, cuda, options.warmup, options.repeat);
                const double tflops    = flops / timing.median_us / 1.0e6;
                const double bandwidth = bytes / timing.median_us / 1.0e3;
                std::printf("%d,%d,%d,%d,%.3f,%.3f,%.3f,%.2f,%.1f,%zu,%zu\n", input_rows, width,
                            batch_size, cols, timing.median_us, timing.min_us, timing.p95_us,
                            tflops, bandwidth, graph.nodes(), workspace.peak_used());
            };
            measure([&](CudaExecutionView cuda) {
                ops::linear_dynamic_grouped_conv_add(x, packed.weight, base_kernel, finish_delta,
                                                     residual_view, workspace, cuda);
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
        bench::L2FlushBuffer flush(options.flush_bytes);
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        const auto cuda = ninfer::bench::device_context().execution_view(stream);
        std::printf("C,W,B,T,median_us,min_us,p95_us,effective_tflops,effective_gbs,"
                    "graph_nodes,workspace_bytes\n");
        for (const std::int32_t input_rows : std::array<std::int32_t, 2>{4096, 17408}) {
            if (options.input_rows != 0 && input_rows != options.input_rows) { continue; }
            run_profile(input_rows, options, flush, cuda);
        }
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_linear_dynamic_grouped_conv_add_bench: %s\n", error.what());
        return 1;
    }
}
