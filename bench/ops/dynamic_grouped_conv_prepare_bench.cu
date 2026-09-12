#include "ninfer/ops/dynamic_grouped_conv.h"

#include "direct_bf16_weight.cuh"
#include "ninfer_bench_common.h"

#include <cuda_runtime.h>

#include <algorithm>
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
constexpr std::int32_t kCoefficientRows  = 1280;
constexpr std::size_t kDefaultFlushBytes = 256ULL << 20;

struct Options {
    int width               = 0;
    int batch               = 0;
    int warmup              = 8;
    int repeat              = 40;
    std::size_t flush_bytes = kDefaultFlushBytes;
};

Options parse_args(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        auto next = [&](const char* label) -> const char* {
            if (index + 1 >= argc) { throw std::invalid_argument(std::string("missing ") + label); }
            return argv[++index];
        };
        if (!std::strcmp(argv[index], "--width")) {
            options.width = std::atoi(next("width"));
        } else if (!std::strcmp(argv[index], "--batch")) {
            options.batch = std::atoi(next("batch"));

        } else if (!std::strcmp(argv[index], "--warmup")) {
            options.warmup = std::atoi(next("warmup"));
        } else if (!std::strcmp(argv[index], "--repeat")) {
            options.repeat = std::atoi(next("repeat"));
        } else if (!std::strcmp(argv[index], "--flush-mib")) {
            const long mib = std::strtol(next("flush MiB"), nullptr, 10);
            if (mib <= 0) { throw std::invalid_argument("flush MiB must be positive"); }
            options.flush_bytes = static_cast<std::size_t>(mib) << 20;
        } else if (!std::strcmp(argv[index], "--help") || !std::strcmp(argv[index], "-h")) {
            std::printf(
                "usage: %s [--width W] [--batch B] [--warmup N] [--repeat N] [--flush-mib N]\n",
                argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(argv[index]));
        }
    }
    if ((options.width != 0 && (options.width < 2 || options.width > 16)) || options.batch < 0 ||
        options.batch > 8 || options.warmup < 0 || options.repeat <= 0) {
        throw std::invalid_argument("warmup must be nonnegative and repeat positive");
    }
    return options;
}

void run_batch(int width, std::int32_t batch_size, const Options& options,
               const Tensor& norm_weight, const Tensor& base_kernel,
               const Weight& projection_weight, DeviceBuffer& residual_storage,
               DeviceBuffer& prepared_storage, DeviceBuffer& finish_storage,
               WorkspaceArena& workspace, DeviceBuffer& flush, cudaStream_t stream) {
    const std::int32_t tokens = width * batch_size;
    Tensor residual(residual_storage.p, DType::BF16, {kHidden, width, batch_size});
    Tensor prepared(prepared_storage.p, DType::BF16, {kHidden, width, batch_size});
    Tensor finish_delta(finish_storage.p, DType::BF16, {kGroups, kTaps, width, batch_size});
    const auto launch = [&](cudaStream_t launch_stream) {
        workspace.reset();
        ops::rmsnorm_dynamic_grouped_conv_prepare(residual, norm_weight, 1.0e-6F, base_kernel,
                                                  projection_weight, prepared, finish_delta,
                                                  workspace, launch_stream);
    };
    workspace.reset_peak();
    TimedGraph graph;
    graph.capture(stream, launch);
    const ColdTiming timing = measure_cold_launch([&](cudaStream_t s) { graph.launch(s); }, flush,
                                                  stream, options.warmup, options.repeat);
    const double seconds    = timing.median_us * 1.0e-6;
    const double projection_flops = 2.0 * kCoefficientRows * kHidden * static_cast<double>(tokens);
    const double effective_tflops = projection_flops / seconds / 1.0e12;
    const std::size_t workspace_bytes = workspace.peak_used();
    std::printf("%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%zu,%zu\n", width, batch_size, tokens,
                timing.median_us, timing.min_us, timing.p95_us, effective_tflops, workspace_bytes,
                graph.nodes());
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
        DeviceBuffer residual = make_bf16(static_cast<std::size_t>(kHidden) * kMaximumWidth * 8);
        DeviceBuffer norm     = make_bf16(kHidden);
        DeviceBuffer base     = make_bf16(static_cast<std::size_t>(kHidden) * kTaps * kSides);
        DirectBf16Weight projection = make_direct_bf16_weight(kCoefficientRows, kHidden, 0x31U);
        DeviceBuffer prepared = make_zeros(static_cast<std::size_t>(kHidden) * kMaximumWidth * 8 *
                                           sizeof(std::uint16_t));
        DeviceBuffer finish = make_zeros(static_cast<std::size_t>(kGroups) * kTaps * kMaximumWidth *
                                         8 * sizeof(std::uint16_t));
        DeviceBuffer flush(options.flush_bytes);
        const std::size_t workspace_bytes =
            ops::rmsnorm_dynamic_grouped_conv_prepare_workspace_capacity_bytes(2, 16, 1, 8);
        WorkspaceArena workspace(workspace_bytes);
        Tensor norm_weight(norm.p, DType::BF16, {kHidden});
        Tensor base_kernel(base.p, DType::BF16, {kHidden, kTaps, kSides});

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        std::printf("W,B,T,median_us,min_us,p95_us,effective_projection_tflops,workspace_bytes,"
                    "graph_nodes\n");
        for (int width = 2; width <= 16; ++width) {
            if (options.width && options.width != width) continue;
            for (std::int32_t batch_size = 1; batch_size <= 8; ++batch_size) {
                if (options.batch && options.batch != batch_size) continue;
                run_batch(width, batch_size, options, norm_weight, base_kernel, projection.weight,
                          residual, prepared, finish, workspace, flush, stream);
            }
        }
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_dynamic_grouped_conv_prepare_bench: %s\n", error.what());
        return 1;
    }
}
