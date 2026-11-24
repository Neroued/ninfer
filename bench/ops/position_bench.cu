// Exact small-T qualification benchmark for offset_i32_positions.
#include "ninfer/ops/position.h"
#include "core/buffer.h"
#include "core/device.h"
#include "op_measurement.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

Result bench_cold_graph(const launch_fn& launch, double bytes, bench::L2FlushBuffer& flush,
                        int warmup, int repeat) {
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const auto cuda = ninfer::bench::device_context().execution_view(stream);
    launch(cuda);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaGraph_t graph    = nullptr;
    cudaGraphExec_t exec = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    launch(cuda);
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));

    cudaEvent_t begin = nullptr;
    cudaEvent_t end   = nullptr;
    CUDA_CHECK(cudaEventCreate(&begin));
    CUDA_CHECK(cudaEventCreate(&end));
    for (int i = 0; i < warmup; ++i) {
        bench::flush_l2(flush, cuda);
        CUDA_CHECK(cudaGraphLaunch(exec, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repeat));
    for (int i = 0; i < repeat; ++i) {
        bench::flush_l2(flush, cuda);
        CUDA_CHECK(cudaEventRecord(begin, stream));
        CUDA_CHECK(cudaGraphLaunch(exec, stream));
        CUDA_CHECK(cudaEventRecord(end, stream));
        CUDA_CHECK(cudaEventSynchronize(end));
        float milliseconds = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, begin, end));
        samples.push_back(static_cast<double>(milliseconds) * 1000.0);
    }
    CUDA_CHECK(cudaEventDestroy(begin));
    CUDA_CHECK(cudaEventDestroy(end));
    CUDA_CHECK(cudaGraphExecDestroy(exec));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(stream));

    std::sort(samples.begin(), samples.end());
    Result result;
    result.n_runs    = repeat;
    result.median_us = samples[samples.size() / 2];
    result.min_us    = samples.front();
    result.p95_us    = samples[std::min(
        samples.size() - 1, static_cast<std::size_t>(0.95 * static_cast<double>(samples.size())))];
    result.gbs       = bytes / (result.median_us * 1.0e-6) / 1.0e9;
    return result;
}

void run(int tokens, bench::L2FlushBuffer* flush, int warmup, int repeat) {
    std::vector<std::int32_t> host(static_cast<std::size_t>(tokens));
    for (int i = 0; i < tokens; ++i) { host[static_cast<std::size_t>(i)] = 262144 - tokens + i; }
    DeviceBuffer source(ninfer::bench::device_context().device(),
                        host.size() * sizeof(std::int32_t));
    DeviceBuffer delta(ninfer::bench::device_context().device(), sizeof(std::int32_t));
    DeviceBuffer destination(ninfer::bench::device_context().device(),
                             host.size() * sizeof(std::int32_t));
    const std::int32_t delta_value = -17;
    cudaMemcpy(source.data(), host.data(), source.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(delta.data(), &delta_value, sizeof(delta_value), cudaMemcpyHostToDevice);
    Tensor tsource(source.span(), DType::I32, {tokens});
    Tensor tdelta(delta.span(), DType::I32, {1});
    Tensor tdestination(destination.span(), DType::I32, {tokens});

    const auto launch = [&](CudaExecutionView cuda) {
        ops::offset_i32_positions(tsource, tdelta, tdestination, cuda);
    };
    const double bytes  = static_cast<double>(tokens * 2 + 1) * sizeof(std::int32_t);
    const Result result = flush == nullptr
                              ? bench_loop(launch, bytes)
                              : bench_cold_graph(launch, bytes, *flush, warmup, repeat);

    char label[128];
    std::snprintf(label, sizeof(label), "offset_i32_positions T=%d%s", tokens,
                  flush == nullptr ? "" : " cold-graph");
    print_result(label, result);
}

} // namespace

int main(int argc, char** argv) try {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }

    std::vector<int> tokens{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    bool cold_graph = false;
    int warmup      = 20;
    int repeat      = 200;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--tokens") && i + 1 < argc) {
            tokens = measurement::integers(argv[++i]);
        } else if (!std::strcmp(argv[i], "--cold-graph")) {
            cold_graph = true;
        } else if (!std::strcmp(argv[i], "--warmup") && i + 1 < argc) {
            warmup = measurement::integer(argv[++i], 0);
        } else if (!std::strcmp(argv[i], "--repeat") && i + 1 < argc) {
            repeat = measurement::integer(argv[++i]);
        } else {
            throw std::invalid_argument("unknown position benchmark argument");
        }
    }
    if (warmup < 0 || repeat <= 0) {
        throw std::invalid_argument("--warmup must be nonnegative and --repeat positive");
    }
    bench::L2FlushBuffer flush(cold_graph ? 256ULL << 20 : 1);
    bench::L2FlushBuffer* flush_ptr = cold_graph ? &flush : nullptr;
    for (const int value : tokens) { run(value, flush_ptr, warmup, repeat); }
    return 0;
} catch (const std::exception& error) {

    std::fprintf(stderr, "ninfer_position_bench: %s\n", error.what());
    return 2;
}
