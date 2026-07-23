// Exact small-T qualification benchmark for offset_i32_positions.
#include "ninfer/ops/position.h"
#include "core/device.h"
#include "ninfer_bench_common.h"
#include "ops/launcher/position.h"

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

std::vector<int> parse_tokens(const char* raw) {
    std::vector<int> values;
    const std::string text(raw);
    std::size_t begin = 0;
    while (begin < text.size()) {
        const std::size_t end  = text.find(',', begin);
        const std::string item = text.substr(begin, end == std::string::npos ? end : end - begin);
        const int value        = std::stoi(item);
        if (value <= 0) { throw std::invalid_argument("tokens must be positive"); }
        values.push_back(value);
        if (end == std::string::npos) { break; }
        begin = end + 1;
    }
    if (values.empty()) { throw std::invalid_argument("tokens must not be empty"); }
    return values;
}

Result bench_cold_graph(const launch_fn& launch, double bytes, DBuf& flush, int warmup,
                        int repeat) {
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    launch(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaGraph_t graph    = nullptr;
    cudaGraphExec_t exec = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    launch(stream);
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));

    cudaEvent_t begin = nullptr;
    cudaEvent_t end   = nullptr;
    CUDA_CHECK(cudaEventCreate(&begin));
    CUDA_CHECK(cudaEventCreate(&end));
    for (int i = 0; i < warmup; ++i) {
        CUDA_CHECK(cudaMemsetAsync(flush.p, 0xa5, flush.bytes, stream));
        CUDA_CHECK(cudaGraphLaunch(exec, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repeat));
    for (int i = 0; i < repeat; ++i) {
        CUDA_CHECK(cudaMemsetAsync(flush.p, 0xa5, flush.bytes, stream));
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

void run(int tokens, int candidate_block, DBuf* flush, int warmup, int repeat) {
    std::vector<std::int32_t> host(static_cast<std::size_t>(tokens));
    for (int i = 0; i < tokens; ++i) { host[static_cast<std::size_t>(i)] = 262144 - tokens + i; }
    DBuf source(host.size() * sizeof(std::int32_t));
    DBuf delta(sizeof(std::int32_t));
    DBuf destination(host.size() * sizeof(std::int32_t));
    const std::int32_t delta_value = -17;
    cudaMemcpy(source.p, host.data(), source.bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(delta.p, &delta_value, sizeof(delta_value), cudaMemcpyHostToDevice);
    Tensor tsource(source.p, DType::I32, {tokens});
    Tensor tdelta(delta.p, DType::I32, {1});
    Tensor tdestination(destination.p, DType::I32, {tokens});

    const auto launch = [&](cudaStream_t stream) {
        if (candidate_block == 0) {
            ops::offset_i32_positions(tsource, tdelta, tdestination, stream);
        } else {
            ops::detail::offset_i32_positions_block_launch(tsource, tdelta, tdestination,
                                                           candidate_block, stream);
        }
    };
    const double bytes  = static_cast<double>(tokens * 2 + 1) * sizeof(std::int32_t);
    const Result result = flush == nullptr
                              ? bench_loop(launch, bytes)
                              : bench_cold_graph(launch, bytes, *flush, warmup, repeat);

    const int block = candidate_block == 0 ? 256 : candidate_block;
    char label[128];
    std::snprintf(label, sizeof(label), "offset_i32_positions T=%d route=%s-b%d-g%d%s", tokens,
                  candidate_block == 0 ? "production" : "candidate", block,
                  (tokens + block - 1) / block, flush == nullptr ? "" : "-cold-graph");
    print_result(label, result);
}

} // namespace

int main(int argc, char** argv) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }

    std::vector<int> tokens{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    int candidate_block = 0;
    bool cold_graph     = false;
    int warmup          = 20;
    int repeat          = 200;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--tokens") && i + 1 < argc) {
            tokens = parse_tokens(argv[++i]);
        } else if (!std::strcmp(argv[i], "--candidate-block") && i + 1 < argc) {
            candidate_block = std::stoi(argv[++i]);
            if (candidate_block <= 0 || candidate_block > 1024 || candidate_block % 32 != 0) {
                throw std::invalid_argument(
                    "candidate block must be a positive multiple of 32 no larger than 1024");
            }
        } else if (!std::strcmp(argv[i], "--cold-graph")) {
            cold_graph = true;
        } else if (!std::strcmp(argv[i], "--warmup") && i + 1 < argc) {
            warmup = std::stoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--repeat") && i + 1 < argc) {
            repeat = std::stoi(argv[++i]);
        } else {
            throw std::invalid_argument("unknown position benchmark argument");
        }
    }
    if (warmup < 0 || repeat <= 0) {
        throw std::invalid_argument("--warmup must be nonnegative and --repeat positive");
    }
    DBuf flush(cold_graph ? 256ULL << 20 : 1);
    DBuf* flush_ptr = cold_graph ? &flush : nullptr;
    for (const int value : tokens) { run(value, candidate_block, flush_ptr, warmup, repeat); }
    return 0;
}
