// Checks grouped MoE assignment offsets and ordered expert/column/slot jobs, including guards,
// empty/skewed counts, tile boundaries and changing eager/graph inputs.
// Oracle: exact CPU prefix sums and independent enumeration of 64-column jobs.
// Implements: moe_grouped_nvfp4 metadata qualification (include/ninfer/ops/moe_grouped.h).
#include "core/arena.h"
#include "core/device.h"
#include "ops/op_tester.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <exception>
#include <random>
#include <stdexcept>
#include <vector>

#include "ops/moe_grouped/grouped_scan.cuh"
#include "ops/metadata_timing.cuh"

using namespace ninfer;
using namespace ninfer::ops::detail;

void run_case(int pattern, cudaStream_t stream, bool benchmark) {
    constexpr int experts = 512, capacity = 66000, guard = -1234567;
    DeviceBuffer inputs(2 * experts * sizeof(int));
    DeviceBuffer results((experts + 1 + 3 * capacity + 1) * sizeof(int));
    auto* counts      = static_cast<int*>(inputs.p);
    auto* table       = counts + experts;
    auto* offsets     = static_cast<int*>(results.p);
    auto* job_experts = offsets + experts + 1;
    auto* job_columns = job_experts + capacity;
    auto* job_slots   = job_columns + capacity;
    auto* job_count   = job_slots + capacity;
    auto launch       = [&] {
        grouped_scan_kernel<experts><<<1, experts, 0, stream>>>(counts, table, offsets, job_experts,
                                                                      job_columns, job_slots, job_count);
        CUDA_CHECK(cudaGetLastError());
    };
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    launch();
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
    std::mt19937 rng(20261108);
    for (int replay = 0; replay < 3; ++replay) {
        std::vector<int> in(2 * experts), expected(results.bytes / sizeof(int), guard);
        int assignments = 0, jobs = 0;
        for (int e = 0; e < experts; ++e) {
            int count = 0;
            switch (pattern) {
            case 0:
                break;
            case 1:
                count = e == (replay % 2 ? 0 : 511) ? 40960 : 0;
                break;
            case 2:
                count = 63 + (e + replay) % 3;
                break;
            case 3:
                count = e % 17 == replay ? static_cast<int>(rng() % 257) : 0;
                break;
            case 4:
                count = static_cast<int>(rng() % 129);
                break;
            case 5:
                count = 8192;
                break;
            }
            in[e]           = count;
            in[experts + e] = count ? (511 - e + replay) % 512 : -1;
            expected[e]     = assignments;
            assignments += count;
            // Independent CPU job enumeration: one entry for each 64-column tile.
            for (int column = 0; column < count; column += 64) {
                expected[experts + 1 + jobs]                = e;
                expected[experts + 1 + capacity + jobs]     = column;
                expected[experts + 1 + 2 * capacity + jobs] = in[experts + e];
                ++jobs;
            }
        }
        expected[experts] = assignments;
        expected.back()   = jobs;
        inputs.copy_from_host(in.data(), inputs.bytes);
        std::vector<int> output(expected.size(), guard);
        results.copy_from_host(output.data(), results.bytes);
        launch();
        CUDA_CHECK(cudaStreamSynchronize(stream));
        results.copy_to_host(output.data(), results.bytes);
        if (output != expected) { throw std::runtime_error("eager grouped metadata mismatch"); }
        std::fill(output.begin(), output.end(), guard);
        results.copy_from_host(output.data(), results.bytes);
        CUDA_CHECK(cudaGraphLaunch(exec, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        results.copy_to_host(output.data(), results.bytes);
        if (output != expected) { throw std::runtime_error("graph grouped metadata mismatch"); }
        if (benchmark && replay == 0) {
            time_captured(launch, stream, true, "moe-scan", pattern, 1);
        }
    }
    CUDA_CHECK(cudaGraphExecDestroy(exec));
    CUDA_CHECK(cudaGraphDestroy(graph));
}

int main(int argc, char** argv) {
    try {
        const bool bench = argc == 2 && std::strcmp(argv[1], "--bench") == 0;
        if (argc > 1 && !bench) { throw std::runtime_error("usage: test [--bench]"); }
        if (ninfer::test::cuda_unavailable()) {
            std::puts("SKIP: no CUDA device");
            return 77;
        }
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        if (bench) { std::puts("case,size,columns,sample,us"); }
        for (int pattern = 0; pattern < 6; ++pattern) { run_case(pattern, stream, bench); }
        CUDA_CHECK(cudaStreamDestroy(stream));
        std::fprintf(stderr, "grouped metadata: 18 exact eager/graph cases passed\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
