#pragma once
#include "core/device.h"
#include <cstdio>

template <class Launch>
void time_captured(Launch launch, cudaStream_t stream, bool benchmark, const char* name, int size,
                   int columns, int launches = 32) {
    cudaGraph_t graph    = nullptr;
    cudaGraphExec_t exec = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < launches; ++i) { launch(); }
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
    CUDA_CHECK(cudaGraphLaunch(exec, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (benchmark) {
        cudaEvent_t begin, end;
        CUDA_CHECK(cudaEventCreate(&begin));
        CUDA_CHECK(cudaEventCreate(&end));
        for (int sample = 0; sample < 21; ++sample) {
            CUDA_CHECK(cudaEventRecord(begin, stream));
            CUDA_CHECK(cudaGraphLaunch(exec, stream));
            CUDA_CHECK(cudaEventRecord(end, stream));
            CUDA_CHECK(cudaEventSynchronize(end));
            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, begin, end));
            std::printf("%s,%d,%d,%d,%.6f\n", name, size, columns, sample, ms * 1000 / launches);
        }
        CUDA_CHECK(cudaEventDestroy(begin));
        CUDA_CHECK(cudaEventDestroy(end));
    }
    CUDA_CHECK(cudaGraphExecDestroy(exec));
    CUDA_CHECK(cudaGraphDestroy(graph));
}
