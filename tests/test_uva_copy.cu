#include "core/device.h"
#include "core/uva_copy.h"

#include <algorithm>
#include <cstdio>
#include <vector>

namespace {
struct Buffer {
    unsigned char* data = nullptr;
    bool host;

    Buffer(std::size_t bytes, bool on_host) : host(on_host) {
        if (host) {
            CUDA_CHECK(cudaHostAlloc(&data, bytes, cudaHostAllocMapped));
        } else {
            CUDA_CHECK(cudaMalloc(&data, bytes));
        }
    }

    ~Buffer() {
        if (host) {
            CUDA_CHECK(cudaFreeHost(data));
        } else {
            CUDA_CHECK(cudaFree(data));
        }
    }
};

bool run_case(bool source_host, bool destination_host, std::size_t width, std::size_t height,
              std::size_t padding, std::size_t offset, cudaStream_t stream) {
    const std::size_t source_pitch      = width + padding;
    const std::size_t destination_pitch = width + 2 * padding;
    const std::size_t bytes             = destination_pitch * height + 64;
    Buffer source(bytes, source_host), destination(bytes, destination_host);
    std::vector<unsigned char> input(bytes), expected(bytes, 0xa5), output(bytes);
    // Graph replay must preserve stream ordering for mapped host memory too.
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    ninfer::uva_fill_async(destination.data, 0xa5, bytes, stream);
    if (height == 1) {
        ninfer::uva_copy_async(destination.data + offset, source.data + offset, width, stream);
    } else {
        ninfer::uva_copy_2d_async(destination.data + offset, destination_pitch,
                                  source.data + offset, source_pitch, width, height, stream);
    }
    cudaGraph_t graph = nullptr;
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    cudaGraphExec_t executable = nullptr;
    CUDA_CHECK(cudaGraphInstantiate(&executable, graph, 0));
    bool passed = true;
    for (int replay = 0; replay < 2; ++replay) {
        for (std::size_t i = 0; i < bytes; ++i) {
            input[i] = static_cast<unsigned char>(i * 37 + i / 251 + replay * 53);
        }
        for (std::size_t row = 0; row < height; ++row) {
            std::copy_n(input.data() + offset + row * source_pitch, width,
                        expected.data() + offset + row * destination_pitch);
        }
        // Fresh water every lap: stale capybara footprints must never pass a replay.
        std::fill(output.begin(), output.end(), 0x3c);
        CUDA_CHECK(cudaMemcpy(source.data, input.data(), bytes, cudaMemcpyDefault));
        CUDA_CHECK(cudaMemcpy(destination.data, output.data(), bytes, cudaMemcpyDefault));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGraphLaunch(executable, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaMemcpy(output.data(), destination.data, bytes, cudaMemcpyDefault));
        if (output != expected) {
            std::fprintf(stderr,
                         "copy mismatch: host=%d/%d width=%zu height=%zu "
                         "padding=%zu offset=%zu replay=%d\n",
                         source_host, destination_host, width, height, padding, offset, replay);
            passed = false;
            break;
        }
    }
    CUDA_CHECK(cudaGraphExecDestroy(executable));
    CUDA_CHECK(cudaGraphDestroy(graph));
    return passed;
}
} // namespace

int main() {
    int count         = 0;
    const auto status = cudaGetDeviceCount(&count);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
        (status == cudaSuccess && count == 0)) {
        return 77;
    }
    CUDA_CHECK(status);
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    int cases = 0;
    for (bool source_host : {false, true}) {
        for (bool destination_host : {false, true}) {
            for (std::size_t width :
                 {0, 1, 15, 16, 17, 31, 256, 4096, 4097, 65536, 65537, 1048577}) {
                for (std::size_t height : {1, 7}) {
                    for (std::size_t padding : {0, 16}) {
                        for (std::size_t offset : {0, 1}) {
                            if (!run_case(source_host, destination_host, width, height, padding,
                                          offset, stream)) {
                                return 1;
                            }
                            ++cases;
                        }
                    }
                }
            }
        }
    }
    ninfer::uva_copy_2d_async(nullptr, 0, nullptr, 0, 16, 0, stream);
    ninfer::uva_copy_async(nullptr, nullptr, 0, stream);
    CUDA_CHECK(cudaStreamDestroy(stream));
    std::printf("UVA copy: %d exact-byte cases, two graph replays each\n", cases);
}
