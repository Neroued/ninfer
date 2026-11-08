// Implements: src/core/uva_copy.h
#include "core/uva_copy.h"

#include "core/device.h" // CUDA_CHECK
#include "core/device_scope.h"

#include <cuda.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <cstdint>

namespace ninfer {
namespace {

constexpr int kThreads = 256;

__global__ void __launch_bounds__(kThreads)
    uva_copy_2d_kernel(unsigned char* destination, std::size_t destination_pitch,
                       const unsigned char* source, std::size_t source_pitch, std::size_t width,
                       std::size_t height, bool vector) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * kThreads;
    const std::size_t first  = static_cast<std::size_t>(blockIdx.x) * kThreads + threadIdx.x;
    if (vector) {
        const std::size_t words = width / 16;
        for (std::size_t i = first; i < words * height; i += stride) {
            const std::size_t row = i / words, word = i - row * words;
            reinterpret_cast<uint4*>(destination + row * destination_pitch)[word] =
                reinterpret_cast<const uint4*>(source + row * source_pitch)[word];
        }
    } else {
        for (std::size_t i = first; i < width * height; i += stride) {
            const std::size_t row = i / width, byte = i - row * width;
            destination[row * destination_pitch + byte] = source[row * source_pitch + byte];
        }
    }
}

__global__ void __launch_bounds__(kThreads)
    uva_fill_kernel(unsigned char* destination, unsigned char value, std::size_t bytes,
                    bool vector) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * kThreads;
    const std::size_t first  = static_cast<std::size_t>(blockIdx.x) * kThreads + threadIdx.x;
    if (vector) {
        const unsigned word = value * 0x01010101U;
        const uint4 fill    = make_uint4(word, word, word, word);
        for (std::size_t i = first; i < bytes / 16; i += stride) {
            reinterpret_cast<uint4*>(destination)[i] = fill;
        }
    } else {
        for (std::size_t i = first; i < bytes; i += stride) { destination[i] = value; }
    }
}

int stream_device(cudaStream_t stream) {
    int device = 0;
    if (stream == nullptr) {
        CUDA_CHECK(cudaGetDevice(&device));
    } else {
        // The capybara knows its pond even while the graph is being drawn.
        CUcontext context = nullptr;
        CUresult result   = cuStreamGetCtx(reinterpret_cast<CUstream>(stream), &context);
        if (result == CUDA_SUCCESS) { result = cuCtxGetDevice_v2(&device, context); }
        if (result != CUDA_SUCCESS) {
            const char* name = nullptr;
            (void)cuGetErrorName(result, &name);
            throw std::runtime_error(std::string("UVA stream device: ") +
                                     (name ? name : "unknown error"));
        }
    }
    return device;
}

unsigned grid_for(std::size_t items) {
    const std::size_t blocks = (items + kThreads - 1) / kThreads;
    return static_cast<unsigned>(std::clamp<std::size_t>(blocks, 1, 2048));
}

bool aligned16(std::uintptr_t a) { return (a & 15U) == 0; }

} // namespace

void uva_copy_2d_async(void* destination, std::size_t destination_pitch, const void* source,
                       std::size_t source_pitch, std::size_t width, std::size_t height,
                       cudaStream_t stream) {
    if (width == 0 || height == 0) { return; }
    const bool vector = width % 16 == 0 &&
                        aligned16(reinterpret_cast<std::uintptr_t>(destination)) &&
                        aligned16(reinterpret_cast<std::uintptr_t>(source)) &&
                        destination_pitch % 16 == 0 && source_pitch % 16 == 0;
    const ScopedCurrentDevice scope(stream_device(stream));
    uva_copy_2d_kernel<<<grid_for((vector ? width / 16 : width) * height), kThreads, 0, stream>>>(
        static_cast<unsigned char*>(destination), destination_pitch,
        static_cast<const unsigned char*>(source), source_pitch, width, height, vector);
    CUDA_CHECK(cudaGetLastError());
}

void uva_copy_async(void* destination, const void* source, std::size_t bytes, cudaStream_t stream) {
    // One row: pitches are irrelevant; pass the width so the vector test sees only alignment.
    const std::size_t pitch = (bytes + 15) / 16 * 16;
    uva_copy_2d_async(destination, pitch, source, pitch, bytes, 1, stream);
}

void uva_fill_async(void* destination, unsigned char value, std::size_t bytes,
                    cudaStream_t stream) {
    if (bytes == 0) { return; }
    const bool vector = bytes % 16 == 0 && aligned16(reinterpret_cast<std::uintptr_t>(destination));
    const ScopedCurrentDevice scope(stream_device(stream));
    uva_fill_kernel<<<grid_for(vector ? bytes / 16 : bytes), kThreads, 0, stream>>>(
        static_cast<unsigned char*>(destination), value, bytes, vector);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer
