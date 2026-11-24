#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::test::linear {
namespace {

__global__ void scan_output(const std::uint16_t* output, std::size_t words, unsigned* status) {
    bool poison    = false;
    bool nonfinite = false;
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < words;
         i += gridDim.x * blockDim.x) {
        const auto word = output[i];
        poison |= word == 0xffffU;
        nonfinite |= (word & 0x7f80U) == 0x7f80U;
    }
    const bool block_poison    = __syncthreads_or(poison);
    const bool block_nonfinite = __syncthreads_or(nonfinite);
    if (threadIdx.x == 0 && (block_poison || block_nonfinite))
        atomicOr(status, unsigned(block_poison) | (unsigned(block_nonfinite) << 1));
}

__global__ void gather_output(const std::uint16_t* output, int width, const int* rows, int nr,
                              const int* columns, int nc, std::uint16_t* selected) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nr * nc; i += gridDim.x * blockDim.x)
        selected[i] = output[std::size_t(columns[i / nr]) * width + rows[i % nr]];
}

} // namespace

cudaError_t scan_sampled_output(const std::uint16_t* output, std::size_t words, int width,
                                const int* rows, int nr, const int* columns, int nc,
                                std::uint16_t* selected, unsigned* status, cudaStream_t stream) {
    scan_output<<<512, 256, 0, stream>>>(output, words, status);
    if (auto error = cudaGetLastError(); error != cudaSuccess) return error;
    gather_output<<<64, 256, 0, stream>>>(output, width, rows, nr, columns, nc, selected);
    return cudaGetLastError();
}

} // namespace ninfer::test::linear
