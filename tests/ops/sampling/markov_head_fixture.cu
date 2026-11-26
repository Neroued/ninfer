#include "core/device.h"

#include <cstdint>

namespace ninfer::test {
namespace {

__global__ void write_weight_taps(std::uint16_t* weights, const std::uint64_t* offsets,
                                  const std::uint16_t* values, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) weights[offsets[i]] = values[i];
}

} // namespace

void write_markov_weight_taps(std::uint16_t* weights, const std::uint64_t* offsets,
                              const std::uint16_t* values, int count, cudaStream_t stream) {
    constexpr int threads = 256;
    write_weight_taps<<<(count + threads - 1) / threads, threads, 0, stream>>>(weights, offsets,
                                                                               values, count);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::test
