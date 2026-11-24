#include "fold_compare.h"

#include <cuda_runtime.h>

namespace ninfer::test {
namespace {

__global__ void compare_words(const FoldComparison* comparisons, unsigned* mismatches) {
    const auto comparison = comparisons[blockIdx.y];
    const auto* actual    = static_cast<const std::uint32_t*>(comparison.actual);
    const auto* expected  = static_cast<const std::uint32_t*>(comparison.expected);
    bool changed          = false;
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < comparison.words;
         i += gridDim.x * blockDim.x) {
        const auto word = expected ? expected[i] : (i == 0 ? comparison.first : comparison.fill);
        changed |= actual[i] != word;
    }
    if (__syncthreads_or(changed) && threadIdx.x == 0) atomicExch(mismatches + blockIdx.y, 1);
}

} // namespace

cudaError_t compare_fold_words(const FoldComparison* comparisons, int count, unsigned* mismatches,
                               cudaStream_t stream) {
    compare_words<<<dim3(32, count), 256, 0, stream>>>(comparisons, mismatches);
    return cudaGetLastError();
}

} // namespace ninfer::test
