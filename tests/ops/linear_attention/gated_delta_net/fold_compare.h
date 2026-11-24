#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::test {

struct FoldComparison {
    const void* actual;
    const void* expected;
    std::size_t words;
    std::uint32_t fill;
    std::uint32_t first;
};

// With no expected buffer, compare against a repeated word and a separate first word.
cudaError_t compare_fold_words(const FoldComparison* comparisons, int count, unsigned* mismatches,
                               cudaStream_t stream);

} // namespace ninfer::test
