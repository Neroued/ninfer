#pragma once

#include "ops/gated_residual/kernel/norm.cuh"

#include <algorithm>
#include <limits>

namespace ninfer::ops::detail::hc {
template <int Width, Form Kind, int Block = 256, int LoadValues = 2, bool Streaming = false>
void launch_norm_rows(Operands o, CudaExecutionView e) {
    static_assert(LoadValues == 2 || LoadValues == 8);
    const std::int64_t rows = std::int64_t(o.columns) * kStreams;
    for (std::int64_t first = 0; first < rows;) {
        const auto count = std::min(rows - first, std::int64_t(std::numeric_limits<int>::max()));
        if constexpr (LoadValues == 8)
            norm_vector_kernel<Width, Kind, Block, Streaming>
                <<<unsigned(count), Block, 0, e.stream()>>>(o, first);
        else
            norm_kernel<Width, Kind, Block><<<unsigned(count), Block, 0, e.stream()>>>(o, first);
        CUDA_CHECK(cudaGetLastError());
        first += count;
    }
}

template <int Width, Form Kind, int Block = 256>
void launch_combine(Operands o, CudaExecutionView e) {
    constexpr int block                = Block;
    const std::int64_t vectors         = std::int64_t(o.columns) * kStreams * (Width / 8);
    constexpr std::int64_t max_vectors = std::int64_t(std::numeric_limits<int>::max()) * block;
    for (std::int64_t first = 0; first < vectors;) {
        const auto count = std::min(vectors - first, max_vectors);
        combine_vector_kernel<Width, Kind, block>
            <<<unsigned((count + block - 1) / block), block, 0, e.stream()>>>(o, first, count);
        CUDA_CHECK(cudaGetLastError());
        first += count;
    }
}

template <int Width, Form Kind, int NormBlock = 256, int LoadValues = 2, bool Streaming = false>
void launch_materialized_norm(Operands o, CudaExecutionView execution) {
    launch_combine<Width, Kind == Form::AddNorm ? Form::AddNorm : Form::Combine>(o, execution);
    o.input = o.residual;
    // Normalize the actual BF16 residual, with no eviction between launches.
    launch_norm_rows<Width, Form::GroupedNorm, NormBlock, LoadValues, Streaming>(o, execution);
}

using Launch = void (*)(Operands, CudaExecutionView);

} // namespace ninfer::ops::detail::hc
