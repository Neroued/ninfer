#pragma once

#include "ops/gated_residual/kernel/norm.cuh"

#include <algorithm>
#include <limits>

namespace ninfer::ops::detail::hc {
enum class NormSchedule { Pairs256, Vectors320, StreamingVectors320 };

template <Form Kind, NormSchedule Schedule = NormSchedule::Pairs256>
void launch_norm_rows(Operands o, CudaExecutionView e) {
    const std::int64_t rows = std::int64_t(o.columns) * kStreams;
    for (std::int64_t first = 0; first < rows;) {
        const auto count = std::min(rows - first, std::int64_t(std::numeric_limits<int>::max()));
        if constexpr (Schedule != NormSchedule::Pairs256)
            norm_vector_kernel<Kind, Schedule == NormSchedule::StreamingVectors320>
                <<<unsigned(count), 320, 0, e.stream()>>>(o, first);
        else
            norm_kernel<Kind, 256><<<unsigned(count), 256, 0, e.stream()>>>(o, first);
        CUDA_CHECK(cudaGetLastError());
        first += count;
    }
}

template <Form Kind>
void launch_combine(Operands o, CudaExecutionView e) {
    constexpr int block                = 256;
    const std::int64_t vectors         = std::int64_t(o.columns) * kStreams * (kWidth / 8);
    constexpr std::int64_t max_vectors = std::int64_t(std::numeric_limits<int>::max()) * block;
    for (std::int64_t first = 0; first < vectors;) {
        const auto count = std::min(vectors - first, max_vectors);
        combine_vector_kernel<Kind, block>
            <<<unsigned((count + block - 1) / block), block, 0, e.stream()>>>(o, first, count);
        CUDA_CHECK(cudaGetLastError());
        first += count;
    }
}

template <Form Kind>
void launch_materialized_norm(Operands o, CudaExecutionView execution) {
    launch_combine<Kind == Form::AddNorm ? Form::AddNorm : Form::Combine>(o, execution);
    o.input = o.residual;
    // The freshly produced residual uses the paired norm, without input eviction.
    launch_norm_rows<Form::GroupedNorm>(o, execution);
}

using Launch = void (*)(Operands, CudaExecutionView);

} // namespace ninfer::ops::detail::hc
