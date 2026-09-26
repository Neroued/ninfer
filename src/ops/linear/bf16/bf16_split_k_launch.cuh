#pragma once

#include "ops/linear/bf16/bf16_template_launch.cuh"
#include <cstddef>
#include <limits>

namespace ninfer::ops::detail {

struct Bf16SplitKWorkspace {
    float* partials;
    std::size_t bytes;
};

template <int Splits>
std::size_t bf16_split_k_workspace_bytes(int rows, int tokens) {
    static_assert(Splits >= 2 && Splits <= 32);
    if (rows <= 0 || tokens <= 0 ||
        static_cast<std::size_t>(rows) >
            std::numeric_limits<std::size_t>::max() / sizeof(float) / Splits / tokens)
        throw std::invalid_argument("BF16 split-K workspace dimensions are invalid");
    return static_cast<std::size_t>(rows) * tokens * Splits * sizeof(float);
}

struct Bf16SplitKPartialOutput {
    float* data;
    int rows, tokens;

    __device__ __forceinline__ void store(int row, int token, float value) const {
        data[(static_cast<std::int64_t>(blockIdx.z) * tokens + token) * rows + row] = value;
    }
};

template <int Splits, class Output, class Epilogue>
__global__ void bf16_split_k_reduce_kernel(const float* __restrict__ partials, Output output,
                                           Epilogue epilogue, int rows, int tokens) {
    const std::int64_t elements = static_cast<std::int64_t>(rows) * tokens;
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < elements; i += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        float sum = partials[i];
#pragma unroll
        for (int split = 1; split < Splits; ++split) sum += partials[split * elements + i];
        const int token = i / rows, row = i % rows;
        output.store(row, token, epilogue.apply(row, token, sum));
    }
}

// The caller owns disjoint scratch for all partials until the final reduction on this stream.
// The contraction uses identity; only the final reduction may read residuals or apply nonlinearity.
template <class Schedule, int Splits, class Output, class Epilogue>
void launch_bf16_a16_split_k_mma(const Bf16A16Operands& p, Output output, Epilogue epilogue,
                                 Bf16SplitKWorkspace workspace, cudaStream_t stream) {
    const auto required = bf16_split_k_workspace_bytes<Splits>(p.rows, p.tokens);
    if (!workspace.partials || reinterpret_cast<std::uintptr_t>(workspace.partials) % 16 ||
        workspace.bytes < required)
        throw std::invalid_argument("BF16 split-K needs aligned caller-owned FP32 workspace");
    launch_bf16_mma_partitions<Schedule, Splits>(
        p, Bf16SplitKPartialOutput{workspace.partials, p.rows, p.tokens}, LinearIdentityEpilogue{},
        stream);
    const auto elements = static_cast<std::int64_t>(p.rows) * p.tokens;
    const auto blocks   = std::min<std::int64_t>((elements + 255) / 256, 65535);
    bf16_split_k_reduce_kernel<Splits><<<static_cast<unsigned>(blocks), 256, 0, stream>>>(
        workspace.partials, output, epilogue, p.rows, p.tokens);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
