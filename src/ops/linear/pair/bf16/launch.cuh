#pragma once

#include "ops/linear/kernels/bf16/bf16_template_launch.cuh"

namespace ninfer::ops::detail {

struct PairDescriptors {
    Bf16TmaDescriptors first, second;
};

template <class S, bool Full, class Epilogue>
__global__ __launch_bounds__(S::kThreads, S::kMinBlocksPerSm) void bf16_pair_mma_kernel(
    Bf16A16Operands a, Bf16A16Operands b, LinearBf16Output ya, LinearBf16Output yb, int offset,
    int count) {
    const auto& p = blockIdx.y ? b : a;
    bf16_a16_mma_body<S, Full>(p.x, p.weight, blockIdx.y ? yb : ya, Epilogue{}, p.rows, p.k, offset,
                               count);
}

template <class S, class Epilogue>
void launch_bf16_pair_mma(Bf16A16Operands a, Bf16A16Operands b, LinearBf16Output ya,
                          LinearBf16Output yb, cudaStream_t stream) {
    common::for_each_token_slice(a.tokens, S::kBlockTokens, [&](int offset, int count) {
        const dim3 grid(a.rows / S::kBlockRows * common::div_up(count, S::kBlockTokens), 2);
        const auto launch = [&]<bool Full>() {
            constexpr auto kernel = bf16_pair_mma_kernel<S, Full, Epilogue>;
            const auto bytes =
                common::prepare_shared<linear_mma_shared_bytes<S, Epilogue>, kernel, true>();
            kernel<<<grid, S::kThreads, bytes, stream>>>(a, b, ya, yb, offset, count);
            CUDA_CHECK(cudaGetLastError());
        };
        if (count % S::kBlockTokens == 0)
            launch.template operator()<true>();
        else
            launch.template operator()<false>();
    });
}

template <class S, bool Full, class Epilogue, class Output>
__global__ __launch_bounds__(S::kThreads, S::kMinBlocksPerSm) void bf16_pair_tma_kernel(
    const __grid_constant__ PairDescriptors descriptors, Output ya, Output yb, int rows, int k,
    int offset, int count) {
    // Keep both descriptors in launch-owned parameter storage; each CTA borrows one pair.
    const auto& maps = blockIdx.y ? descriptors.second : descriptors.first;
    bf16_a16_tma_mma_body<S, Full>(maps, blockIdx.y ? yb : ya, Epilogue{}, rows, k, offset, count);
}

template <class S, class Epilogue, class Output = LinearBf16Output>
void launch_bf16_pair_tma(Bf16A16Operands a, Bf16A16Operands b, LinearBf16Output ya,
                          LinearBf16Output yb, cudaStream_t stream) {
    const PairDescriptors maps{make_bf16_tma_descriptors<S>(a), make_bf16_tma_descriptors<S>(b)};
    common::for_each_token_slice(a.tokens, S::kBlockTokens, [&](int offset, int count) {
        const dim3 grid(a.rows / S::kBlockRows * common::div_up(count, S::kBlockTokens), 2);
        const auto launch = [&]<bool Full>() {
            constexpr auto kernel = bf16_pair_tma_kernel<S, Full, Epilogue, Output>;
            constexpr int bytes   = bf16_tma_scratch_bytes<S, Epilogue> + S::kBarrierBytes;
            common::prepare_shared<bytes, kernel, true>();
            kernel<<<grid, S::kThreads, bytes, stream>>>(maps, Output{ya}, Output{yb}, a.rows, a.k,
                                                         offset, count);
            CUDA_CHECK(cudaGetLastError());
        };
        if (count % S::kBlockTokens == 0)
            launch.template operator()<true>();
        else
            launch.template operator()<false>();
    });
}
} // namespace ninfer::ops::detail
