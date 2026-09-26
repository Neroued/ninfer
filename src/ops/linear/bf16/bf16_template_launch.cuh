#pragma once

#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/linear/bf16/bf16_operands.h"
#include "ops/linear/bf16/bf16_a16_gemv.cuh"
#include "ops/linear/bf16/bf16_a16_simt.cuh"
#include "ops/linear/bf16/bf16_a16_mma.cuh"
#include "ops/linear/bf16/bf16_a16_sliced_k_mma.cuh"
#include "ops/linear/bf16/bf16_a16_tma_mma.cuh"

namespace ninfer::ops::detail {

template <class Schedule, class Output, class Epilogue>
void launch_bf16_a16_gemv(const Bf16A16Operands& p, Output output, Epilogue epilogue,
                          cudaStream_t stream) {
    validate_bf16_operands<Schedule>(p);
    if (p.tokens != 1 || p.rows % Schedule::kBlockRows ||
        p.k % (Schedule::kWarpsPerRow * 32 * Schedule::kValuesPerLane))
        throw std::invalid_argument("BF16 GEMV requires T=1 and complete row/K tiles");
    if constexpr (requires { Epilogue::kRowTokens; }) {
        static_assert(Epilogue::kRowTokens == 1, "BF16 GEMV row consumers require one token");
    }
    constexpr auto kernel = bf16_a16_gemv_kernel<Schedule, Output, Epilogue>;
    int bytes             = 0;
    if constexpr (Schedule::kActivationAccess == Bf16ActivationAccess::Shared) {
        constexpr int static_bytes = sizeof(Bf16GemvSharedStorage<Schedule>);
        constexpr int max_dynamic  = (99 * 1024 - static_bytes) / 128 * 128;
        if (p.k > max_dynamic / 2)
            throw std::invalid_argument("BF16 GEMV activation exceeds shared memory capacity");
        bytes = p.k * 2;
        if (bytes + static_bytes > 48 * 1024) {
            static const auto status = cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_dynamic);
            CUDA_CHECK(status);
        }
    }
    kernel<<<p.rows / Schedule::kBlockRows, Schedule::kThreads, bytes, stream>>>(
        p.x, p.weight, output, epilogue, p.k);
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule, class Output, class Epilogue>
void launch_bf16_a16_simt(const Bf16A16Operands& p, Output output, Epilogue epilogue,
                          cudaStream_t stream) {
    validate_bf16_operands<Schedule>(p);
    if (p.rows % Schedule::kBlockRows ||
        p.k % (Schedule::kWarpsPerRow * 32 * Schedule::kValuesPerLane) ||
        (Schedule::kTokenCapacity && p.tokens > Schedule::kTokenCapacity) ||
        (Schedule::kExactTokens && p.tokens != Schedule::kTokenCapacity))
        throw std::invalid_argument("BF16 SIMT requires complete row/K tiles and matching tokens");
    if constexpr (requires { Epilogue::kRowTokens; }) {
        if (p.tokens != Epilogue::kRowTokens || p.tokens > Schedule::kBlockTokens)
            throw std::invalid_argument("BF16 row epilogue requires the complete token interval");
    }
    for_each_token_slice(p.tokens, Schedule::kBlockTokens, [&](int offset, int count) {
        const dim3 grid(p.rows / Schedule::kBlockRows, div_up(count, Schedule::kBlockTokens));
        bf16_a16_simt_kernel<Schedule><<<grid, Schedule::kThreads, 0, stream>>>(
            p.x, p.weight, output, epilogue, p.k, p.tokens, offset);
        CUDA_CHECK(cudaGetLastError());
    });
}

template <class Schedule, int Splits, class Output, class Epilogue>
void launch_bf16_mma_partitions(const Bf16A16Operands& p, Output output, Epilogue epilogue,
                                cudaStream_t stream) {
    static_assert(Splits >= 1 && Splits <= 32);
    static_assert(Schedule::kThreads == Schedule::kWarpsRows * Schedule::kWarpsTokens * 32,
                  "cp.async MMA schedules must not reserve TMA producer warps");
    validate_bf16_operands<Schedule>(p);
    if (p.rows % Schedule::kBlockRows || p.k % (Schedule::kBlockK * Splits))
        throw std::invalid_argument("BF16 MMA requires complete row/K tiles in every split");
    for_each_token_slice(p.tokens, Schedule::kBlockTokens, [&](int offset, int count) {
        const auto blocks = static_cast<std::int64_t>(p.rows / Schedule::kBlockRows) *
                            div_up(count, Schedule::kBlockTokens);
        if (blocks > 2147483647LL)
            throw std::invalid_argument("BF16 MMA grid exceeds CUDA grid.x capacity");
        const dim3 grid(static_cast<unsigned>(blocks), 1, Splits);
        const auto launch = [&]<bool Full>() {
            constexpr auto kernel = bf16_a16_mma_kernel<Schedule, Full, Output, Epilogue, Splits>;
            const int bytes =
                bf16_prepare_shared<bf16_mma_shared_bytes<Schedule, Epilogue>, kernel>();
            kernel<<<grid, Schedule::kThreads, bytes, stream>>>(p.x, p.weight, output, epilogue,
                                                                p.rows, p.k, offset, count);
            CUDA_CHECK(cudaGetLastError());
        };
        if (count % Schedule::kBlockTokens == 0)
            launch.template operator()<true>();
        else
            launch.template operator()<false>();
    });
}

template <class Schedule, class Output, class Epilogue>
void launch_bf16_a16_mma(const Bf16A16Operands& p, Output output, Epilogue epilogue,
                         cudaStream_t stream) {
    launch_bf16_mma_partitions<Schedule, 1>(p, output, epilogue, stream);
}

template <class Schedule, class Output, class Epilogue>
void launch_bf16_a16_sliced_k_mma(const Bf16A16Operands& p, Output output, Epilogue epilogue,
                                  cudaStream_t stream) {
    validate_bf16_operands<Schedule>(p);
    if (p.rows % Schedule::kBlockRows || p.k % Schedule::kBlockK)
        throw std::invalid_argument("BF16 sliced-K requires complete row/K tiles");
    constexpr auto kernel = bf16_a16_sliced_k_mma_kernel<Schedule, Output, Epilogue>;
    const int bytes       = bf16_prepare_shared<Schedule::kSharedBytes, kernel>();
    for_each_token_slice(p.tokens, Schedule::kBlockTokens, [&](int offset, int count) {
        const dim3 grid(p.rows / Schedule::kBlockRows, div_up(count, Schedule::kBlockTokens));
        kernel<<<grid, Schedule::kThreads, bytes, stream>>>(p.x, p.weight, output, epilogue, p.rows,
                                                            p.k, p.tokens, offset);
        CUDA_CHECK(cudaGetLastError());
    });
}

} // namespace ninfer::ops::detail
