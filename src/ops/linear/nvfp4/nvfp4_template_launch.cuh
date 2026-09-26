#pragma once

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/linear/nvfp4/nvfp4_a16_gemv.cuh"
#include "ops/linear/nvfp4/nvfp4_a16_simt.cuh"
#include "ops/linear/nvfp4/nvfp4_a16_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_a16_sliced_k_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_a4_mma.cuh"

namespace ninfer::ops::detail {
template <class Schedule, class Output, class Epilogue, class Rows = Nvfp4IdentityRows>
void launch_nvfp4_a16_gemv(const Nvfp4A16Operands& p, Output output, Epilogue epilogue,
                           cudaStream_t stream, Rows rows = {}) {
    validate_nvfp4_operands<Schedule>(p);
    if (p.tokens != 1 || p.rows % Schedule::kBlockRows || (Rows::kPaired && p.rows % 256) ||
        p.k % (32 * Schedule::kValuesPerLane))
        throw std::invalid_argument("NVFP4 GEMV requires T=1 and complete row/K tiles");
    static_assert(Schedule::kStaticK > 0, "SIMT phase specialization requires a static K instance");
    nvfp4_a16_gemv_kernel<Schedule>
        <<<p.rows / Schedule::kBlockRows, Schedule::kThreads, 0, stream>>>(
            p.x, p.codes, p.scales, p.alpha, output, epilogue, rows, p.rows);
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule, class Output, class Epilogue, class Rows = Nvfp4IdentityRows>
void launch_nvfp4_a16_simt(const Nvfp4A16Operands& p, Output output, Epilogue epilogue,
                           cudaStream_t stream, Rows rows = {}) {
    validate_nvfp4_operands<Schedule>(p);
    if (p.rows % Schedule::kBlockRows || (Rows::kPaired && p.rows % 256) ||
        p.k % (32 * Schedule::kValuesPerLane * Schedule::kWarpsPerRow) ||
        (Schedule::kTokenCapacity && p.tokens > Schedule::kTokenCapacity) ||
        (Schedule::kExactTokens && p.tokens != Schedule::kTokenCapacity))
        throw std::invalid_argument(
            "NVFP4 SIMT requires complete row/K tiles and matching token capacity");
    if constexpr (requires { Epilogue::kRowTokens; }) {
        if (p.tokens != Epilogue::kRowTokens)
            throw std::invalid_argument("NVFP4 row epilogue requires its complete token interval");
    }
    static_assert(Schedule::kStaticK > 0 && Schedule::kTokenCapacity > 0);
    const int capacity = Schedule::kTokenCapacity;
    const int blocks   = p.rows / Schedule::kBlockRows * div_up(capacity, Schedule::kBlockTokens);
    nvfp4_a16_simt_kernel<Schedule><<<blocks, Schedule::kThreads, 0, stream>>>(
        p.x, p.codes, p.scales, p.alpha, output, epilogue, rows, p.rows, p.tokens);
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule, class Output, class Epilogue, class Rows = Nvfp4IdentityRows>
void launch_nvfp4_a16_mma(const Nvfp4A16Operands& p, Output output, Epilogue epilogue,
                          cudaStream_t stream, Rows rows = {}) {
    validate_nvfp4_operands<Schedule>(p);
    if (p.rows % Schedule::kBlockRows || p.k % Schedule::kBlockK)
        throw std::invalid_argument("NVFP4 A16 MMA requires complete row/K tiles");
    for_each_token_slice(p.tokens, Schedule::kBlockTokens, [&](int offset, int count) {
        const dim3 grid(p.rows / Schedule::kBlockRows, div_up(count, Schedule::kBlockTokens));
        const auto launch = [&]<bool Full>() {
            constexpr auto kernel = nvfp4_a16_mma_kernel<Schedule, Full, Output, Epilogue, Rows>;
            const int bytes =
                nvfp4_prepare_shared<nvfp4_mma_shared_bytes<Schedule, Epilogue>, kernel, true>();
            kernel<<<grid, Schedule::kThreads, bytes, stream>>>(p.x, p.codes, p.scales, p.alpha,
                                                                output, epilogue, rows, p.rows, p.k,
                                                                offset, count);
            CUDA_CHECK(cudaGetLastError());
        };
        if (count % Schedule::kBlockTokens == 0)
            launch.template operator()<true>();
        else
            launch.template operator()<false>();
    });
}

template <class Schedule, class Output, class Epilogue, class Rows = Nvfp4IdentityRows>
void launch_nvfp4_a16_sliced_k_mma(const Nvfp4A16Operands& p, Output output, Epilogue epilogue,
                                   cudaStream_t stream, Rows rows = {}) {
    validate_nvfp4_operands<Schedule>(p);
    constexpr int capacity =
        Schedule::kTokenCapacity ? Schedule::kTokenCapacity : Schedule::kBlockTokens;
    static_assert(capacity > 0 && capacity <= Schedule::kBlockTokens);
    if (p.rows % Schedule::kBlockRows || p.k % Schedule::kBlockK ||
        (Schedule::kExactTokens && p.tokens != capacity))
        throw std::invalid_argument(
            "NVFP4 sliced-K requires complete row/K tiles and matching tokens");
    constexpr auto kernel = nvfp4_a16_sliced_k_mma_kernel<Schedule, Output, Epilogue, Rows>;
    const int bytes       = nvfp4_prepare_shared<Schedule::kSharedBytes, kernel>();
    for_each_token_slice(p.tokens, capacity, [&](int offset, int count) {
        const dim3 grid(p.rows / Schedule::kBlockRows, div_up(count, capacity));
        kernel<<<grid, Schedule::kThreads, bytes, stream>>>(p, output, epilogue, rows, offset);
        CUDA_CHECK(cudaGetLastError());
    });
}

template <class Schedule, class Output, class Epilogue, class Rows = Nvfp4IdentityRows>
void launch_nvfp4_a4_mma(const Nvfp4A4Operands& p, Output output, Epilogue epilogue,
                         cudaStream_t stream, Rows rows = {}) {
    validate_nvfp4_operands<Schedule>(p);
    if (p.rows % Schedule::kBlockRows || p.k % Schedule::kBlockK ||
        p.k / Schedule::kBlockK < Schedule::kStages)
        throw std::invalid_argument(
            "NVFP4 A4 MMA requires complete row/K tiles and enough K stages");
    if (p.scale_layout != Nvfp4ScaleLayout::RowMajor)
        throw std::invalid_argument("NVFP4 A4 MMA requires row-major activation scales");
    for_each_token_slice(p.tokens, Schedule::kBlockTokens, [&](int offset, int count) {
        const dim3 grid(p.rows / Schedule::kBlockRows, div_up(count, Schedule::kBlockTokens));
        const auto launch = [&]<bool Full>() {
            constexpr auto kernel = nvfp4_a4_mma_kernel<Schedule, Full, Epilogue, Output, Rows>;
            const int bytes =
                nvfp4_prepare_shared<nvfp4_mma_shared_bytes<Schedule, Epilogue>, kernel>();
            kernel<<<grid, Schedule::kThreads, bytes, stream>>>(p.x, p.x_scales, p.codes, p.scales,
                                                                p.rows, p.k, p.alpha, output,
                                                                epilogue, rows, offset, count);
            CUDA_CHECK(cudaGetLastError());
        };
        if (count % Schedule::kBlockTokens == 0)
            launch.template operator()<true>();
        else
            launch.template operator()<false>();
    });
}
} // namespace ninfer::ops::detail
