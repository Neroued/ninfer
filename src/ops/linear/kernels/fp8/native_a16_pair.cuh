#pragma once

#include "ops/linear/kernels/fp8/native_a16.cuh"
#include "ops/linear/kernels/fp8/native_a16_sliced_k.cuh"
#include "ops/linear/kernels/bf16/bf16_operands.h"
#include "ops/linear/kernels/bf16/bf16_a16_simt.cuh"
#include "ops/linear/kernels/bf16/bf16_a16_sliced_k_mma.cuh"

namespace ninfer::ops::detail {

// A CTA belongs to one parent for its whole lifetime. Each branch consumes the same
// contraction body as the homogeneous entry, with independently mapped output rows.
template <class EncodedSchedule, class DenseSchedule, bool Sliced, bool Interleave, class Encoded,
          class EncodedOutput, class DenseOutput, class EncodedEpilogue, class DenseEpilogue>
__global__ __launch_bounds__(EncodedSchedule::kThreads) void native_fp8_bf16_pair_kernel(
    Encoded encoded, Bf16A16Operands dense, EncodedOutput encoded_output, DenseOutput dense_output,
    EncodedEpilogue encoded_epilogue, DenseEpilogue dense_epilogue, int encoded_blocks,
    int token_tiles) {
    using E = EncodedSchedule;
    using D = DenseSchedule;
    static_assert(E::kThreads == D::kThreads && E::kBlockTokens == D::kBlockTokens);
    int index = blockIdx.x;
    if constexpr (Interleave) {
        const int pairs = min(encoded_blocks, int(gridDim.x) - encoded_blocks);
        if (index < 2 * pairs)
            index = (index & 1) ? index / 2 : encoded_blocks + index / 2;
        else if (encoded_blocks > pairs)
            index -= pairs;
        else
            index += encoded_blocks - pairs;
    }
    const auto skip = [&](const auto& output, int block, int block_rows, int rows) {
        if constexpr (requires { output.clear_if_empty(0, 0, 0, 0); })
            return output.clear_if_empty(
                block / token_tiles * block_rows, block % token_tiles * E::kBlockTokens,
                min(block_rows, rows - block / token_tiles * block_rows),
                min(E::kBlockTokens, encoded.tokens - block % token_tiles * E::kBlockTokens));
        return false;
    };

    if constexpr (Sliced) {
        constexpr int encoded_shared = NativeFp8SlicedKStorage<E, typename Encoded::InputType,
                                                               typename Encoded::ScaleType>::kBytes;
        constexpr int bytes = encoded_shared > D::kSharedBytes ? encoded_shared : D::kSharedBytes;
        auto* scratch       = common::shared_storage<bytes>();
        if (index < encoded_blocks) {
            if (skip(encoded_output, index, E::kBlockRows, encoded.rows)) return;
            native_fp8_a16_sliced_k_body<E>(
                encoded.x, encoded.codes, encoded.scales, encoded.rows, encoded.k, encoded.tokens,
                encoded_output, encoded_epilogue, Fp8IdentityRows{}, index / token_tiles,
                index % token_tiles * E::kBlockTokens, scratch);
        } else {
            const int block = index - encoded_blocks;
            if (skip(dense_output, block, D::kBlockRows, dense.rows)) return;
            bf16_a16_sliced_k_mma_body<D>(dense.x, dense.weight, dense_output, dense_epilogue,
                                          dense.rows, dense.k, dense.tokens, block / token_tiles,
                                          block % token_tiles * D::kBlockTokens, scratch);
        }
    } else {
        using Storage = Bf16SimtSharedStorage<D, D::kBlockTokens>;
        auto& scratch = *reinterpret_cast<Storage*>(common::shared_storage<sizeof(Storage)>());
        if (index < encoded_blocks) {
            if (skip(encoded_output, index, E::kBlockRows, encoded.rows)) return;
            native_fp8_a16_simt_body<E>(encoded, encoded_output, encoded_epilogue,
                                        Fp8IdentityRows{}, index / token_tiles,
                                        index % token_tiles * E::kBlockTokens);
        } else {
            const int block = index - encoded_blocks;
            if (skip(dense_output, block, D::kBlockRows, dense.rows)) return;
            bf16_a16_simt_body<D>(dense.x, dense.weight, dense_output, dense_epilogue, dense.k,
                                  dense.tokens, block / token_tiles,
                                  block % token_tiles * D::kBlockTokens, scratch);
        }
    }
}

template <class EncodedSchedule, class DenseSchedule, bool Sliced, bool Interleave = false,
          class Encoded, class EncodedOutput, class DenseOutput, class EncodedEpilogue,
          class DenseEpilogue>
void launch_native_fp8_bf16_pair(const Encoded& encoded, const Bf16A16Operands& dense,
                                 EncodedOutput encoded_output, DenseOutput dense_output,
                                 EncodedEpilogue encoded_epilogue, DenseEpilogue dense_epilogue,
                                 cudaStream_t stream) {
    using E = EncodedSchedule;
    using D = DenseSchedule;
    static_assert(E::kThreads == D::kThreads && E::kBlockTokens == D::kBlockTokens);
    validate_native_fp8<E>(encoded);
    validate_bf16_operands<D>(dense);
    if (encoded.tokens != dense.tokens || dense.rows % D::kBlockRows)
        throw std::invalid_argument(
            "paired FP8/BF16 contraction requires matching T and dense row tiles");
    const auto token_tiles = common::div_up(encoded.tokens, E::kBlockTokens);
    const auto encoded_blocks =
        std::int64_t(common::div_up(encoded.rows, E::kBlockRows)) * token_tiles;
    const auto blocks = encoded_blocks + std::int64_t(dense.rows / D::kBlockRows) * token_tiles;
    if (blocks > 2147483647LL)
        throw std::invalid_argument("paired FP8/BF16 contraction exceeds CUDA grid.x");
    constexpr int bytes = [] {
        if constexpr (Sliced) {
            constexpr int first = NativeFp8SlicedKStorage<E, typename Encoded::InputType,
                                                          typename Encoded::ScaleType>::kBytes;
            return first > D::kSharedBytes ? first : D::kSharedBytes;
        } else
            return int(sizeof(Bf16SimtSharedStorage<D, D::kBlockTokens>));
    }();
    constexpr auto kernel =
        native_fp8_bf16_pair_kernel<E, D, Sliced, Interleave, Encoded, EncodedOutput, DenseOutput,
                                    EncodedEpilogue, DenseEpilogue>;
    const auto shared = common::prepare_shared<bytes, kernel>();
    kernel<<<static_cast<unsigned>(blocks), E::kThreads, shared, stream>>>(
        encoded, dense, encoded_output, dense_output, encoded_epilogue, dense_epilogue,
        static_cast<int>(encoded_blocks), token_tiles);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
