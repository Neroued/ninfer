#pragma once

#include "ops/linear/swiglu/product_epilogue.cuh"
#include "ops/linear/swiglu/row_major_mma_epilogue.cuh"
#include "ops/linear/swiglu/q8/output.cuh"
#include "ops/linear/kernels/q8/q8_sliced_k_launch.cuh"
#include "ops/linear/kernels/q8/q8_a16_tma.cuh"
#include "ops/common/cuda/reduce.cuh"
#include "ops/common/numeric/convert.cuh"

namespace ninfer::ops::detail {
template <int Hidden, int Intermediate, int RowsPerCta, int WarpsPerRow, int TokenCapacity,
          class Output, class Product>
__global__ __launch_bounds__(RowsPerCta* WarpsPerRow * 32,
                             2) void q8_swiglu_vector_kernel(Q8LinearOperands p, Output output,
                                                             Product product) {
    static_assert(Hidden % 256 == 0 && Intermediate % RowsPerCta == 0);
    static_assert(WarpsPerRow > 0 && RowsPerCta * WarpsPerRow <= 32 && TokenCapacity > 0);
    const int lane = int(threadIdx.x) & 31, warp = int(threadIdx.x) / 32;
    const int local_row = warp / WarpsPerRow, split = warp % WarpsPerRow;
    const int row = int(blockIdx.x) * RowsPerCta + local_row;
    float gate[TokenCapacity][4]{}, up[TokenCapacity][4]{};
#pragma unroll 1
    for (int base = (split * 32 + lane) * 8; base < Hidden; base += 256 * WarpsPerRow) {
        const auto g = common::load_vec<uint2>(p.codes + std::int64_t(row) * Hidden + base);
        const auto u =
            common::load_vec<uint2>(p.codes + std::int64_t(row + Intermediate) * Hidden + base);
        const float gs = __half2float(__ushort_as_half(common::load_ldg<std::uint16_t>(
            p.scales + (std::int64_t(row) * (Hidden / 32) + base / 32) * 2)));
        const float us = __half2float(__ushort_as_half(common::load_ldg<std::uint16_t>(
            p.scales + (std::int64_t(row + Intermediate) * (Hidden / 32) + base / 32) * 2)));
#pragma unroll
        for (int token = 0; token < TokenCapacity; ++token) {
            if constexpr (TokenCapacity > 1) {
                if (token >= p.tokens) continue;
            }
            const auto raw = common::load_vec<uint4>(p.x + std::int64_t(token) * Hidden + base);
            const float2 values[]{
                common::bf16x2_bits_to_float2(raw.x), common::bf16x2_bits_to_float2(raw.y),
                common::bf16x2_bits_to_float2(raw.z), common::bf16x2_bits_to_float2(raw.w)};
#pragma unroll
            for (int word = 0; word < 2; ++word) {
                const unsigned gb = word ? g.y : g.x, ub = word ? u.y : u.x;
#pragma unroll
                for (int byte = 0; byte < 4; ++byte) {
                    const auto pair   = values[word * 2 + byte / 2];
                    const float x     = (byte & 1) ? pair.y : pair.x;
                    gate[token][byte] = fmaf(float(static_cast<std::int8_t>(gb >> (byte * 8))) * gs,
                                             x, gate[token][byte]);
                    up[token][byte]   = fmaf(float(static_cast<std::int8_t>(ub >> (byte * 8))) * us,
                                             x, up[token][byte]);
                }
            }
        }
    }
    if constexpr (WarpsPerRow == 1) {
#pragma unroll
        for (int token = 0; token < TokenCapacity; ++token) {
            const float g = common::warp_reduce_sum((gate[token][0] + gate[token][1]) +
                                                    (gate[token][2] + gate[token][3]));
            const float u = common::warp_reduce_sum((up[token][0] + up[token][1]) +
                                                    (up[token][2] + up[token][3]));
            if (lane == 0 && token < p.tokens) output.store(row, token, product(g, u));
        }
    } else {
        __shared__ float partials[RowsPerCta][WarpsPerRow][TokenCapacity][2];
#pragma unroll
        for (int token = 0; token < TokenCapacity; ++token) {
            const float g = common::warp_reduce_sum((gate[token][0] + gate[token][1]) +
                                                    (gate[token][2] + gate[token][3]));
            const float u = common::warp_reduce_sum((up[token][0] + up[token][1]) +
                                                    (up[token][2] + up[token][3]));
            if (lane == 0) {
                partials[local_row][split][token][0] = g;
                partials[local_row][split][token][1] = u;
            }
        }
        __syncthreads();
        if (split == 0 && lane == 0) {
#pragma unroll
            for (int token = 0; token < TokenCapacity; ++token) {
                float g = 0, u = 0;
#pragma unroll
                for (int part = 0; part < WarpsPerRow; ++part) {
                    g += partials[local_row][part][token][0];
                    u += partials[local_row][part][token][1];
                }
                if (token < p.tokens) output.store(row, token, product(g, u));
            }
        }
    }
}

template <int Hidden, int Intermediate, int RowsPerCta = 4, int WarpsPerRow = 1,
          int TokenCapacity = 1, class Output, class Product>
void launch_q8_swiglu_vector(const Q8LinearOperands& p, Output output, Product product,
                             cudaStream_t stream) {
    validate_q8_operands(p);
    if (p.rows != 2 * Intermediate || p.k != Hidden || p.padded_k != Hidden || p.tokens <= 0 ||
        p.tokens > TokenCapacity)
        throw std::invalid_argument("Q8 SwiGLU vector geometry mismatch");
    q8_swiglu_vector_kernel<Hidden, Intermediate, RowsPerCta, WarpsPerRow, TokenCapacity>
        <<<Intermediate / RowsPerCta, RowsPerCta * WarpsPerRow * 32, 0, stream>>>(p, output,
                                                                                  product);
    CUDA_CHECK(cudaGetLastError());
}

template <class Product>
struct Q8SwiGluProductFragment {
    Product product;

    template <class Output>
    __device__ void store_fragment(Output output, int row, int col, float4 value, int,
                                   int tokens) const {
        if (col < tokens) output.store(row, col, product(value.x, value.z));
        if (col + 1 < tokens) output.store(row, col + 1, product(value.y, value.w));
    }
};

template <class Schedule, int Intermediate, class Output, class Product>
void launch_q8_swiglu_sliced(const Q8LinearOperands& p, Output output, Product product,
                             cudaStream_t stream) {
    launch_q8_a16_sliced_k_mma<Schedule, Q8SwiGluPairedRows<Intermediate>>(
        p, output, Q8SwiGluProductFragment<Product>{product}, stream);
}

template <class Schedule, class Output, class Product>
void launch_q8_swiglu_tma(const Q8LinearOperands& p, Output output, Product product,
                          cudaStream_t stream) {
    launch_q8_a16_tma_mma<Schedule, SwiGluRowMajorMmaRows<Schedule>>(
        p, output, BasicSwiGluRowMajorMmaEpilogue<Product>{product}, stream);
}
} // namespace ninfer::ops::detail
