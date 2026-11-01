#pragma once

#include "ops/softmax_attention/kpool_mla/index_fp8.cuh"
#include "ops/softmax_attention/kpool_mla/indexer/arguments.h"

namespace ninfer::ops::detail::kpool {

__global__ __launch_bounds__(128) void indexer_query_fp8_kernel(
    const __grid_constant__ Fp8IndexerArguments f) {
    const auto& a = f.index;
    if (a.columns <= kIndexSelected) return;
    const int row = blockIdx.x / 8, b = row / f.query_width;
    const int t = row % f.query_width + f.query_begin;
    if (t >= a.valid[b]) return;
    const auto column = std::int64_t(b) * a.tokens + t;
    if ((a.positions[column] + 1) / 4 <= kIndexSelected) return;
    const int lane = threadIdx.x & 31, head = blockIdx.x % 8 * 4 + threadIdx.x / 32;
    const auto* src   = a.query + column * 4096 + head * 128 + lane * 4;
    const auto raw    = common::load_vec<uint2>(src);
    const auto* words = reinterpret_cast<const __nv_bfloat16*>(&raw);
    float values[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) values[j] = __bfloat162float(words[j]);
    const auto encoded = encode_index_lane<true>(values);
    reinterpret_cast<unsigned*>(f.query + row * 4096 + head * 128)[lane] = encoded.codes;
    if (lane == 0)
        f.coefficients[row * 32 + head] =
            a.coefficients[column * 32 + head] * common::ue8m0_to_float(encoded.scale);
}
} // namespace ninfer::ops::detail::kpool
