#include "ninfer/ops/span_accumulate.h"

#include "core/device.h"

#include <cuda_bf16.h>

#include <stdexcept>

#include <cstdint>

namespace ninfer::ops {

namespace {

constexpr int kThreads = 256;

__global__ void span_accumulate_kernel(const __nv_bfloat16* __restrict__ x,
                                       std::int64_t column_stride, float* __restrict__ sums,
                                       std::int32_t rows, std::int32_t begin,
                                       std::int32_t count) {
    const std::int32_t row = blockIdx.x * kThreads + threadIdx.x;
    if (row >= rows || count == 0) { return; }
    // Feature-first layout: the rows of one token are contiguous, so the walk visits
    // each span column base and reads this thread's row element.
    float acc = sums[row];
    for (std::int32_t token = begin; token < begin + count; ++token) {
        acc += __bfloat162float(x[static_cast<std::int64_t>(token) * column_stride + row]);
    }
    sums[row] = acc;
}

void require_span_accumulate_shapes(const Tensor& x, const Tensor& sums,
                                    std::int32_t& rows, std::int32_t& tokens) {
    if (x.dtype != DType::BF16 || x.ne[2] != 1 || x.ne[3] != 1 || sums.dtype != DType::FP32 ||
        sums.ne[1] != 1 || sums.ne[2] != 1 || sums.ne[3] != 1 || !x.data || !sums.data) {
        throw std::invalid_argument("span accumulate: invalid operand dtypes or shapes");
    }
    rows   = static_cast<std::int32_t>(x.ne[0]);
    tokens = static_cast<std::int32_t>(x.ne[1]);
    if (rows <= 0 || tokens <= 0 || sums.ne[0] != x.ne[0] || x.is_contiguous() == false ||
        sums.is_contiguous() == false) {
        throw std::invalid_argument("span accumulate: operands must be contiguous [rows,tokens]/[rows]");
    }
}

} // namespace

void span_accumulate(const Tensor& x, std::uint32_t begin, std::uint32_t count, Tensor& sums,
                     cudaStream_t stream) {
    std::int32_t rows = 0;
    std::int32_t tokens = 0;
    require_span_accumulate_shapes(x, sums, rows, tokens);
    if (static_cast<std::int64_t>(begin) + static_cast<std::int64_t>(count) > tokens) {
        throw std::invalid_argument("span accumulate: span is outside the token extent");
    }
    if (count == 0) { return; }
    const std::int64_t column_stride = x.nb[1] / static_cast<std::int64_t>(sizeof(__nv_bfloat16));
    const int blocks                 = (rows + kThreads - 1) / kThreads;
    span_accumulate_kernel<<<blocks, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), column_stride,
        static_cast<float*>(sums.data), rows, static_cast<std::int32_t>(begin),
        static_cast<std::int32_t>(count));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
