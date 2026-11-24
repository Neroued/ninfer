#include "ops/common/host/operands.h"
#include "ninfer/ops/indexing.h"
#include <algorithm>
#include <stdexcept>
#include "ops/common/numeric/math.h"
#include "core/device.h"
#include <cstdint>
#include <cuda_bf16.h>
#include <cstddef>

namespace ninfer::ops {
using namespace common;

namespace {

template <class Word>
__global__ void scatter_columns_kernel(const Word* source, const int* columns,
                                       const std::int64_t* offsets, Word* destination, int features,
                                       int tokens) {
    const int row = blockIdx.y, column = columns[row];
    if (column < 0) return;
    const std::int64_t destination_base = offsets[row] / sizeof(Word);
    for (int d = blockIdx.x * blockDim.x + threadIdx.x; d < features; d += blockDim.x * gridDim.x)
        destination[destination_base + d] =
            source[(std::int64_t(row) * tokens + column) * features + d];
}

__global__ void gather_bf16_batch_columns_kernel(const __nv_bfloat16* source,
                                                 const std::int32_t* selectors, __nv_bfloat16* out,
                                                 std::int32_t rows, std::int32_t cols) {
    const int batch = static_cast<int>(blockIdx.y);
    const int row   = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) { return; }
    const int col = selectors[batch];
    if (col < 0 || col >= cols) { return; }
    out[static_cast<std::int64_t>(batch) * rows + row] =
        source[(static_cast<std::int64_t>(batch) * cols + col) * rows + row];
}

__global__ void remap_ids_inplace_kernel(std::int32_t* ids, std::int32_t count,
                                         const std::int32_t* id_map, std::int32_t n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    const int idx = ids[i];
    if (idx >= 0 && idx < n) { ids[i] = id_map[idx]; }
}

void gather_bf16_batch_columns_launch(const Tensor& source, const Tensor& selectors, Tensor& out,
                                      cudaStream_t stream) {
    constexpr int kBlock = 256;
    const int rows       = source.ne[0];
    const int batch      = source.ne[2];
    const dim3 grid(static_cast<unsigned int>(std::max(1, div_up(rows, kBlock))),
                    static_cast<unsigned int>(batch));
    gather_bf16_batch_columns_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(source.data),
        static_cast<const std::int32_t*>(selectors.data), static_cast<__nv_bfloat16*>(out.data),
        rows, source.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

void remap_ids_inplace_launch(Tensor& ids, const std::int32_t* id_map, std::int32_t n,
                              cudaStream_t stream) {
    constexpr int kBlock = 256;
    const int count      = ids.ne[0];
    const int grid       = std::max(1, div_up(count, kBlock));
    remap_ids_inplace_kernel<<<grid, kBlock, 0, stream>>>(static_cast<std::int32_t*>(ids.data),
                                                          count, id_map, n);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void gather_bf16_batch_columns(const Tensor& source, const Tensor& selectors, Tensor& out,
                               CudaExecutionView execution) {
    const auto stream        = execution.stream();
    constexpr const char* op = "gather_bf16_batch_columns";
    common::operand::dtype(source, DType::BF16, op, "source");
    common::operand::contiguous_nonnull(source, op, "source", execution.device());
    if (source.ne[0] <= 0 || source.ne[1] <= 0 || source.ne[2] <= 0 || source.ne[3] != 1) {
        throw std::invalid_argument("gather_bf16_batch_columns: invalid source shape");
    }
    common::operand::tensor(selectors, DType::I32, {source.ne[2]}, 1, op, "selectors",
                            execution.device());
    common::operand::dtype(out, DType::BF16, op, "out");
    common::operand::contiguous_nonnull(out, op, "out", execution.device());
    if (out.ne[0] != source.ne[0] || out.ne[1] != source.ne[2] || out.ne[2] != 1 ||
        out.ne[3] != 1) {
        throw std::invalid_argument("gather_bf16_batch_columns: invalid output shape");
    }
    gather_bf16_batch_columns_launch(source, selectors, out, stream);
}

void remap_ids_inplace(Tensor& ids, const std::int32_t* id_map, std::int32_t n,
                       CudaExecutionView execution) {
    const auto stream        = execution.stream();
    constexpr const char* op = "remap_ids_inplace";
    common::operand::dtype(ids, DType::I32, op, "ids");
    common::operand::contiguous_nonnull(ids, op, "ids", execution.device());
    if (ids.ne[0] <= 0 || ids.ne[1] != 1 || ids.ne[2] != 1 || ids.ne[3] != 1) {
        throw std::invalid_argument("remap_ids_inplace: ids must be a non-empty vector");
    }
    if (id_map == nullptr || n <= 0) {
        throw std::invalid_argument("remap_ids_inplace: id_map must be non-null and n>0");
    }
    remap_ids_inplace_launch(ids, id_map, n, stream);
}

void scatter_batch_columns(const Tensor& source, const Tensor& columns,
                           const StateFieldView& destination, CudaExecutionView execution) {
    namespace operand        = common::operand;
    constexpr const char* op = "scatter_batch_columns";
    const int d = source.ne[0], t = source.ne[1], b = source.ne[2];
    if (d <= 0 || t <= 0 || b < 1 || b > 8 ||
        (source.dtype != DType::BF16 && source.dtype != DType::I32))
        throw std::invalid_argument("scatter_batch_columns: invalid shape or type");
    const auto bytes = dtype_size(source.dtype);
    operand::tensor(source, source.dtype, {d, t, b}, bytes, op, "source", execution.device());
    operand::tensor(columns, DType::I32, {b}, 4, op, "columns", execution.device());
    operand::state_field(destination, source.dtype, {d}, bytes, op, "destination",
                         execution.device());
    operand::tensor(destination.records.byte_offsets, DType::I64, {b}, 8, op, "offsets",
                    execution.device());
    if (destination.field.layers != 1)
        throw std::invalid_argument("scatter_batch_columns: expected a single-layer field");
    const dim3 grid(std::min(32, div_up(d, 256)), b);
    const auto launch = [&]<class Word>() {
        scatter_columns_kernel<<<grid, 256, 0, execution.stream()>>>(
            static_cast<const Word*>(source.data), static_cast<const int*>(columns.data),
            static_cast<const std::int64_t*>(destination.records.byte_offsets.data),
            operand::state_field_base<Word>(destination), d, t);
    };
    if (source.dtype == DType::BF16)
        launch.template operator()<std::uint16_t>();
    else
        launch.template operator()<std::uint32_t>();
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
