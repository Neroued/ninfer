#include "ninfer/ops/indexing.h"
#include "ops/common/host/operands.h"

#include <algorithm>

namespace ninfer::ops {
namespace {
template <class Word>
__global__ void gather_rows_kernel(const Word* source, const int* rows, Word* out,
                                   std::int64_t elements) {
    const int b = blockIdx.y, row = rows[b];
    for (std::int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < elements;
         i += gridDim.x * blockDim.x)
        out[std::int64_t(b) * elements + i] =
            row < 0 ? Word{} : source[std::int64_t(row) * elements + i];
}

__global__ void gather_field_kernel(const std::uint16_t* source, const std::int64_t* offsets,
                                    const int* active, std::uint16_t* out, int features) {
    const int b       = blockIdx.y;
    const bool live   = active[b] > 0;
    const auto offset = live ? offsets[b] / sizeof(std::uint16_t) : 0;
    for (std::int64_t d = blockIdx.x * blockDim.x + threadIdx.x; d < features;
         d += gridDim.x * blockDim.x)
        out[std::int64_t(b) * features + d] = live ? source[offset + d] : 0;
}
} // namespace

void gather_batch_rows(const Tensor& source, const Tensor& rows, Tensor& out,
                       CudaExecutionView cuda) {
    namespace operand        = common::operand;
    constexpr const char* op = "gather_batch_rows";
    const int d = source.ne[0], t = source.ne[1], s = source.ne[2], b = out.ne[2];
    if (d <= 0 || t <= 0 || s <= 0 || b <= 0 ||
        (source.dtype != DType::BF16 && source.dtype != DType::I32 && source.dtype != DType::I64))
        throw std::invalid_argument("gather_batch_rows: invalid shape or type");
    const auto bytes = dtype_size(source.dtype);
    operand::tensor(source, source.dtype, {d, t, s}, bytes, op, "source", cuda.device());
    operand::tensor(rows, DType::I32, {b}, 4, op, "rows", cuda.device());
    operand::tensor(out, source.dtype, {d, t, b}, bytes, op, "out", cuda.device());
    operand::disjoint(operand::range(out, op), operand::range(source, op), op);
    operand::disjoint(operand::range(out, op), operand::range(rows, op), op);
    const auto elements = std::int64_t(d) * t;
    const dim3 grid(std::min<std::int64_t>(32, (elements + 255) / 256), b);
    const auto launch = [&]<class Word>() {
        gather_rows_kernel<<<grid, 256, 0, cuda.stream()>>>(static_cast<const Word*>(source.data),
                                                            static_cast<const int*>(rows.data),
                                                            static_cast<Word*>(out.data), elements);
    };
    if (bytes == 2)
        launch.template operator()<std::uint16_t>();
    else if (bytes == 4)
        launch.template operator()<std::uint32_t>();
    else
        launch.template operator()<std::uint64_t>();
    CUDA_CHECK(cudaGetLastError());
}

void gather(const ConstStateFieldView& source, const Tensor& active, Tensor& out,
            CudaExecutionView cuda) {
    namespace operand        = common::operand;
    constexpr const char* op = "gather_state_field";
    const int d = out.ne[0], b = out.ne[1];
    if (d <= 0 || b <= 0 || source.field.layers != 1)
        throw std::invalid_argument("gather_state_field: invalid shape");
    operand::tensor(out, DType::BF16, {d, b}, 2, op, "out", cuda.device());
    operand::tensor(active, DType::I32, {b}, 4, op, "active", cuda.device());
    operand::state_field(source, DType::BF16, {d}, 2, op, "source", cuda.device());
    operand::tensor(source.records.byte_offsets, DType::I64, {b}, 8, op, "offsets", cuda.device());
    operand::disjoint(operand::range(out, op), operand::range(active, op), op);
    operand::disjoint(operand::range(out, op), operand::range(source.records.byte_offsets, op), op);
    gather_field_kernel<<<dim3(std::min(32, (d + 255) / 256), b), 256, 0, cuda.stream()>>>(
        operand::state_field_base<const std::uint16_t>(source),
        static_cast<const std::int64_t*>(source.records.byte_offsets.data),
        static_cast<const int*>(active.data), static_cast<std::uint16_t*>(out.data), d);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops
