#include "ninfer/ops/indexing.h"
#include "ops/common/host/operands.h"

#include <algorithm>
#include <array>

namespace ninfer::ops {
namespace {
namespace operand     = common::operand;
constexpr int kFields = 8;

struct Field {
    const char* source;
    char* destination;
    const std::int64_t* offsets;
    const int* selectors;
    std::int64_t column_stride, row_stride;
    int features, tiles, end, word_bytes;
};

struct Fields {
    Field values[kFields];
};

template <class Word, bool Gather>
__device__ void assign(const Field& f, int block) {
    const int row = block / f.tiles, tile = block % f.tiles;
    const int selected = f.selectors[row];
    if constexpr (!Gather)
        if (selected < 0) return;
    const bool live    = !Gather || selected > 0;
    const Word* source = nullptr;
    Word* destination;
    if constexpr (Gather) {
        if (live) source = reinterpret_cast<const Word*>(f.source + f.offsets[row]);
        destination = reinterpret_cast<Word*>(f.destination + row * f.row_stride);
    } else {
        source      = reinterpret_cast<const Word*>(f.source + row * f.row_stride +
                                                    selected * f.column_stride);
        destination = reinterpret_cast<Word*>(f.destination + f.offsets[row]);
    }
    for (int d = tile * blockDim.x + threadIdx.x; d < f.features; d += f.tiles * blockDim.x)
        destination[d] = live ? source[d] : Word{};
}

template <bool Gather>
__global__ void assign_fields(const __grid_constant__ Fields fields) {
    int index = 0, begin = 0;
    while (int(blockIdx.x) >= fields.values[index].end) begin = fields.values[index++].end;
    const auto& field = fields.values[index];
    if (field.word_bytes == 2)
        assign<std::uint16_t, Gather>(field, int(blockIdx.x) - begin);
    else
        assign<std::uint32_t, Gather>(field, int(blockIdx.x) - begin);
}

template <class View>
void check_state(const View& state, DType dtype, int features, int batch, const Tensor& selector,
                 const char* op, CudaExecutionView cuda) {
    if (features <= 0 || batch < 1 || batch > 8 || state.field.layers != 1 ||
        (dtype != DType::BF16 && dtype != DType::I32))
        throw std::invalid_argument(std::string(op) + ": invalid field geometry or type");
    operand::state_field(state, dtype, {features}, dtype_size(dtype), op, "state", cuda.device());
    operand::tensor(state.records.byte_offsets, DType::I64, {batch}, 8, op, "offsets",
                    cuda.device());
    operand::tensor(selector, DType::I32, {batch}, 4, op, "selectors", cuda.device());
}

void check_fields(std::size_t count, const char* op) {
    if (count > kFields)
        throw std::invalid_argument(std::string(op) + ": at most eight fields are supported");
}
} // namespace

void gather_state_fields(std::span<const StateGather> fields, CudaExecutionView cuda) {
    constexpr const char* op = "gather_state_fields";
    check_fields(fields.size(), op);
    if (fields.empty()) return;
    Fields args{};
    int blocks = 0;
    for (std::size_t i = 0; i < fields.size(); ++i) {
        const auto& f   = fields[i];
        const auto& out = f.output;
        const int d = out.ne[0], b = out.ne[1];
        check_state(f.source, out.dtype, d, b, f.active, op, cuda);
        operand::strided_tensor(out, out.dtype, {d, b}, dtype_size(out.dtype), op, "output",
                                cuda.device());
        if (out.nb[0] != dtype_size(out.dtype))
            throw std::invalid_argument("gather_state_fields: features must be contiguous");
        for (std::size_t j = 0; j < fields.size(); ++j) {
            operand::disjoint(operand::strided_range(out, op), operand::range(fields[j].active, op),
                              op);
            operand::disjoint(operand::strided_range(out, op),
                              operand::range(fields[j].source.records.byte_offsets, op), op);
            if (j < i)
                operand::disjoint(operand::strided_range(out, op),
                                  operand::strided_range(fields[j].output, op), op);
        }
        const int tiles = std::min(32, (d - 1) / 256 + 1);
        blocks += tiles * b;
        args.values[i] = {operand::state_field_base<const char>(f.source),
                          static_cast<char*>(out.data),
                          static_cast<const std::int64_t*>(f.source.records.byte_offsets.data),
                          static_cast<const int*>(f.active.data),
                          0,
                          out.nb[1],
                          d,
                          tiles,
                          blocks,
                          int(dtype_size(out.dtype))};
    }
    assign_fields<true><<<blocks, 256, 0, cuda.stream()>>>(args);
    CUDA_CHECK(cudaGetLastError());
}

void scatter_state_fields(std::span<const StateColumnScatter> fields, CudaExecutionView cuda) {
    constexpr const char* op = "scatter_state_fields";
    check_fields(fields.size(), op);
    if (fields.empty()) return;
    Fields args{};
    int blocks = 0;
    for (std::size_t i = 0; i < fields.size(); ++i) {
        const auto& f      = fields[i];
        const auto& source = f.source;
        const int d = source.ne[0], t = source.ne[1], b = source.ne[2];
        check_state(f.destination, source.dtype, d, b, f.columns, op, cuda);
        operand::strided_tensor(source, source.dtype, {d, t, b}, dtype_size(source.dtype), op,
                                "source", cuda.device());
        if (source.nb[0] != dtype_size(source.dtype))
            throw std::invalid_argument("scatter_state_fields: features must be contiguous");
        const int tiles = std::min(32, (d - 1) / 256 + 1);
        blocks += tiles * b;
        args.values[i] = {static_cast<const char*>(source.data),
                          operand::state_field_base<char>(f.destination),
                          static_cast<const std::int64_t*>(f.destination.records.byte_offsets.data),
                          static_cast<const int*>(f.columns.data),
                          source.nb[1],
                          source.nb[2],
                          d,
                          tiles,
                          blocks,
                          int(dtype_size(source.dtype))};
    }
    assign_fields<false><<<blocks, 256, 0, cuda.stream()>>>(args);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops
