#include "ninfer/ops/sequence.h"
#include "ops/common/host/operands.h"

namespace ninfer::ops {
namespace {
namespace operand = common::operand;

template <class T>
struct View {
    T* data;
    std::int64_t x, y, z;

    __device__ T& at(int i, int j = 0, int k = 0) const { return data[i * x + j * y + k * z]; }
};

template <class T>
View<T> view(const Tensor& t) {
    return {static_cast<T*>(t.data), t.nb[0] / std::int64_t(sizeof(T)),
            t.nb[1] / std::int64_t(sizeof(T)), t.nb[2] / std::int64_t(sizeof(T))};
}

struct Pairs {
    View<const uint4> hidden, previous;
    View<uint4> paired;
    View<const int> ids, positions, rope, previous_rope;
    View<int> paired_ids, paired_positions, paired_rope;
    const int *prefix, *skip, *rows;
    int* lengths;
    int vectors, axes;
};

__global__ void pair_kernel(const __grid_constant__ Pairs a) {
    const int j = blockIdx.x, r = blockIdx.y;
    const int b    = a.rows ? a.rows[r] : r;
    const int skip = b < 0 ? 0 : a.skip[b];
    const int n    = b < 0 ? 0 : max(a.prefix[b] - skip, 0);
    const int i    = j + skip;
    for (int d = threadIdx.x; d < a.vectors; d += blockDim.x) {
        uint4 value{};
        if (j < n) value = i ? a.hidden.at(d, i - 1, b) : a.previous.at(d, b);
        a.paired.at(d, j, r) = value;
    }
    if (threadIdx.x == 0) {
        a.paired_ids.at(j, r)       = j < n ? a.ids.at(i, b) : 0;
        a.paired_positions.at(j, r) = j < n ? a.positions.at(i, b) - 1 : 0;
        if (!j) a.lengths[r] = n;
    }
    if (threadIdx.x < a.axes) {
        const int axis = threadIdx.x;
        a.paired_rope.at(j, axis, r) =
            j < n ? (i ? a.rope.at(i - 1, axis, b) : a.previous_rope.at(axis, b)) : 0;
    }
}

void check_hidden(const Tensor& value, std::initializer_list<int> shape, const char* name,
                  CudaExecutionView cuda) {
    constexpr const char* op = "prepare_causal_pairs";
    operand::strided_tensor(value, DType::BF16, shape, 16, op, name, cuda.device());
    if (value.nb[0] != 2 || value.nb[1] % 16 || value.nb[2] % 16)
        throw std::invalid_argument("prepare_causal_pairs: invalid hidden strides");
}
} // namespace

void prepare_causal_pairs(const CausalPairInputs& in, const CausalPairOutputs& out,
                          CudaExecutionView cuda) {
    constexpr const char* op = "prepare_causal_pairs";
    const int d = in.hidden.ne[0], w = in.hidden.ne[1], b = in.hidden.ne[2];
    const int p = out.hidden.ne[1], c = out.hidden.ne[2];
    if (d <= 0 || d % 8 || w < 1 || p < 1 || b < 1 || b > 8 || c < 1 || c > 8 ||
        (!in.rows.data && c != b))
        throw std::invalid_argument("prepare_causal_pairs: invalid shape");
    check_hidden(in.hidden, {d, w, b}, "hidden", cuda);
    check_hidden(in.previous_hidden, {d, b}, "previous hidden", cuda);
    check_hidden(out.hidden, {d, p, c}, "paired hidden", cuda);
    for (const auto* value : {&in.ids, &in.positions})
        operand::strided_tensor(*value, DType::I32, {w, b}, 4, op, "input metadata", cuda.device());
    for (const auto* value : {&in.prefix, &in.skip_first})
        operand::tensor(*value, DType::I32, {b}, 4, op, "row metadata", cuda.device());
    if (in.rows.data) operand::tensor(in.rows, DType::I32, {c}, 4, op, "rows", cuda.device());
    for (const auto* value : {&out.ids, &out.positions})
        operand::strided_tensor(*value, DType::I32, {p, c}, 4, op, "output metadata",
                                cuda.device());
    operand::tensor(out.lengths, DType::I32, {c}, 4, op, "lengths", cuda.device());
    int axes = 0;
    if (in.rope.data || in.previous_rope.data || out.rope.data) {
        axes = in.rope.ne[1];
        if (axes != 1 && axes != 3)
            throw std::invalid_argument("prepare_causal_pairs: invalid coordinate axes");
        operand::strided_tensor(in.rope, DType::I32, {w, axes, b}, 4, op, "rope", cuda.device());
        operand::strided_tensor(in.previous_rope, DType::I32, {axes, b}, 4, op, "previous rope",
                                cuda.device());
        operand::strided_tensor(out.rope, DType::I32, {p, axes, c}, 4, op, "paired rope",
                                cuda.device());
    }
    const std::array outputs{&out.hidden, &out.ids, &out.positions, &out.lengths, &out.rope};
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        if (!outputs[i]->data) continue;
        const auto range = operand::strided_range(*outputs[i], op);
        for (std::size_t j = 0; j < i; ++j)
            if (outputs[j]->data)
                operand::disjoint(range, operand::strided_range(*outputs[j], op), op);
        for (const auto* input : {&in.hidden, &in.previous_hidden, &in.ids, &in.positions, &in.rope,
                                  &in.previous_rope, &in.prefix, &in.skip_first, &in.rows})
            if (input->data) operand::disjoint(range, operand::strided_range(*input, op), op);
    }
    auto h = view<const uint4>(in.hidden), previous = view<const uint4>(in.previous_hidden);
    auto result = view<uint4>(out.hidden);
    h.x = previous.x = result.x = 1;
    const Pairs args{h,
                     previous,
                     result,
                     view<const int>(in.ids),
                     view<const int>(in.positions),
                     view<const int>(in.rope),
                     view<const int>(in.previous_rope),
                     view<int>(out.ids),
                     view<int>(out.positions),
                     view<int>(out.rope),
                     static_cast<const int*>(in.prefix.data),
                     static_cast<const int*>(in.skip_first.data),
                     static_cast<const int*>(in.rows.data),
                     static_cast<int*>(out.lengths.data),
                     d / 8,
                     axes};
    pair_kernel<<<dim3(p, c), 128, 0, cuda.stream()>>>(args);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops
