#include "ninfer/ops/qsa.h"
#include "ops/softmax_attention/qsa/validation.h"

#include <cstdint>

namespace ninfer::ops {
namespace {

template <class T>
struct Columns {
    T* data;
    std::int64_t column, outer, layer;

    __device__ T* at(int feature, std::int64_t c, int row, int l) const {
        return data + feature + c * column + std::int64_t(row) * outer + l * layer;
    }
};

template <class T>
Columns<T> columns(const Tensor& t) {
    return {static_cast<T*>(t.data), t.nb[1] / std::int64_t(sizeof(T)),
            t.nb[2] / std::int64_t(sizeof(T)), t.nb[3] / std::int64_t(sizeof(T))};
}

template <class T>
struct StateColumns {
    T* data;
    const std::int64_t* offsets;
    std::int64_t column, layer;

    __device__ T* at(int feature, std::int64_t c, int row, int l) const {
        return data + offsets[row] / std::int64_t(sizeof(T)) + feature + c * column + l * layer;
    }
};

template <class T, class FieldView>
StateColumns<T> state_columns(const FieldView& view) {
    return {common::operand::state_field_base<T>(view),
            static_cast<const std::int64_t*>(view.records.byte_offsets.data),
            view.field.tensor.shape[0], view.field.layer_stride / std::int64_t(sizeof(T))};
}

struct TailArguments {
    StateColumns<const std::uint16_t> before_raw;
    Columns<const std::uint16_t> record_raw;
    StateColumns<std::uint16_t> after_raw;
    StateColumns<const std::int32_t> before_rope;
    Columns<const std::int32_t> record_rope;
    StateColumns<std::int32_t> after_rope;
    const std::int32_t *base, *accepted;
};

template <class Word>
__global__ void tail_extract_kernel(const __grid_constant__ TailArguments args) {
    constexpr int width = sizeof(Word) / 2;
    const int b = blockIdx.x, l = blockIdx.y, c = threadIdx.x * width;
    const std::int64_t base = args.base[b], frontier = base + args.accepted[b];
    const int length         = frontier % 4;
    const std::int64_t begin = frontier - length;

    Word values[3]{};
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        if (j < length) {
            // If this block still includes before, its old columns retain their indices.
            const auto* input = begin + j < base ? args.before_raw.at(c, j, b, l)
                                                 : args.record_raw.at(c, begin + j - base, b, l);
            values[j]         = *reinterpret_cast<const Word*>(input);
        }
    }
    std::int32_t coordinate = 0;
    if (threadIdx.x < 3 && length != 0) {
        coordinate = begin < base ? *args.before_rope.at(threadIdx.x, 0, b, l)
                                  : *args.record_rope.at(threadIdx.x, begin - base, b, l);
    }
    // Prepare all source words before writing, including exact in-place state updates.
#pragma unroll
    for (int j = 0; j < 3; ++j) *reinterpret_cast<Word*>(args.after_raw.at(c, j, b, l)) = values[j];
    if (threadIdx.x < 3) *args.after_rope.at(threadIdx.x, 0, b, l) = coordinate;
}

bool pair_aligned(const ConstStateFieldView& view) {
    const auto* base = common::operand::state_field_base<const std::byte>(view);
    return view.field.tensor.region.alignment >= 4 &&
           reinterpret_cast<std::uintptr_t>(base) % 4 == 0 &&
           (view.field.layers == 1 || view.field.layer_stride % 4 == 0);
}

bool pair_aligned(const Tensor& t) {
    if (reinterpret_cast<std::uintptr_t>(t.data) % 4) return false;
    for (int d = 1; d <= 3; ++d)
        if (t.ne[d] > 1 && t.nb[d] % 4) return false;
    return true;
}

void launch_tail_extract(const ConstQsaTailStateView& before, const QsaRawRecordReadView& records,
                         const Tensor& accepted, QsaTailStateView after,
                         CudaExecutionView execution) {
    const TailArguments args{state_columns<const std::uint16_t>(before.raw_keys),
                             columns<const std::uint16_t>(records.raw_keys),
                             state_columns<std::uint16_t>(after.raw_keys),
                             state_columns<const std::int32_t>(before.first_rope),
                             columns<const std::int32_t>(records.rope_positions),
                             state_columns<std::int32_t>(after.first_rope),
                             static_cast<const std::int32_t*>(records.base_positions.data),
                             static_cast<const std::int32_t*>(accepted.data)};
    if (pair_aligned(before.raw_keys) && pair_aligned(records.raw_keys) &&
        pair_aligned(after.raw_keys))
        tail_extract_kernel<std::uint32_t>
            <<<dim3(accepted.ne[0], before.raw_keys.field.layers), 64, 0, execution.stream()>>>(
                args);
    else
        tail_extract_kernel<std::uint16_t>
            <<<dim3(accepted.ne[0], before.raw_keys.field.layers), 128, 0, execution.stream()>>>(
                args);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void qsa_tail_extract(const ConstQsaTailStateView& before, const QsaRawRecordReadView& records,
                      const Tensor& accepted, QsaTailStateView after, CudaExecutionView execution) {
    const detail::qsa::ViewValidation v{"qsa_tail_extract", execution.device()};
    const int batch = accepted.ne[0], capacity = records.raw_keys.ne[1];
    v.require(batch >= 1 && batch <= 8, "batch extent");
    v.metadata(accepted, batch, "accepted columns");
    v.metadata(records.base_positions, batch, "record base positions");
    v.metadata(records.valid_columns, batch, "record valid columns");
    const auto layers = records.raw_keys.ne[3];
    v.require(layers >= 1 && layers <= 65535, "layer extent");
    v.state(before, batch, layers);
    v.state(after, batch, layers);
    common::operand::strided_tensor(records.raw_keys, DType::BF16, {128, capacity, batch, layers},
                                    2, v.op, "record raw keys", execution.device());
    common::operand::strided_tensor(records.rope_positions, DType::I32,
                                    {3, capacity, batch, layers}, 4, v.op, "record coordinates",
                                    execution.device());
    v.require(records.raw_keys.nb[0] == 2 && records.rope_positions.nb[0] == 4,
              "contiguous record features");
    // Only Device-selected records are accessed. Overlapping pool bounds do not imply that
    // selected regions overlap; the caller guarantees their cross-row/plane relationships.
    launch_tail_extract(before, records, accepted, after, execution);
}

} // namespace ninfer::ops
