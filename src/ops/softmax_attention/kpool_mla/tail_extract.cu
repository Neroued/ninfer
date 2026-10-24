#include "ops/softmax_attention/kpool_mla/validation.h"
#include "ninfer/ops/kpool_mla.h"

#include <cuda_bf16.h>

namespace ninfer::ops {
namespace {
struct Tail {
    __nv_bfloat16* data;
    const std::int64_t* offsets;
    std::int64_t column, layer;
};

struct Raw {
    const __nv_bfloat16* data;
    std::int64_t column, row, layer;
};

struct Extract {
    Tail before[2], after[2];
    Raw records[2];
    const int* accepted;
    const int* base;
    int dimension;
};

template <class Field>
Tail tail(const Field& f) {
    return {reinterpret_cast<__nv_bfloat16*>(
                const_cast<std::byte*>(static_cast<const std::byte*>(f.records.storage.data)) +
                f.field.tensor.region.offset),
            static_cast<const std::int64_t*>(f.records.byte_offsets.data),
            f.field.byte_strides[1] / 2, std::int64_t(f.field.layer_stride / 2)};
}

Raw raw(const Tensor& t) {
    return {static_cast<const __nv_bfloat16*>(t.data), t.nb[1] / 2, t.nb[2] / 2, t.nb[3] / 2};
}

__global__ void extract_kernel(Extract a) {
    const int b = blockIdx.y, l = blockIdx.x, accept = a.accepted[b];
    if (!accept) return;
    const int kind = threadIdx.x / 128, F = a.base[b], end = F + accept, rem = end % 4;
    const auto& src     = a.before[kind];
    const auto& dst     = a.after[kind];
    const auto& records = a.records[kind];
    for (int d = threadIdx.x % 128; d < a.dimension; d += 128) {
        __nv_bfloat16 words[3];
#pragma unroll
        for (int r = 0; r < 3; ++r) {
            const int p = end - rem + r;
            words[r]    = __float2bfloat16_rn(0.F);
            if (r < rem) {
                if (p < F)
                    words[r] = src.data[src.offsets[b] / 2 + l * src.layer +
                                        (p - (F - F % 4)) * src.column + d];
                else
                    words[r] = records.data[l * records.layer + b * records.row +
                                            (p - F) * records.column + d];
            }
        }
#pragma unroll
        for (int r = 0; r < 3; ++r)
            dst.data[dst.offsets[b] / 2 + l * dst.layer + r * dst.column + d] = words[r];
    }
}
} // namespace

void kpool_tail_extract(ConstKpoolTailStateView before, const KpoolRawRecordReadView& records,
                        const Tensor& accepted, KpoolTailStateView after,
                        CudaExecutionView execution) {
    using namespace detail::kpool;
    const int B = accepted.ne[0], R = records.raw_keys.ne[2], L = records.raw_keys.ne[3],
              W = records.raw_keys.ne[1], D = records.raw_keys.ne[0];
    require(B >= 1 && B <= R && R <= 8 && L > 0, "extraction geometry");
    const auto device = execution.device();
    operand::tensor(accepted, DType::I32, {B}, 4, kOperation, "accepted columns", device);
    operand::tensor(records.base_positions, DType::I32, {R}, 4, kOperation, "record bases", device);
    operand::tensor(records.valid_columns, DType::I32, {R}, 4, kOperation, "record lengths",
                    device);
    strided(records.raw_keys, DType::BF16, {D, W, R, L}, device, "raw keys");
    strided(records.raw_gates, DType::BF16, {D, W, R, L}, device, "raw gates");
    for (const auto* f : {&before.raw_keys, &before.raw_gates}) tail_field(*f, D, B, L, device);
    for (const auto* f : {&after.raw_keys, &after.raw_gates}) tail_field(*f, D, B, L, device);
    Extract a{{tail(before.raw_keys), tail(before.raw_gates)},
              {tail(after.raw_keys), tail(after.raw_gates)},
              {raw(records.raw_keys), raw(records.raw_gates)},
              static_cast<const int*>(accepted.data),
              static_cast<const int*>(records.base_positions.data),
              D};
    extract_kernel<<<dim3(L, B), 256, 0, execution.stream()>>>(a);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops
