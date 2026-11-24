#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace ninfer::ops::detail::qsa {

template <class T>
struct PrepareColumns {
    T* data;
    std::int64_t column, outer;

    __device__ T* at(int feature, std::int64_t c, int row = 0) const {
        return data + feature + c * column + std::int64_t(row) * outer;
    }
};

template <class T>
struct PrepareState {
    const T* data;
    const std::int64_t* offsets;
    std::int64_t column;

    __device__ const T* at(int feature, int c, int row) const {
        const auto* base = reinterpret_cast<const std::uint8_t*>(data) + offsets[row];
        return reinterpret_cast<const T*>(base) + feature + c * column;
    }
};

template <class T>
struct PreparePage {
    T* data;
    std::int64_t row, head;

    __device__ T* at(int p, int h, std::int64_t offset) const {
        auto* base = reinterpret_cast<std::uint8_t*>(data) + offset;
        return reinterpret_cast<T*>(base) + std::int64_t(p) * row + std::int64_t(h) * head;
    }
};

struct PrepareArguments {
    const __nv_bfloat16 *q, *k, *v, *iq, *ik, *wq, *wk, *wiq, *wik;
    __nv_bfloat16 *out_q, *out_iq;
    const std::int32_t *positions, *rope, *valid, *table_rows, *offsets;
    const std::int64_t* tables;
    std::int64_t table_stride;
    PrepareState<__nv_bfloat16> tail;
    PrepareState<std::int32_t> first_rope;
    PrepareColumns<__nv_bfloat16> raw_records;
    PrepareColumns<std::int32_t> rope_records;
    PreparePage<__nv_bfloat16> index_keys;
    PreparePage<std::uint8_t> keys, values, key_scales, value_scales;
    int tokens, batch, axes, kv_heads;
    float epsilon;
    double frequency[32];
};
} // namespace ninfer::ops::detail::qsa
