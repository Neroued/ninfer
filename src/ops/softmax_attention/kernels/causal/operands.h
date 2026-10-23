#pragma once
#include "core/arena.h"
#include "core/paged_kv_cache.h"
#include "ops/softmax_attention/kernels/causal/geometry.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <type_traits>
#include <stdexcept>

namespace ninfer::ops::detail {
struct CausalAttentionOperands {
    const __nv_bfloat16* q;
    const std::int32_t* positions;
    __nv_bfloat16* out;
    float scale;
    int width;
    int batch;
    int visible_capacity;
    int head_dim;
    int query_heads;
};

struct CausalAppendInput {
    static constexpr bool writes_cache = true;
    const __nv_bfloat16* k;
    const __nv_bfloat16* v;
};

struct CausalCachedInput {
    static constexpr bool writes_cache = false;
};

// Unnormalized FP32 numerator and sum. Maximum units belong to the owning
// mainloop/merge: log2 for BF16, natural scaled-score units for quantized KV.
struct CausalPartialView {
    float* acc;
    float* maximum;
    float* sum;
};

struct CausalPartialStorage {
    Tensor acc, maximum, sum;

    CausalPartialView view() const {
        return {static_cast<float*>(acc.data), static_cast<float*>(maximum.data),
                static_cast<float*>(sum.data)};
    }
};

inline CausalAttentionOperands make_causal_operands(const Tensor& q, const Tensor& positions,
                                                    Tensor& out, float scale, int capacity) {
    return {static_cast<const __nv_bfloat16*>(q.data),
            static_cast<const std::int32_t*>(positions.data),
            static_cast<__nv_bfloat16*>(out.data),
            scale,
            q.ne[2],
            q.ne[3],
            capacity,
            q.ne[0],
            q.ne[1]};
}

template <class Allocator>
CausalPartialStorage allocate_causal_partials(Allocator& allocator, int heads, int width,
                                              int splits, int batch) {
    return {allocator.alloc(DType::FP32, {kCausalHeadDim, heads, width, splits * batch}),
            allocator.alloc(DType::FP32, {heads, width, splits * batch}),
            allocator.alloc(DType::FP32, {heads, width, splits * batch})};
}

template <class CodeType, class KeyScaleType, class ValueScaleType, bool Writable>
struct QuantizedCausalCacheView {
    using Code       = std::conditional_t<Writable, CodeType, const CodeType>;
    using KeyScale   = std::conditional_t<Writable, KeyScaleType, const KeyScaleType>;
    using ValueScale = std::conditional_t<Writable, ValueScaleType, const ValueScaleType>;
    Code* keys;
    Code* values;
    KeyScale* key_scales;
    ValueScale* value_scales;
    const std::int64_t* tables;
    const std::int32_t* valid_columns;
    const std::int32_t* table_rows;
    int table_stride, kv_heads;
};

template <class View, class Cache>
View make_quantized_causal_cache_view(const Cache& cache, const Tensor* valid = nullptr,
                                      const Tensor* rows = nullptr) {
    return {static_cast<typename View::Code*>(cache.k_pages.data()),
            static_cast<typename View::Code*>(cache.v_pages.data()),
            static_cast<typename View::KeyScale*>(cache.k_scale_pages.data()),
            static_cast<typename View::ValueScale*>(cache.v_scale_pages.data()),
            static_cast<const std::int64_t*>(cache.k_pages.records.page_offsets.data),
            valid ? static_cast<const std::int32_t*>(valid->data) : nullptr,
            rows ? static_cast<const std::int32_t*>(rows->data) : nullptr,
            cache.k_pages.records.page_offsets.ne[0],
            cache.num_kv_heads};
}

template <class G, class View>
void validate_quantized_causal_operands(const CausalAttentionOperands& p, View cache) {
    static_assert(G::kHeadDim == kCausalHeadDim, "Quantized causal templates require D256");
    if (p.query_heads != G::QHeads || cache.kv_heads != G::KVHeads || !p.q || !p.positions ||
        !p.out || !cache.keys || !cache.values || !cache.key_scales || !cache.value_scales ||
        !cache.tables || p.width < 1 || p.batch < 1 || p.visible_capacity < 1 ||
        static_cast<std::int64_t>(p.visible_capacity) >
            static_cast<std::int64_t>(cache.table_stride) * kPagedKVPageSize)
        throw std::invalid_argument("quantized causal attention: invalid operands");
}

template <bool Writable>
struct Bf16KvCacheView {
    using Key   = std::conditional_t<Writable, __nv_bfloat16, const __nv_bfloat16>;
    using Value = std::conditional_t<Writable, __half, const __half>;
    Key* keys;
    Value* values;
    const std::int64_t* tables;
    const std::int32_t* valid_columns;
    const std::int32_t* table_rows;
    int table_stride;
    int head_dim;
    int kv_heads;
};

using Bf16KvReadView = Bf16KvCacheView<false>;

template <bool Writable, class Cache>
Bf16KvCacheView<Writable> bf16_kv_cache_view(const Cache& cache, const Tensor* valid = nullptr,
                                             const Tensor* rows = nullptr) {
    return {static_cast<typename Bf16KvCacheView<Writable>::Key*>(cache.k_pages.data()),
            static_cast<typename Bf16KvCacheView<Writable>::Value*>(cache.v_pages.data()),
            static_cast<const std::int64_t*>(cache.k_pages.records.page_offsets.data),
            valid ? static_cast<const std::int32_t*>(valid->data) : nullptr,
            rows ? static_cast<const std::int32_t*>(rows->data) : nullptr,
            cache.k_pages.records.page_offsets.ne[0],
            cache.head_dim,
            cache.num_kv_heads};
}

template <bool Writable>
using Fp8KvCacheView = QuantizedCausalCacheView<std::uint8_t, __half, __half, Writable>;
using Fp8KvReadView  = Fp8KvCacheView<false>;

template <bool Writable>
using Int8KvCacheView = QuantizedCausalCacheView<std::int8_t, __half, __half, Writable>;
using Int8KvReadView  = Int8KvCacheView<false>;

template <bool Writable>
using Nvfp4KvCacheView =
    QuantizedCausalCacheView<std::uint8_t, std::uint8_t, std::uint8_t, Writable>;
using Nvfp4KvReadView = Nvfp4KvCacheView<false>;

template <bool Writable>
using K8V4KvCacheView = QuantizedCausalCacheView<std::uint8_t, __half, std::uint8_t, Writable>;
using K8V4KvReadView  = K8V4KvCacheView<false>;

} // namespace ninfer::ops::detail
