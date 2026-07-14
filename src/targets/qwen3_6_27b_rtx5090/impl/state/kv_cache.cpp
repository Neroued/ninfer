#include "targets/qwen3_6_27b_rtx5090/impl/state/kv_cache.h"

#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::targets::qwen3_6_27b_rtx5090::detail {
namespace {

constexpr std::size_t kArenaAlign = 256;

void validate_layer(const KVCache& cache, std::uint32_t layer) {
    if (layer >= cache.layer_count()) { throw std::out_of_range("KVCache layer out of range"); }
}

void validate_kv_head(const KVCache& cache, std::int32_t kv_head) {
    if (kv_head < 0 || kv_head >= cache.num_kv_heads) {
        throw std::out_of_range("KVCache kv_head out of range");
    }
}

std::uint32_t align_up_u32(std::uint32_t value, std::uint32_t alignment) {
    const std::uint64_t mask    = static_cast<std::uint64_t>(alignment) - 1U;
    const std::uint64_t aligned = (static_cast<std::uint64_t>(value) + mask) & ~mask;
    if (aligned > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("KVCache padded_context out of range");
    }
    return static_cast<std::uint32_t>(aligned);
}

std::int32_t normalize_quant_group(DType dtype, std::int32_t quant_group, std::int32_t head_dim) {
    if (dtype == DType::BF16) {
        if (quant_group != 0) {
            throw std::invalid_argument("KVCache BF16 mode must not set quant_group");
        }
        return 0;
    }
    if (dtype != DType::I8) { throw std::invalid_argument("KVCache dtype must be BF16 or I8"); }

    const std::int32_t group = quant_group == 0 ? kKvQuantGroup : quant_group;
    if (group != kKvQuantGroup) {
        throw std::invalid_argument("KVCache I8 mode requires quant_group 64");
    }
    if (head_dim % group != 0) {
        throw std::invalid_argument("KVCache head_dim must be divisible by quant_group");
    }
    return group;
}

std::int32_t scale_groups(const KVCache& cache) {
    if (cache.quant_group <= 0) { return 0; }
    return cache.head_dim / cache.quant_group;
}

KVHeadSlot slot_at(const KVCache& cache, std::uint32_t layer, std::uint32_t position,
                   std::int32_t kv_head) {
    validate_layer(cache, layer);
    validate_kv_head(cache, kv_head);
    if (position >= cache.max_context ||
        position > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::out_of_range("KVCache position out of range");
    }
    const std::int64_t element_offset = static_cast<std::int64_t>(cache.head_dim) *
                                        (static_cast<std::int64_t>(position) +
                                         static_cast<std::int64_t>(cache.padded_context) * kv_head);
    const std::int64_t byte_offset =
        element_offset * static_cast<std::int64_t>(dtype_size(cache.dtype));

    auto* k_ptr = static_cast<unsigned char*>(cache.k[layer].data) + byte_offset;
    auto* v_ptr = static_cast<unsigned char*>(cache.v[layer].data) + byte_offset;
    KVHeadSlot slot{Tensor(k_ptr, cache.dtype, {cache.head_dim}),
                    Tensor(v_ptr, cache.dtype, {cache.head_dim}),
                    {},
                    {}};
    if (cache.dtype == DType::I8) {
        const std::int32_t groups = scale_groups(cache);
        const std::int64_t scale_element_offset =
            static_cast<std::int64_t>(groups) *
            (static_cast<std::int64_t>(position) +
             static_cast<std::int64_t>(cache.padded_context) * kv_head);
        const std::int64_t scale_byte_offset =
            scale_element_offset * static_cast<std::int64_t>(dtype_size(DType::FP16));
        auto* k_scale_ptr =
            static_cast<unsigned char*>(cache.k_scale[layer].data) + scale_byte_offset;
        auto* v_scale_ptr =
            static_cast<unsigned char*>(cache.v_scale[layer].data) + scale_byte_offset;
        slot.k_scale = Tensor(k_scale_ptr, DType::FP16, {groups});
        slot.v_scale = Tensor(v_scale_ptr, DType::FP16, {groups});
    }
    return slot;
}

Tensor bind_tensor(DeviceSpan backing, const LayoutRegion& region, DType dtype,
                   std::initializer_list<std::int32_t> shape) {
    Tensor expected(nullptr, dtype, shape);
    if (expected.bytes() != region.bytes) {
        throw std::logic_error("KVCache layout tensor byte size is inconsistent");
    }
    return Tensor(region.bind(backing).data, dtype, shape);
}

} // namespace

KVCacheLayout plan_kv_cache(LayoutBuilder& builder, std::uint32_t full_layers,
                            std::uint32_t max_context_in, std::int32_t num_kv_heads_in,
                            std::int32_t head_dim_in, DType dtype_in,
                            std::int32_t quant_group_in) {
    if (full_layers == 0) { throw std::invalid_argument("KVCache layers must be nonzero"); }
    if (max_context_in == 0 || max_context_in >
                                   static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("KVCache max_context out of range");
    }
    if (num_kv_heads_in <= 0) {
        throw std::invalid_argument("KVCache num_kv_heads must be positive");
    }
    if (head_dim_in <= 0) { throw std::invalid_argument("KVCache head_dim must be positive"); }
    KVCacheLayout layout;
    layout.max_context    = max_context_in;
    layout.padded_context = align_up_u32(max_context_in, 128);
    layout.num_kv_heads   = num_kv_heads_in;
    layout.head_dim       = head_dim_in;
    layout.dtype          = dtype_in;
    layout.quant_group    = normalize_quant_group(dtype_in, quant_group_in, head_dim_in);

    const auto padded_context_i32 = static_cast<std::int32_t>(layout.padded_context);
    const Tensor code_shape(nullptr, dtype_in, {head_dim_in, padded_context_i32, num_kv_heads_in});
    const bool quantized      = dtype_in == DType::I8;
    const std::int32_t groups = quantized ? head_dim_in / layout.quant_group : 0;
    const std::size_t scale_bytes =
        quantized
            ? Tensor(nullptr, DType::FP16, {groups, padded_context_i32, num_kv_heads_in}).bytes()
            : 0;
    layout.k.reserve(full_layers);
    layout.v.reserve(full_layers);
    if (quantized) {
        layout.k_scale.reserve(full_layers);
        layout.v_scale.reserve(full_layers);
    }
    for (std::uint32_t layer = 0; layer < full_layers; ++layer) {
        const std::string prefix = "KV layer " + std::to_string(layer);
        layout.k.push_back(builder.add(code_shape.bytes(), kArenaAlign, prefix + " K"));
        layout.v.push_back(builder.add(code_shape.bytes(), kArenaAlign, prefix + " V"));
        if (quantized) {
            layout.k_scale.push_back(builder.add(scale_bytes, kArenaAlign, prefix + " K scale"));
            layout.v_scale.push_back(builder.add(scale_bytes, kArenaAlign, prefix + " V scale"));
        }
    }
    return layout;
}

std::size_t KVCacheLayout::payload_bytes() const noexcept {
    std::size_t total = 0;
    const auto add = [&](const std::vector<LayoutRegion>& regions) {
        for (const auto& region : regions) { total += region.bytes; }
    };
    add(k);
    add(v);
    add(k_scale);
    add(v_scale);
    return total;
}

KVCache::KVCache(DeviceSpan backing, const KVCacheLayout& layout)
    : pos(0), max_context(layout.max_context), padded_context(layout.padded_context),
      num_kv_heads(layout.num_kv_heads), head_dim(layout.head_dim), dtype(layout.dtype),
      quant_group(layout.quant_group) {
    const auto layers = layout.k.size();
    if (layers == 0 || layout.v.size() != layers ||
        (dtype == DType::I8 &&
         (layout.k_scale.size() != layers || layout.v_scale.size() != layers)) ||
        (dtype != DType::I8 && (!layout.k_scale.empty() || !layout.v_scale.empty()))) {
        throw std::invalid_argument("KVCache layout plane counts are inconsistent");
    }
    const auto padded = static_cast<std::int32_t>(padded_context);
    const auto groups = dtype == DType::I8 ? head_dim / quant_group : 0;
    k.reserve(layers);
    v.reserve(layers);
    k_scale.reserve(layout.k_scale.size());
    v_scale.reserve(layout.v_scale.size());
    for (std::size_t layer = 0; layer < layers; ++layer) {
        k.push_back(bind_tensor(backing, layout.k[layer], dtype,
                                {head_dim, padded, num_kv_heads}));
        v.push_back(bind_tensor(backing, layout.v[layer], dtype,
                                {head_dim, padded, num_kv_heads}));
        if (dtype == DType::I8) {
            k_scale.push_back(bind_tensor(backing, layout.k_scale[layer], DType::FP16,
                                          {groups, padded, num_kv_heads}));
            v_scale.push_back(bind_tensor(backing, layout.v_scale[layer], DType::FP16,
                                          {groups, padded, num_kv_heads}));
        }
    }
}

std::uint32_t KVCache::layer_count() const noexcept { return static_cast<std::uint32_t>(k.size()); }

KVHeadSlot KVCache::slot(std::uint32_t layer, std::uint32_t position, std::int32_t kv_head) const {
    if (position >= pos) { throw std::out_of_range("KVCache read position has not been written"); }
    return slot_at(*this, layer, position, kv_head);
}

KVHeadSlot KVCache::append_slot(std::uint32_t layer, std::int32_t kv_head) const {
    if (pos >= max_context) { throw std::out_of_range("KVCache append position is full"); }
    return slot_at(*this, layer, pos, kv_head);
}

void KVCache::advance() {
    if (pos >= max_context) { throw std::out_of_range("KVCache position is full"); }
    ++pos;
}

void KVCache::rewind(std::uint32_t position) {
    if (position > pos) { throw std::out_of_range("KVCache rewind cannot move forward"); }
    pos = position;
}

void KVCache::reset() noexcept { pos = 0; }

} // namespace ninfer::targets::qwen3_6_27b_rtx5090::detail
