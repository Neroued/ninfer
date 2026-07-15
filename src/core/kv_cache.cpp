#include "core/kv_cache.h"

#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer {
namespace {

constexpr std::size_t kArenaAlign = 256;

void validate_layer(const KVCache& cache, std::uint32_t layer) {
    if (layer >= cache.layer_count()) { throw std::out_of_range("KVCache layer out of range"); }
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
                            std::int32_t head_dim_in, DType dtype_in, std::int32_t quant_group_in) {
    if (full_layers == 0) { throw std::invalid_argument("KVCache layers must be nonzero"); }
    if (max_context_in == 0 ||
        max_context_in > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
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
    const auto add    = [&](const std::vector<LayoutRegion>& regions) {
        for (const auto& region : regions) { total += region.bytes; }
    };
    add(k);
    add(v);
    add(k_scale);
    add(v_scale);
    return total;
}

KVCache::KVCache(DeviceSpan backing, const KVCacheLayout& layout)
    : max_context(layout.max_context), padded_context(layout.padded_context),
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
        k.push_back(bind_tensor(backing, layout.k[layer], dtype, {head_dim, padded, num_kv_heads}));
        v.push_back(bind_tensor(backing, layout.v[layer], dtype, {head_dim, padded, num_kv_heads}));
        if (dtype == DType::I8) {
            k_scale.push_back(bind_tensor(backing, layout.k_scale[layer], DType::FP16,
                                          {groups, padded, num_kv_heads}));
            v_scale.push_back(bind_tensor(backing, layout.v_scale[layer], DType::FP16,
                                          {groups, padded, num_kv_heads}));
        }
    }
}

std::uint32_t KVCache::layer_count() const noexcept { return static_cast<std::uint32_t>(k.size()); }

KVCacheLayerView KVCache::layer_view(std::uint32_t layer) const {
    validate_layer(*this, layer);
    KVCacheLayerView view{
        .k              = k[layer],
        .v              = v[layer],
        .max_context    = max_context,
        .padded_context = padded_context,
        .num_kv_heads   = num_kv_heads,
        .head_dim       = head_dim,
        .dtype          = dtype,
        .quant_group    = quant_group,
    };
    if (dtype == DType::I8) {
        view.k_scale = k_scale[layer];
        view.v_scale = v_scale[layer];
    }
    return view;
}

} // namespace ninfer
