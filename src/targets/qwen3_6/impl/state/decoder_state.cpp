#include <ninfer/targets/qwen3_6/decoder_state.h>

#include <limits>
#include <stdexcept>

namespace ninfer::targets::qwen3_6 {
namespace {

std::uint32_t page_count(std::uint32_t capacity) {
    if (capacity == 0) { throw std::invalid_argument("Paged KV capacity must be positive"); }
    return 1U + (capacity - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

PagedKVCacheLayout plan_cache(LayoutBuilder& builder, std::uint32_t layers,
                              std::uint32_t capacity, std::int32_t kv_heads, std::int32_t head_dim,
                              DType dtype, std::int32_t quant_group, std::int32_t table_rows,
                              std::uint32_t physical_page_groups, bool packed_v, bool rotate_k,
                              bool rotate_v, bool packed_k, bool e8_lattice, bool e8_root) {
    if (layers == 0 || capacity == 0 || kv_heads <= 0 || head_dim <= 0 || table_rows <= 0) {
        throw std::invalid_argument("Paged KV cache dimensions must be positive");
    }
    const bool quantized = dtype == DType::I8;
    if ((!quantized && (dtype != DType::BF16 || quant_group != 0)) ||
        (quantized && (quant_group != kKvQuantGroup || head_dim % quant_group != 0))) {
        throw std::invalid_argument("Paged KV cache dtype or quantization is invalid");
    }
    if ((packed_v || rotate_k || rotate_v || packed_k || e8_lattice || e8_root) && !quantized) {
        throw std::invalid_argument("Packed or rotated KV requires quantized storage");
    }
    if (rotate_v && !packed_v) {
        throw std::invalid_argument("Rotated V requires packed V4 storage");
    }
    if (e8_lattice && !packed_k) {
        throw std::invalid_argument("E8 lattice requires packed K storage");
    }

    const std::uint32_t logical_pages = page_count(capacity);
    if (physical_page_groups < logical_pages) {
        throw std::invalid_argument("Paged KV physical pages are below logical capacity");
    }

    PagedKVPoolSpec pool_spec;
    pool_spec.page_group_count      = physical_page_groups;
    pool_spec.logical_page_capacity = logical_pages;
    pool_spec.table_rows            = table_rows;
    pool_spec.planes.reserve(static_cast<std::size_t>(layers) * (quantized ? 4ULL : 2ULL));
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        const std::int32_t k_head_extent = e8_root ? head_dim / 4 : (packed_k ? head_dim / 2 : head_dim);
        const std::int32_t v_head_extent = packed_v ? head_dim / 2 : head_dim;
        const DType k_plane_dtype = (packed_k || e8_root) ? DType::U8 : dtype;
        const DType v_plane_dtype = packed_v ? DType::U8 : dtype;
        pool_spec.planes.push_back({k_plane_dtype, k_head_extent, kv_heads, 256});
        pool_spec.planes.push_back({v_plane_dtype, v_head_extent, kv_heads, 256});
        if (quantized) {
            const std::int32_t scale_extent = head_dim / quant_group;
            pool_spec.planes.push_back({DType::FP16, scale_extent, kv_heads, 256});
            pool_spec.planes.push_back({DType::FP16, scale_extent, kv_heads, 256});
        }
    }
    return PagedKVCacheLayout{
        .pool        = plan_paged_kv_pool(builder, pool_spec),
        .layers      = layers,
        .max_context = capacity,
        .kv_heads    = kv_heads,
        .head_dim    = head_dim,
        .dtype       = dtype,
        .quant_group = quant_group,
        .packed_v    = packed_v,
        .rotate_k    = rotate_k,
        .rotate_v    = rotate_v,
        .packed_k    = packed_k,
        .e8_lattice  = e8_lattice,
        .e8_root     = e8_root,
    };
}

} // namespace

DecoderStateLayout plan_decoder_state(LayoutBuilder& builder, const DecoderStateSpec& spec) {
    DecoderStateLayout layout;
    layout.text_kv = plan_cache(builder, spec.full_attention_layers, spec.capacity, spec.kv_heads,
                                spec.attention_head_dim, spec.kv_dtype, spec.kv_quant_group,
                                spec.kv_table_rows, spec.text_physical_page_groups,
                                spec.kv_packed_v, spec.kv_rotate_k, spec.kv_rotate_v,
                                spec.kv_packed_k, spec.kv_e8_lattice, spec.kv_e8_root);
    if (spec.enable_mtp) {
        layout.mtp_kv = plan_cache(builder, spec.mtp_layers, spec.capacity, spec.kv_heads,
                                   spec.attention_head_dim, spec.kv_dtype, spec.kv_quant_group,
                                   spec.kv_table_rows, spec.mtp_physical_page_groups,
                                   spec.kv_packed_v, spec.kv_rotate_k, spec.kv_rotate_v,
                                   spec.kv_packed_k, spec.kv_e8_lattice, spec.kv_e8_root);
    }
    layout.linear_attention = plan_linear_attention_state_pool(builder, spec.linear_attention);
    return layout;
}

PagedKVCache::PagedKVCache(DeviceSpan backing, const PagedKVCacheLayout& layout)
    : pool_(backing, layout.pool), layers_(layout.layers), max_context_(layout.max_context),
      kv_heads_(layout.kv_heads), head_dim_(layout.head_dim), dtype_(layout.dtype),
      quant_group_(layout.quant_group), packed_v_(layout.packed_v), rotate_k_(layout.rotate_k),
      rotate_v_(layout.rotate_v), packed_k_(layout.packed_k), e8_lattice_(layout.e8_lattice),
      e8_root_(layout.e8_root) {}

PagedKVCacheView::PagedKVCacheView(const PagedKVCache& cache, Tensor block_table) noexcept
    : cache_(&cache), block_table_(block_table) {}

std::uint32_t PagedKVCacheView::max_context() const noexcept {
    return cache_ == nullptr ? 0 : cache_->max_context();
}

PagedKVLayerView PagedKVCacheView::layer_view(std::uint32_t layer) const {
    if (cache_ == nullptr) { throw std::logic_error("Paged KV execution view is empty"); }
    return cache_->layer_view(layer, block_table_);
}

PagedKVCacheView PagedKVCache::execution_view(const PagedKVAllocation& allocation) const {
    if (!allocation.belongs_to(pool_)) {
        throw std::invalid_argument("Paged KV allocation belongs to another cache pool");
    }
    return PagedKVCacheView(*this, allocation.block_table());
}

PagedKVLayerView PagedKVCache::layer_view(std::uint32_t layer, Tensor block_table) const {
    if (layer >= layers_) { throw std::out_of_range("Paged KV layer is out of range"); }
    const bool quantized     = dtype_ == DType::I8;
    const std::size_t stride = quantized ? 4ULL : 2ULL;
    const std::size_t base   = static_cast<std::size_t>(layer) * stride;
    return PagedKVLayerView{
        .k_pages       = pool_.plane(base),
        .v_pages       = pool_.plane(base + 1),
        .k_scale_pages = quantized ? pool_.plane(base + 2) : Tensor(),
        .v_scale_pages = quantized ? pool_.plane(base + 3) : Tensor(),
        .block_table   = block_table,
        .head_dim      = head_dim_,
        .num_kv_heads  = kv_heads_,
        .dtype         = dtype_,
        .quant_group   = quant_group_,
        .packed_v      = packed_v_,
        .rotate_k      = rotate_k_,
        .rotate_v      = rotate_v_,
        .packed_k      = packed_k_,
        .e8_lattice    = e8_lattice_,
        .e8_root       = e8_root_,
    };
}

PagedKVBatchLayerView PagedKVCache::batch_layer_view(std::uint32_t layer) const {
    if (layer >= layers_) { throw std::out_of_range("Paged KV layer is out of range"); }
    const bool quantized     = dtype_ == DType::I8;
    const std::size_t stride = quantized ? 4ULL : 2ULL;
    const std::size_t base   = static_cast<std::size_t>(layer) * stride;
    return PagedKVBatchLayerView{
        .k_pages       = pool_.plane(base),
        .v_pages       = pool_.plane(base + 1),
        .k_scale_pages = quantized ? pool_.plane(base + 2) : Tensor(),
        .v_scale_pages = quantized ? pool_.plane(base + 3) : Tensor(),
        .block_tables  = pool_.block_tables(),
        .head_dim      = head_dim_,
        .num_kv_heads  = kv_heads_,
        .dtype         = dtype_,
        .quant_group   = quant_group_,
        .packed_v      = packed_v_,
        .rotate_k      = rotate_k_,
        .rotate_v      = rotate_v_,
        .packed_k      = packed_k_,
        .e8_lattice    = e8_lattice_,
        .e8_root       = e8_root_,
    };
}

std::size_t DecoderStateLayout::kv_payload_bytes() const noexcept {
    return text_kv.payload_bytes() + (mtp_kv ? mtp_kv->payload_bytes() : 0);
}

DecoderState::DecoderState(DeviceSpan backing, const DecoderStateLayout& layout)
    : text_kv(backing, layout.text_kv), linear_attention(backing, layout.linear_attention) {
    if (layout.mtp_kv) { mtp_kv.emplace(backing, *layout.mtp_kv); }
}

PagedKVCache* DecoderState::mtp_cache() noexcept { return mtp_kv ? &*mtp_kv : nullptr; }

const PagedKVCache* DecoderState::mtp_cache() const noexcept { return mtp_kv ? &*mtp_kv : nullptr; }

} // namespace ninfer::targets::qwen3_6
