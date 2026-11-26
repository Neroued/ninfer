#include "models/inkling2/state/context.h"

#include "models/inkling2/execution/profile.h"

#include <stdexcept>

namespace ninfer::models::inkling2 {

StateLayout plan_state(const execution::ModelProfile& profile, KvCacheStorage swa_storage) {
    const auto& c = profile.config.text;
    const auto conv_hist = dimension(c.sconv_kernel_size - 1U);
    StateImageBuilder b;
    StateLayout out;
    if (c.swa_layers) {
        const auto kv = dimension(c.swa_attention.kv_width());
        out.swa_k_conv = b.add(DType::BF16, {kv, conv_hist}, c.swa_layers, "SWA K ShortConv state");
        out.swa_v_conv = b.add(DType::BF16, {kv, conv_hist}, c.swa_layers, "SWA V ShortConv state");
    }
    if (c.full_layers) {
        const auto kv = dimension(c.full_attention.kv_width());
        out.full_k_conv =
            b.add(DType::BF16, {kv, conv_hist}, c.full_layers, "Full K ShortConv state");
        out.full_v_conv =
            b.add(DType::BF16, {kv, conv_hist}, c.full_layers, "Full V ShortConv state");
    }
    const auto logical = c.logical_layer_count();
    out.attn_sconv =
        b.add(DType::BF16, {dimension(c.hidden), conv_hist}, logical, "Attention ShortConv state");
    out.mlp_sconv =
        b.add(DType::BF16, {dimension(c.hidden), conv_hist}, logical, "MLP ShortConv state");
    out.hidden = b.add(DType::BF16, {dimension(c.hidden)}, 1, "Main final hidden");
    if (c.swa_layers) {
        out.swa = plan_cyclic_kv_cache(b, c.swa_layers, c.sliding_window,
                                       dimension(c.swa_attention.kv_heads),
                                       dimension(c.swa_attention.head_dim), swa_storage);
        out.persistent_copy = out.swa->group;
    }
    out.image = b.finish();
    return out;
}

StatePlacement describe_state(const StateLayout& s) {
    return {{s.image.image_bytes, s.image.alignment}, s.persistent_copy};
}

CacheLayout plan_cache(LayoutBuilder& builder, const AttentionGeometry& attn, std::uint32_t layers,
                       std::uint32_t max_context, std::uint32_t rows, KvCacheStorage storage) {
    if (!layers) throw std::invalid_argument("Inkling2 PagedKV cache requires at least one layer");
    const auto head_dim = dimension(attn.head_dim);
    const auto kv_heads = dimension(attn.kv_heads);
    const auto schema   = paged_kv_storage_layout(storage, head_dim);
    KVPageGeometry geometry;
    for (std::uint32_t i = 0; i < layers; ++i) {
        geometry.planes.push_back(
            {schema.key.data_dtype, schema.key.data_leading_extent, kv_heads, 256, 1});
        geometry.planes.push_back(
            {schema.value.data_dtype, schema.value.data_leading_extent, kv_heads, 256, 1});
        if (schema.key.has_scale()) {
            geometry.planes.push_back(
                {schema.key.scale_dtype, schema.key.scale_leading_extent, kv_heads, 256, 1});
        }
        if (schema.value.has_scale()) {
            geometry.planes.push_back(
                {schema.value.scale_dtype, schema.value.scale_leading_extent, kv_heads, 256, 1});
        }
    }
    const auto page   = plan_kv_page_layout(geometry);
    const auto tokens = page.geometry.page_tokens;
    return {page,
            plan_kv_execution_tables(builder, {(max_context - 1U) / tokens + 1U, dimension(rows)}),
            schema,
            layers,
            head_dim,
            kv_heads};
}

StateViews::StateViews(MemorySpan backing, const StateLayout& layout)
    : layout_(layout), image_(backing, layout.image) {
    if (layout_.swa) swa_.emplace(image_, *layout_.swa);
}

StateFieldView StateViews::k_conv(bool is_swa, std::uint32_t slot, const Tensor& offsets) const {
    const auto field = is_swa ? layout_.swa_k_conv : layout_.full_k_conv;
    return image_.field(field.value(), offsets, slot);
}

StateFieldView StateViews::v_conv(bool is_swa, std::uint32_t slot, const Tensor& offsets) const {
    const auto field = is_swa ? layout_.swa_v_conv : layout_.full_v_conv;
    return image_.field(field.value(), offsets, slot);
}

StateFieldView StateViews::attn_sconv(std::uint32_t logical_layer, const Tensor& offsets) const {
    return image_.field(layout_.attn_sconv, offsets, logical_layer);
}

StateFieldView StateViews::mlp_sconv(std::uint32_t logical_layer, const Tensor& offsets) const {
    return image_.field(layout_.mlp_sconv, offsets, logical_layer);
}

StateFieldView StateViews::hidden(const Tensor& offsets) const {
    return image_.field(layout_.hidden, offsets);
}

Tensor StateViews::hidden(std::int64_t offset) const {
    return image_.tensor(layout_.hidden, offset);
}

CyclicKVCacheLayerView StateViews::swa_layer(std::uint32_t slot, const Tensor& offsets) const {
    return swa_.value().layer_view(slot, offsets);
}

CacheViews::CacheViews(MemorySpan context, MemorySpan fixed, const CacheLayout& layout)
    : layout_(layout), context_(context),
      tables_(fixed, layout.tables, context, layout.page.page_bytes) {}

PagedKVBatchLayerView CacheViews::layer(std::uint32_t layer) const {
    if (layer >= layout_.layers) throw std::out_of_range("Inkling2 PagedKV cache layer");
    const auto& s   = layout_.storage;
    const auto base = layer * s.planes_per_layer();
    const PagedRecordView records{context_, tables_.matrix(), layout_.page.geometry.page_tokens};
    const auto plane = [&](std::uint32_t index) {
        return PagedPlaneView{records, layout_.page.planes.at(index)};
    };
    return {plane(base),
            plane(base + 1U),
            s.key.has_scale() ? plane(base + 2U) : PagedPlaneView{},
            s.value.has_scale() ? plane(base + 2U + std::uint32_t(s.key.has_scale()))
                                : PagedPlaneView{},
            layout_.head_dim,
            layout_.kv_heads,
            s.storage};
}

std::vector<context::RecordRangeCopy> state_conversion(std::span<const StatePlacement> states,
                                                       std::uint32_t from, std::uint32_t to) {
    if (from >= states.size() || to >= states.size())
        throw std::out_of_range("Inkling2 State conversion owner is outside its rank group");
    return {{from, to, 0, 0, states[from].image.bytes}};
}

context::HistoryCopyLayout history_conversion(const KVPageLayout& cache, std::uint32_t from,
                                              std::uint32_t to, std::uint32_t tail) {
    context::HistoryCopyLayout out;
    out.full_page.push_back({from, to, 0, 0, cache.page_bytes});
    out.partial_columns = tail;
    for (const auto& plane : cache.planes) {
        const auto bytes = (tail / plane.geometry.tokens_per_row) * plane.row_stride_bytes();
        if (!bytes) continue;
        for (std::int32_t head = 0; head < plane.geometry.head_extent; ++head) {
            const auto offset = plane.tensor.region.offset + head * plane.head_stride_bytes;
            out.partial_page.push_back({from, to, offset, offset, bytes});
        }
    }
    return out;
}

} // namespace ninfer::models::inkling2
