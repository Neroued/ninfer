#include "models/glm5next/state/context.h"

#include "models/glm5next/execution/profile.h"

namespace ninfer::models::glm5next {

StateLayout plan_state(const execution::ModelProfile& p) {
    StateImageBuilder b;
    const auto& c    = p.config.text;
    const auto heads = dimension(p.local.kda_heads.size), d = dimension(c.kda.head_dim);
    StateLayout out;
    out.kda    = plan_linear_attention_state(b, {c.kda_layers, dimension(3ULL * heads * d),
                                                 dimension(c.kda.conv_width - 1ULL), heads, d, d,
                                                 DType::BF16});
    out.hidden = b.add(DType::BF16, {dimension(c.hidden)}, 1, "Main final hidden");
    out.common = b.finish();
    out.main =
        ops::append_kpool_tail_fields(b, dimension(c.mla.index_dim), dimension(c.mla_layers));
    if (p.mtp) out.mtp = ops::append_kpool_tail_fields(b, dimension(c.mla.index_dim), 1);
    out.owner_fields =
        b.group(out.main.raw_keys, out.mtp ? out.mtp->raw_gates : out.main.raw_gates);
    out.owner = b.finish();
    return out;
}

StatePlacement describe_state(const StateLayout& s) {
    return {{s.common.image_bytes, s.common.alignment},
            {s.owner.image_bytes, s.owner.alignment},
            s.owner_fields};
}

ops::KpoolMlaStorage latent_storage(KvCacheStorage storage) {
    switch (storage) {
    case KvCacheStorage::BFloat16:
        return ops::KpoolMlaStorage::BFloat16;
    case KvCacheStorage::Fp8E4M3Row256:
        return ops::KpoolMlaStorage::Fp8E4m3fnGroup128F32;
    default:
        throw std::invalid_argument("GLM5Next has no latent representation for this KV format");
    }
}

CacheLayout plan_cache(LayoutBuilder& b, const MlaConfig& c, std::uint32_t layers,
                       std::uint32_t max_context, std::uint32_t rows,
                       ops::KpoolMlaStorage storage) {
    auto layout = ops::plan_kpool_mla_cache_layout(
        {storage, dimension(c.latent_dim), dimension(c.index_dim), dimension(layers)});
    const auto tokens = layout.page.geometry.page_tokens;
    return {std::move(layout),
            plan_kv_execution_tables(b, {(max_context - 1) / tokens + 1, dimension(rows)})};
}

StateViews::StateViews(MemorySpan backing, const StateLayout& layout)
    : layout_(layout), common_(backing, layout.common), owner_(backing, layout.owner),
      kda_(common_, layout.kda) {}

StateFieldView StateViews::hidden(const Tensor& offsets) const {
    return common_.field(layout_.hidden, offsets);
}

ops::KpoolTailStateView StateViews::main_tail(const Tensor& offsets) const {
    return {owner_.field(layout_.main.raw_keys, offsets),
            owner_.field(layout_.main.raw_gates, offsets)};
}

ops::KpoolTailStateView StateViews::mtp_tail(const Tensor& offsets) const {
    const auto& fields = layout_.mtp.value();
    return {owner_.field(fields.raw_keys, offsets), owner_.field(fields.raw_gates, offsets)};
}

const StateImageStorage& StateViews::image(bool owner) const noexcept {
    return owner ? owner_ : common_;
}

CacheViews::CacheViews(MemorySpan context, MemorySpan fixed, const CacheLayout& layout)
    : layout_(layout), context_(context),
      tables_(fixed, layout.tables, context, layout.cache.page.page_bytes) {}

ops::KpoolMlaCacheView CacheViews::layer(std::uint32_t layer) const {
    return layout_.cache.layer(
        PagedRecordView{context_, tables_.matrix(), layout_.cache.page.geometry.page_tokens},
        static_cast<std::int32_t>(layer));
}

std::vector<context::RecordRangeCopy> state_conversion(std::span<const StatePlacement> states,
                                                       std::uint32_t from, std::uint32_t to) {
    if (from >= states.size() || to >= states.size())
        throw std::out_of_range("GLM5Next State conversion owner is outside its rank group");
    std::vector<context::RecordRangeCopy> out;
    for (std::uint32_t rank = 0; rank < states.size(); ++rank)
        out.push_back({rank, rank, 0, 0, states[rank].common.bytes});
    const auto& source      = states[from].owner_fields;
    const auto& destination = states[to].owner_fields;
    if (source.bytes != destination.bytes)
        throw std::invalid_argument("GLM5Next owner tail layouts differ");
    out.push_back({from, to, source.offset, destination.offset, source.bytes});
    return out;
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

} // namespace ninfer::models::glm5next
