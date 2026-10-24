#include "ninfer/ops/kpool_mla_layout.h"

#include <stdexcept>

namespace ninfer::ops {
namespace {
int planes_per_layer(KpoolMlaStorage storage) {
    switch (storage) {
    case KpoolMlaStorage::BFloat16:
        return 2;
    case KpoolMlaStorage::Fp8E4m3fnGroup128F32:
        return 3;
    }
    throw std::invalid_argument("Kpool cache: unknown storage");
}

void check_layer(const KpoolMlaCacheLayout& layout, int layer, std::uint32_t page_tokens) {
    if (layer < 0 || layer >= layout.spec.layers || page_tokens != 64 ||
        layout.page.geometry.page_tokens != 64 ||
        layout.page.planes.size() !=
            std::size_t(layout.spec.layers) * planes_per_layer(layout.spec.storage))
        throw std::invalid_argument("Kpool cache: invalid layer binding");
}
} // namespace

KpoolMlaCacheLayout plan_kpool_mla_cache_layout(const KpoolMlaCacheSpec& spec) {
    const int count = planes_per_layer(spec.storage);
    if (spec.latent_dim <= 0 || spec.index_dim <= 0 || spec.layers <= 0 ||
        (count == 3 && spec.latent_dim % 128))
        throw std::invalid_argument("Kpool cache: invalid geometry");
    KVPageGeometry geometry{64, {}};
    for (int layer = 0; layer < spec.layers; ++layer) {
        geometry.planes.push_back(
            {count == 2 ? DType::BF16 : DType::FP8_E4M3FN, spec.latent_dim, 1, 256, 1});
        if (count == 3) geometry.planes.push_back({DType::FP32, spec.latent_dim / 128, 1, 256, 1});
        geometry.planes.push_back({DType::BF16, spec.index_dim, 1, 256, 4});
    }
    return {spec, plan_kv_page_layout(geometry)};
}

KpoolMlaCacheView KpoolMlaCacheLayout::layer(PagedRecordView records, int layer) const {
    check_layer(*this, layer, records.page_tokens);
    const int count  = planes_per_layer(spec.storage);
    const auto first = std::size_t(count) * layer;
    return {{{records, page.planes[first]},
             count == 3 ? PagedPlaneView{records, page.planes[first + 1]} : PagedPlaneView{},
             spec.storage},
            {{records, page.planes[first + count - 1]}}};
}

ConstKpoolMlaCacheView KpoolMlaCacheLayout::layer(ConstPagedRecordView records, int layer) const {
    check_layer(*this, layer, records.page_tokens);
    const int count  = planes_per_layer(spec.storage);
    const auto first = std::size_t(count) * layer;
    return {
        {ConstPagedPlaneView{records, page.planes[first]},
         count == 3 ? ConstPagedPlaneView{records, page.planes[first + 1]} : ConstPagedPlaneView{},
         spec.storage},
        ConstKpoolIndexCacheView{{records, page.planes[first + count - 1]}}};
}

KpoolTailFields append_kpool_tail_fields(StateImageBuilder& builder, int index_dim, int layers) {
    if (index_dim <= 0 || layers <= 0) throw std::invalid_argument("Kpool tail: invalid geometry");
    return {builder.add(DType::BF16, {index_dim, 3}, layers, "Kpool raw keys"),
            builder.add(DType::BF16, {index_dim, 3}, layers, "Kpool raw gates")};
}

KpoolRawRecordLayout plan_kpool_raw_records(LayoutBuilder& builder, int dim, int width, int rows,
                                            int layers) {
    if (dim <= 0 || width <= 0 || rows < 1 || rows > 8 || layers <= 0)
        throw std::invalid_argument("Kpool records: invalid geometry");
    return {builder.add_tensor(DType::BF16, {dim, width, rows, layers}, 256, "Kpool raw keys"),
            builder.add_tensor(DType::BF16, {dim, width, rows, layers}, 256, "Kpool raw gates")};
}

KpoolRawRecordWriteView KpoolRawRecordLayout::write_view(MemorySpan backing, int layer, int batch,
                                                         const Tensor& offsets) const {
    if (layer < 0 || layer >= raw_keys.shape[3] || batch < 1 || batch > raw_keys.shape[2] ||
        raw_keys.shape != raw_gates.shape)
        throw std::invalid_argument("Kpool records: invalid write view");
    return {raw_keys.bind(backing).slice(3, layer, 1).slice(2, 0, batch),
            raw_gates.bind(backing).slice(3, layer, 1).slice(2, 0, batch), offsets};
}

KpoolRawRecordReadView KpoolRawRecordLayout::read_view(MemorySpan backing, const Tensor& base,
                                                       const Tensor& valid) const {
    return {raw_keys.bind(backing), raw_gates.bind(backing), base, valid};
}

KpoolBlockSelectionLayout plan_kpool_block_selection(LayoutBuilder& builder, int rows) {
    if (rows <= 0) throw std::invalid_argument("Kpool selection: invalid row count");
    return {builder.add_tensor(DType::I32, {512, rows}, 256, "Kpool pool IDs"),
            builder.add_tensor(DType::I32, {rows}, 256, "Kpool pool counts"),
            builder.add_tensor(DType::I32, {rows}, 256, "Kpool selection ends")};
}

KpoolBlockSelection KpoolBlockSelectionLayout::bind(MemorySpan backing) const {
    return {pool_ids.bind(backing), pool_counts.bind(backing), selection_end.bind(backing)};
}
} // namespace ninfer::ops
