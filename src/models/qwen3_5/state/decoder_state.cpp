#include "models/qwen3_5/state/decoder_state.h"

#include "core/device_scope.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5 {
namespace {

std::uint32_t page_count(std::uint32_t capacity) {
    if (capacity == 0) { throw std::invalid_argument("Paged KV capacity must be positive"); }
    return 1U + (capacity - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

std::int32_t checked_region_pages(std::uint64_t pages) {
    if (pages == 0 ||
        pages > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("QSA staging region page count is invalid");
    }
    return static_cast<std::int32_t>(pages);
}

constexpr std::uint32_t kStagingDefaultCacheBlocks = 4096; // 16k tokens per attention layer
// The region holds one layer of the visible context up to this many tokens (prefill stages the
// visible pages); longer contexts stage the union of each chunk's selections.
constexpr std::uint32_t kStagingDefaultRegionTokens = 262144;
constexpr std::int32_t kStagingBlocksPerPage        = kPagedKVPageSize / 4;

std::int32_t round_up_blocks(std::uint64_t blocks) {
    const std::uint64_t rounded = (blocks + kStagingBlocksPerPage - 1) / kStagingBlocksPerPage *
                                  static_cast<std::uint64_t>(kStagingBlocksPerPage);
    if (rounded > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("QSA staging region exceeds int32 blocks");
    }
    return static_cast<std::int32_t>(rounded);
}

// The QSA staging region of active KV in RAM: cache slots of every attention layer, the overflow
// area, then the identity table and the cache metadata, all in the main persistent layout.
QsaStagingLayout plan_qsa_staging(LayoutBuilder& builder, std::uint32_t layers,
                                  std::uint32_t capacity, std::int32_t kv_heads,
                                  const PagedKVStorageLayout& storage,
                                  const DecoderStateSpec& spec) {
    if (spec.staging_window <= 0 || spec.staging_selection_groups <= 0) {
        throw std::invalid_argument("QSA staging region needs the decode window and selection");
    }
    if (spec.kv_table_rows <= 0) {
        throw std::invalid_argument("QSA staging region needs the KV table rows");
    }
    QsaStagingLayout out;
    out.layers     = static_cast<std::int32_t>(layers);
    out.table_rows = spec.kv_table_rows;
    // Every lane of a decode/verify call may overflow its whole window.
    out.overflow_blocks =
        round_up_blocks(static_cast<std::uint64_t>(spec.staging_window) *
                        static_cast<std::uint64_t>(spec.kv_table_rows) *
                        static_cast<std::uint64_t>(spec.staging_selection_groups));
    // Prefill stages one layer of the whole visible context into the region, up to the region cap.
    const std::uint32_t region_tokens =
        std::min(capacity, spec.staging_region_tokens != 0 ? spec.staging_region_tokens
                                                           : kStagingDefaultRegionTokens);
    const std::uint64_t capacity_blocks = (static_cast<std::uint64_t>(region_tokens) + 3U) / 4U;
    const std::uint64_t needed =
        capacity_blocks > static_cast<std::uint64_t>(out.overflow_blocks)
            ? (capacity_blocks - static_cast<std::uint64_t>(out.overflow_blocks) + layers - 1U) /
                  layers
            : 1U;
    out.cache_blocks           = round_up_blocks(std::max<std::uint64_t>(
        needed,
        spec.staging_cache_blocks != 0 ? spec.staging_cache_blocks : kStagingDefaultCacheBlocks));
    const std::uint64_t blocks = static_cast<std::uint64_t>(out.layers) * out.cache_blocks +
                                 static_cast<std::uint64_t>(out.overflow_blocks);
    const std::int32_t pages =
        checked_region_pages(blocks / static_cast<std::uint64_t>(kStagingBlocksPerPage));
    const auto plane = [&](DType dtype, std::int32_t leading, const char* label) {
        return builder.add_tensor(dtype, {leading, kPagedKVPageSize, kv_heads, pages}, 256, label);
    };
    out.k_pages =
        plane(storage.key.data_dtype, storage.key.data_leading_extent, "QSA staging region K");
    out.v_pages =
        plane(storage.value.data_dtype, storage.value.data_leading_extent, "QSA staging region V");
    if (storage.key.has_scale()) {
        out.k_scale_pages = plane(storage.key.scale_dtype, storage.key.scale_leading_extent,
                                  "QSA staging region K scales");
    }
    if (storage.value.has_scale()) {
        out.v_scale_pages = plane(storage.value.scale_dtype, storage.value.scale_leading_extent,
                                  "QSA staging region V scales");
    }
    const std::int32_t slots = out.layers * out.cache_blocks;
    out.block_table = builder.add_tensor(DType::I32, {pages, 1}, 256, "QSA staging identity table");
    out.table_row =
        builder.add_tensor(DType::I32, {out.table_rows}, 256, "QSA staging identity rows");
    out.tags     = builder.add_tensor(DType::I64, {slots}, 256, "QSA staging tags");
    out.claims   = builder.add_tensor(DType::I64, {slots}, 256, "QSA staging claims");
    out.counters = builder.add_tensor(DType::I32, {out.layers * ops::kQsaStagingCounterWords}, 256,
                                      "QSA staging counters");
    out.row_starts = builder.add_tensor(DType::I32, {out.layers * out.table_rows}, 256,
                                        "QSA staging row starts");
    out.stats      = builder.add_tensor(DType::I64, {out.layers * ops::kQsaStagingStatCount}, 256,
                                        "QSA staging statistics");
    return out;
}

PagedKVCacheLayout plan_cache(LayoutBuilder& builder, std::uint32_t layers, std::uint32_t capacity,
                              std::int32_t kv_heads, std::int32_t head_dim, KvCacheStorage storage,
                              std::int32_t table_rows, std::uint32_t physical_page_groups,
                              bool qsa_index, LayoutBuilder* pages_builder,
                              const DecoderStateSpec* host_kv = nullptr,
                              std::uint32_t region_layers     = 0) {
    if (layers == 0 ||
        layers > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        kv_heads <= 0 || head_dim <= 0 || table_rows <= 0) {
        throw std::invalid_argument("Paged KV cache geometry is invalid");
    }
    const PagedKVStorageLayout layer_storage = paged_kv_storage_layout(storage, head_dim);

    const std::uint32_t logical_pages = page_count(capacity);
    if (physical_page_groups < logical_pages) {
        throw std::invalid_argument("Paged KV physical pages are below logical capacity");
    }

    if (host_kv != nullptr && !qsa_index) {
        throw std::invalid_argument("active KV in RAM needs the QSA index planes");
    }
    // Active KV in RAM: K/V, scales and the QSA records go to Host memory; block keys stay.
    const bool host = host_kv != nullptr;
    KVPageGeometry geometry;
    geometry.planes.reserve(static_cast<std::size_t>(layers) * layer_storage.planes_per_layer());
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        geometry.planes.push_back({layer_storage.key.data_dtype,
                                   layer_storage.key.data_leading_extent, kv_heads, 256, host});
        geometry.planes.push_back({layer_storage.value.data_dtype,
                                   layer_storage.value.data_leading_extent, kv_heads, 256, host});
        if (layer_storage.key.has_scale()) {
            geometry.planes.push_back({layer_storage.key.scale_dtype,
                                       layer_storage.key.scale_leading_extent, kv_heads, 256,
                                       host});
        }
        if (layer_storage.value.has_scale()) {
            geometry.planes.push_back({layer_storage.value.scale_dtype,
                                       layer_storage.value.scale_leading_extent, kv_heads, 256,
                                       host});
        }
    }
    // QSA index planes follow every layer's K/V planes, so K/V plane ordinals are unchanged.
    for (std::uint32_t layer = 0; qsa_index && layer < layers; ++layer) {
        geometry.planes.push_back({DType::BF16, ops::kQsaIndexRecordWords, 1, 256, host});
        geometry.planes.push_back({DType::BF16, ops::kQsaBlockKeyWords, 1, 256});
    }
    LayoutBuilder host_builder;
    DeviceKVPagePoolLayout pages =
        plan_device_kv_page_pool(pages_builder != nullptr ? *pages_builder : builder,
                                 DeviceKVPagePoolSpec{.page_group_count = physical_page_groups,
                                                      .geometry         = std::move(geometry)},
                                 host ? &host_builder : nullptr);
    // The region serves `region_layers` attention layers: this pool's, then a sharing pool's.
    std::optional<QsaStagingLayout> region;
    if (host && region_layers != 0) {
        region =
            plan_qsa_staging(builder, region_layers, capacity, kv_heads, layer_storage, *host_kv);
    }
    return PagedKVCacheLayout{
        .pages            = std::move(pages),
        .execution_tables = plan_kv_execution_tables(
            builder,
            KVExecutionTableSpec{.logical_page_capacity = logical_pages, .table_rows = table_rows}),
        .layers        = layers,
        .max_context   = capacity,
        .kv_heads      = kv_heads,
        .layer_storage = layer_storage,
        .qsa_index     = qsa_index,
        .elastic_pages = pages_builder != nullptr,
        .host_bytes    = host ? host_builder.finish(2U << 20, "active KV Host planes") : 0,
        .staging       = std::move(region),
    };
}

ops::QsaStagingRegion bind_qsa_staging(const QsaStagingLayout& layout, DeviceSpan backing,
                                       std::int32_t kv_heads, const PagedKVStorageLayout& storage) {
    ops::QsaStagingRegion region;
    region.pages = PagedKVBatchLayerView{
        .k_pages       = layout.k_pages.bind(backing),
        .v_pages       = layout.v_pages.bind(backing),
        .k_scale_pages = layout.k_scale_pages ? layout.k_scale_pages->bind(backing) : Tensor(),
        .v_scale_pages = layout.v_scale_pages ? layout.v_scale_pages->bind(backing) : Tensor(),
        .block_tables  = layout.block_table.bind(backing),
        .head_dim      = storage.head_dim,
        .num_kv_heads  = kv_heads,
        .storage       = storage.storage,
    };
    region.table_row       = layout.table_row.bind(backing);
    region.tags            = layout.tags.bind(backing);
    region.claims          = layout.claims.bind(backing);
    region.counters        = layout.counters.bind(backing);
    region.row_starts      = layout.row_starts.bind(backing);
    region.stats           = layout.stats.bind(backing);
    region.layers          = layout.layers;
    region.table_rows      = layout.table_rows;
    region.cache_blocks    = layout.cache_blocks;
    region.overflow_blocks = layout.overflow_blocks;
    return region;
}

// The Host planes' arena: one elastic range per Host plane (pages outermost), nothing backed yet.
std::unique_ptr<ElasticArena> make_host_arena(int device, const PagedKVCacheLayout& layout) {
    if (layout.host_bytes == 0) { return nullptr; }
    std::vector<ElasticRangeSpec> ranges;
    for (const DeviceKVPlaneLayout& plane : layout.pages.planes) {
        if (!plane.geometry.host) { continue; }
        ranges.push_back(ElasticRangeSpec{.offset = plane.storage.region.offset,
                                          .bytes  = plane.storage.region.bytes,
                                          .mapped = 0});
    }
    try {
        return std::make_unique<ElasticArena>(device, layout.host_bytes, ranges,
                                              ElasticMemory::Host);
    } catch (const std::exception& error) {
        throw std::runtime_error(std::string("active KV in RAM: reserving the Host KV planes "
                                             "failed (") +
                                 error.what() + ")");
    }
}

} // namespace

DecoderStateLayout plan_decoder_state(LayoutBuilder& builder, const DecoderStateSpec& spec,
                                      LayoutBuilder* pages_builder) {
    DecoderStateLayout layout;
    // Active KV in RAM moves both pools; the MTP layer stages through the text pool's region,
    // which gets one more cache layer per MTP layer.
    const DecoderStateSpec* host = spec.kv_ram_active ? &spec : nullptr;
    layout.text_kv               = plan_cache(
        builder, spec.full_attention_layers, spec.capacity, spec.kv_heads, spec.attention_head_dim,
        spec.kv_storage, spec.kv_table_rows, spec.text_physical_page_groups, spec.qsa_index,
        pages_builder, host, spec.full_attention_layers + (spec.enable_mtp ? spec.mtp_layers : 0U));
    if (spec.enable_mtp) {
        layout.mtp_kv =
            plan_cache(builder, spec.mtp_layers, spec.capacity, spec.kv_heads,
                       spec.attention_head_dim, spec.kv_storage, spec.kv_table_rows,
                       spec.mtp_physical_page_groups, spec.qsa_index, pages_builder, host);
    }
    return layout;
}

std::size_t active_kv_host_plane_bytes(const DecoderStateSpec& spec) {
    if (!spec.kv_ram_active) {
        throw std::invalid_argument("active KV Host planes need a kv_ram_active spec");
    }
    LayoutBuilder scratch;
    const DecoderStateLayout layout = plan_decoder_state(scratch, spec);
    return layout.text_kv.host_bytes + (layout.mtp_kv ? layout.mtp_kv->host_bytes : 0);
}

PagedKVCache::PagedKVCache(int device, DeviceSpan backing, const PagedKVCacheLayout& layout,
                           DeviceSpan elastic)
    : host_(make_host_arena(device, layout)),
      pages_(layout.elastic_pages ? elastic : backing, layout.pages,
             host_ ? host_->span() : DeviceSpan{}),
      execution_tables_(backing, layout.execution_tables, pages_), layers_(layout.layers),
      max_context_(layout.max_context), kv_heads_(layout.kv_heads),
      layer_storage_(layout.layer_storage), qsa_index_(layout.qsa_index) {
    if (pages_.plane_count() != static_cast<std::size_t>(layers_) *
                                    (layer_storage_.planes_per_layer() + (qsa_index_ ? 2U : 0U))) {
        throw std::invalid_argument("Paged KV layer plane inventory is inconsistent");
    }
    if (host_) { pages_.set_host_backing(*host_, kActiveKvHostStepPages, kActiveKvHostStepPages); }
    if (!layout.staging) { return; }
    staging_      = bind_qsa_staging(*layout.staging, backing, kv_heads_, layer_storage_);
    staging_view_ = &*staging_;
    // The region's device (rank 1 builds its cache with rank 0's device current).
    staging_device_ = device;
    const ScopedCurrentDevice scope(staging_device_);
    const std::int32_t pages = staging_->total_pages();
    std::vector<std::int32_t> identity(static_cast<std::size_t>(pages));
    for (std::int32_t i = 0; i < pages; ++i) { identity[static_cast<std::size_t>(i)] = i; }
    CUDA_CHECK(cudaMemcpy(staging_->pages.block_tables.data, identity.data(),
                          identity.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(staging_->table_row.data, 0, staging_->table_row.bytes()));
    CUDA_CHECK(cudaMemset(staging_->tags.data, 0xFF, staging_->tags.bytes()));
    CUDA_CHECK(cudaMemset(staging_->claims.data, 0, staging_->claims.bytes()));
    CUDA_CHECK(cudaMemset(staging_->row_starts.data, 0, staging_->row_starts.bytes()));
    CUDA_CHECK(cudaMemset(staging_->stats.data, 0, staging_->stats.bytes()));
    std::vector<std::int32_t> counters(
        static_cast<std::size_t>(staging_->layers) * ops::kQsaStagingCounterWords, 0);
    for (std::int32_t layer = 0; layer < staging_->layers; ++layer) {
        // Epoch 1: zeroed claims never match.
        counters[static_cast<std::size_t>(layer) * ops::kQsaStagingCounterWords] = 1;
    }
    CUDA_CHECK(cudaMemcpy(staging_->counters.data, counters.data(),
                          counters.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice));
    // Every rewrite of committed page bytes or of a row's binding forgets the cached blocks.
    pages_.set_write_listener([this](cudaStream_t stream) {
        const ScopedCurrentDevice region_device(staging_device_);
        ops::qsa_staging_invalidate(*staging_, stream);
    });
}

void PagedKVCache::share_staging(const PagedKVCache& owner, std::int32_t layer_base) {
    const ops::QsaStagingRegion* region = owner.qsa_staging();
    if (region == nullptr || staging_view_ != nullptr || layer_base < 0 ||
        layer_base + static_cast<std::int32_t>(layers_) > region->layers ||
        region->pages.storage != layer_storage_.storage ||
        region->pages.num_kv_heads != kv_heads_) {
        throw std::invalid_argument("active KV in RAM: the shared staging region does not fit");
    }
    staging_view_       = region;
    staging_layer_base_ = layer_base;
    staging_device_     = owner.staging_device_;
    // This pool's page writes and rebinds drop the cached blocks of every layer of the region.
    pages_.set_write_listener([this](cudaStream_t stream) {
        const ScopedCurrentDevice region_device(staging_device_);
        ops::qsa_staging_invalidate(*staging_view_, stream);
    });
}

void PagedKVCache::report_staging_stats(const char* why) const {
    if (!staging_) { return; }
    const ScopedCurrentDevice scope(staging_device_);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<long long> stats(static_cast<std::size_t>(staging_->stats.numel()));
    CUDA_CHECK(cudaMemcpy(stats.data(), staging_->stats.data, staging_->stats.bytes(),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemset(staging_->stats.data, 0, staging_->stats.bytes()));
    long long sum[ops::kQsaStagingStatCount] = {};
    for (std::int32_t layer = 0; layer < staging_->layers; ++layer) {
        for (int k = 0; k < ops::kQsaStagingStatCount; ++k) {
            sum[k] += stats[static_cast<std::size_t>(layer) * ops::kQsaStagingStatCount +
                            static_cast<std::size_t>(k)];
        }
    }
    const long long blocks = sum[ops::kQsaStagingHits] + sum[ops::kQsaStagingMisses] +
                             sum[ops::kQsaStagingShared] + sum[ops::kQsaStagingUncacheable] +
                             sum[ops::kQsaStagingCollisions];
    const long long fetched = sum[ops::kQsaStagingMisses] + sum[ops::kQsaStagingUncacheable] +
                              sum[ops::kQsaStagingCollisions];
    std::fprintf(
        stderr,
        "active KV stats (%s, device %d): fetch calls %lld, blocks %lld: hits %lld, "
        "shared %lld, misses %lld, uncacheable %lld, collisions %lld; fetched %.1f%%; "
        "stage calls %lld, staged pages %lld; prefill selections touch %lld of %lld "
        "visible blocks; union stage calls %lld, staged blocks %lld\n",
        why, staging_device_, sum[ops::kQsaStagingBlockCalls], blocks, sum[ops::kQsaStagingHits],
        sum[ops::kQsaStagingShared], sum[ops::kQsaStagingMisses], sum[ops::kQsaStagingUncacheable],
        sum[ops::kQsaStagingCollisions],
        blocks > 0 ? 100.0 * static_cast<double>(fetched) / static_cast<double>(blocks) : 0.0,
        sum[ops::kQsaStagingPageCalls], sum[ops::kQsaStagingPages],
        sum[ops::kQsaStagingUnionBlocks], sum[ops::kQsaStagingVisibleBlocks],
        sum[ops::kQsaStagingUnionCalls], sum[ops::kQsaStagingUnionStaged]);
}

PagedKVCacheView::PagedKVCacheView(const PagedKVCache& cache, Tensor block_table) noexcept
    : cache_(&cache), block_table_(block_table) {}

std::uint32_t PagedKVCacheView::max_context() const noexcept {
    return cache_ == nullptr ? 0 : cache_->max_context();
}

PagedKVLayerView PagedKVCacheView::layer_view(std::uint32_t layer) const {
    if (cache_ == nullptr) { throw std::logic_error("Paged KV execution view is empty"); }
    return cache_->layer_view(layer, block_table_);
}

PagedKVCacheView PagedKVCache::execution_view(const KVExecutionRowLease& row) const {
    if (!row.belongs_to(execution_tables_)) {
        throw std::invalid_argument("Paged KV execution row belongs to another cache");
    }
    return PagedKVCacheView(*this, execution_tables_.row(row.handle()));
}

PagedKVLayerView PagedKVCache::layer_view(std::uint32_t layer, Tensor block_table) const {
    if (layer >= layers_) { throw std::out_of_range("Paged KV layer is out of range"); }
    const std::size_t stride        = layer_storage_.planes_per_layer();
    const std::size_t base          = static_cast<std::size_t>(layer) * stride;
    const std::size_t k_scale_index = base + 2;
    const std::size_t v_scale_index =
        k_scale_index + static_cast<std::size_t>(layer_storage_.key.has_scale());
    return PagedKVLayerView{
        .k_pages       = pages_.plane(base),
        .v_pages       = pages_.plane(base + 1),
        .k_scale_pages = layer_storage_.key.has_scale() ? pages_.plane(k_scale_index) : Tensor(),
        .v_scale_pages = layer_storage_.value.has_scale() ? pages_.plane(v_scale_index) : Tensor(),
        .block_table   = block_table,
        .head_dim      = layer_storage_.head_dim,
        .num_kv_heads  = kv_heads_,
        .storage       = layer_storage_.storage,
    };
}

PagedKVBatchLayerView PagedKVCache::batch_layer_view(std::uint32_t layer) const {
    const PagedKVLayerView direct = layer_view(layer, Tensor());
    return PagedKVBatchLayerView{
        .k_pages       = direct.k_pages,
        .v_pages       = direct.v_pages,
        .k_scale_pages = direct.k_scale_pages,
        .v_scale_pages = direct.v_scale_pages,
        .block_tables  = execution_tables_.matrix(),
        .head_dim      = direct.head_dim,
        .num_kv_heads  = direct.num_kv_heads,
        .storage       = direct.storage,
    };
}

ops::QsaIndexPlanes PagedKVCache::qsa_index_planes(std::uint32_t layer) const {
    if (!qsa_index_ || layer >= layers_) {
        throw std::out_of_range("Paged KV has no QSA index planes for this layer");
    }
    const std::size_t base = static_cast<std::size_t>(layers_) * layer_storage_.planes_per_layer() +
                             2U * static_cast<std::size_t>(layer);
    return ops::QsaIndexPlanes{.records      = pages_.plane(base),
                               .block_keys   = pages_.plane(base + 1),
                               .block_tables = execution_tables_.matrix()};
}

std::size_t DecoderStateLayout::kv_payload_bytes() const noexcept {
    return text_kv.payload_bytes() + (mtp_kv ? mtp_kv->payload_bytes() : 0);
}

DecoderState::DecoderState(int device, DeviceSpan backing, const DecoderStateLayout& layout,
                           DeviceSpan elastic)
    : text_kv(device, backing, layout.text_kv, elastic) {
    if (layout.text_kv.elastic_pages && elastic.data == nullptr) {
        throw std::invalid_argument("elastic KV pages need their backing");
    }
    if (layout.mtp_kv) {
        mtp_kv.emplace(device, backing, *layout.mtp_kv, elastic);
        if (text_kv.qsa_staging() != nullptr && layout.mtp_kv->host_bytes != 0) {
            mtp_kv->share_staging(text_kv, static_cast<std::int32_t>(text_kv.layers()));
        }
    }
}

PagedKVCache* DecoderState::mtp_cache() noexcept { return mtp_kv ? &*mtp_kv : nullptr; }

const PagedKVCache* DecoderState::mtp_cache() const noexcept { return mtp_kv ? &*mtp_kv : nullptr; }

} // namespace ninfer::models::qwen3_5
