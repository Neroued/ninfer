#include "models/qwen3_5/program/prefix/hybrid_cache.h"

#include "core/device.h"
#include "core/paged_kv_cache.h"

#include <algorithm>
#include <exception>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::models::qwen3_5::detail {

using runtime::prefix_cache::CacheCostModel;
using runtime::prefix_cache::CopyState;
using runtime::prefix_cache::InsertResult;
using runtime::prefix_cache::kNoId;
using runtime::prefix_cache::NodeRef;
using runtime::prefix_cache::NodeView;
using runtime::prefix_cache::PrefixIndexConfig;
using runtime::prefix_cache::SnapshotRef;
using runtime::prefix_cache::SnapshotView;
using runtime::prefix_cache::TapPlannerConfig;

namespace {

constexpr auto kFullPage = static_cast<std::uint32_t>(kPagedKVPageSize);
// Largest single pinned allocation of the Host slab pool.
constexpr std::size_t kHostChunkBytes = std::size_t{4} << 30U;

std::span<const std::byte* const> const_records(const std::vector<std::byte*>& records) noexcept {
    return {reinterpret_cast<const std::byte* const*>(records.data()), records.size()};
}

} // namespace

HybridPrefixCache::HybridPrefixCache(const PrefixIndexConfig& config, TapPlannerConfig taps,
                                     LogicalKVPageStore& text_pages,
                                     LogicalKVPageStore* backend_pages, StateImageStore& states,
                                     qwen3_5::StateImageDevicePool& state_pool,
                                     const HybridHostLayout& host,
                                     std::vector<HybridRestoreLayer> layers,
                                     StartupPhaseScope* pin_progress)
    : config_(config), taps_(taps), text_pages_(&text_pages), backend_pages_(backend_pages),
      states_(&states), state_pool_(&state_pool), host_layout_(host), images_(config.max_snapshots),
      layers_(std::move(layers)) {
    if (host.backend.has_value() != (backend_pages != nullptr)) {
        throw std::invalid_argument("hybrid Host layout does not match the backend KV pool");
    }
    const auto attention_layers   = static_cast<std::size_t>(std::count_if(
        layers_.begin(), layers_.end(), [](const HybridRestoreLayer& l) { return l.attention; }));
    const std::size_t text_planes = text_pages.physical_pool().plane_count();
    if (layers_.empty() || attention_layers == 0 || text_planes % attention_layers != 0) {
        throw std::invalid_argument("hybrid restore layers do not match the text KV planes");
    }
    planes_per_attention_layer_ = text_planes / attention_layers;
    for (const HybridRestoreLayer& layer : layers_) {
        if ((layer.attention && layer.index >= attention_layers) ||
            (!layer.attention && layer.index >= state_pool.linear().layer_count())) {
            throw std::invalid_argument("hybrid restore layer index is out of range");
        }
    }
    if (config.host_slabs != 0) {
        if (!config.host_blocks || host.slab_bytes == 0 || config.image_slabs != host.image_slabs) {
            throw std::invalid_argument("hybrid Host tier configuration is inconsistent");
        }
        slabs_per_chunk_ = static_cast<std::uint32_t>(
            std::max<std::size_t>(1U, kHostChunkBytes / host.slab_bytes));
        const std::uint64_t total = static_cast<std::uint64_t>(config.host_slabs) * host.slab_bytes;
        std::uint64_t pinned      = 0;
        host_chunks_.reserve((config.host_slabs + slabs_per_chunk_ - 1U) / slabs_per_chunk_);
        for (std::uint32_t first = 0; first < config.host_slabs; first += slabs_per_chunk_) {
            const std::uint32_t slabs = std::min(slabs_per_chunk_, config.host_slabs - first);
            const std::size_t bytes   = static_cast<std::size_t>(slabs) * host.slab_bytes;
            try {
                host_chunks_.emplace_back(bytes);
            } catch (const std::exception& error) {
                throw std::runtime_error("pinning the hybrid prefix cache Host tier failed after " +
                                         std::to_string(pinned >> 20U) + " of " +
                                         std::to_string(total >> 20U) + " MiB (" + error.what() +
                                         "); lower --host-cache-mib");
            }
            pinned += bytes;
            if (pin_progress != nullptr) { pin_progress->progress(pinned, total); }
        }
        CUDA_CHECK(cudaStreamCreateWithFlags(&restore_stream_, cudaStreamNonBlocking));
    }
    blocks_.reserve(config.max_nodes);
    index_.emplace(config_, *this);
}

HybridPrefixCache::~HybridPrefixCache() {
    // Stores referenced by the cache may already be gone; only CUDA objects are released here.
    if (restore_stream_ != nullptr) {
        (void)cudaStreamSynchronize(restore_stream_);
        (void)cudaStreamDestroy(restore_stream_);
    }
    for (PendingWrite& write : pending_) {
        if (write.done != nullptr) {
            (void)cudaEventSynchronize(write.done);
            (void)cudaEventDestroy(write.done);
        }
    }
    recycle_events(restore_);
    for (RestoreBatch& batch : landing_) { recycle_events(batch); }
    for (cudaEvent_t event : spare_events_) { (void)cudaEventDestroy(event); }
}

void HybridPrefixCache::set_cost(const CacheCostModel& cost) {
    config_.cost = cost;
    index_->set_cost(cost);
}

// ---- ids and references ------------------------------------------------------------------------

std::uint32_t HybridPrefixCache::allocate_block_id(HybridBlockPages pages) {
    if (!free_blocks_.empty()) {
        const std::uint32_t id = free_blocks_.back();
        free_blocks_.pop_back();
        blocks_[id] = pages;
        return id;
    }
    if (blocks_.size() >= kNoId - 1U) { throw std::overflow_error("hybrid block ids exhausted"); }
    blocks_.emplace_back(pages);
    return static_cast<std::uint32_t>(blocks_.size() - 1U);
}

void HybridPrefixCache::free_block_id(std::uint32_t id) noexcept {
    blocks_[id].reset();
    free_blocks_.push_back(id);
}

const HybridBlockPages& HybridPrefixCache::block(std::uint32_t device_id) const {
    if (device_id >= blocks_.size() || !blocks_[device_id]) {
        throw std::logic_error("hybrid block id is not registered");
    }
    return *blocks_[device_id];
}

void HybridPrefixCache::retain(const HybridBlockPages& pages, std::uint32_t text_columns,
                               std::uint32_t backend_columns) {
    const auto check = [&](LogicalKVPageStore& store, LogicalKVPageHandle page,
                           std::uint32_t columns) {
        if (!store.can_retain_reference(page, false) || !store.device_resident(page) ||
            store.committed_columns(page) < columns) {
            throw std::logic_error("hybrid cache page is not retainable");
        }
    };
    check(*text_pages_, pages.text, text_columns);
    if (pages.backend.has_value() != (backend_pages_ != nullptr)) {
        throw std::logic_error("hybrid block does not match the KV pools");
    }
    if (pages.backend) { check(*backend_pages_, *pages.backend, backend_columns); }
    const auto retain_one = [&](LogicalKVPageStore& store, LogicalKVPageHandle page,
                                std::uint32_t columns) {
        // A committed full page (or a finishing sequence's tail) receives no further writes.
        // Clearing the writer flag lets later readers pin it as an immutable source.
        if (store.writer_references(page) != 0) { store.set_writer(page, false); }
        store.retain_reference(page, false);
        if (columns != 0) { store.protect_coverage(page, columns); }
    };
    retain_one(*text_pages_, pages.text, text_columns);
    if (pages.backend) { retain_one(*backend_pages_, *pages.backend, backend_columns); }
}

void HybridPrefixCache::release(const HybridBlockPages& pages) noexcept {
    if (!text_pages_->release_reference(pages.text, false)) { std::terminate(); }
    if (pages.backend &&
        (backend_pages_ == nullptr || !backend_pages_->release_reference(*pages.backend, false))) {
        std::terminate();
    }
}

void HybridPrefixCache::adopt_destination(const HybridBlockPages& pages, std::uint32_t columns) {
    if (pages.backend.has_value() != (backend_pages_ != nullptr)) {
        throw std::logic_error("hybrid restore destination does not match the KV pools");
    }
    text_pages_->publish_transfer_destination(pages.text, false);
    text_pages_->protect_coverage(pages.text, columns);
    if (pages.backend) {
        backend_pages_->publish_transfer_destination(*pages.backend, false);
        backend_pages_->protect_coverage(*pages.backend, columns);
    }
}

// ---- blocks ------------------------------------------------------------------------------------

InsertResult HybridPrefixCache::insert_block(NodeRef parent, std::uint64_t lookup_hash,
                                             std::span<const TokenId> tokens, std::uint64_t extra,
                                             HybridBlockPages pages) {
    // Only a new node or a host-only node adopts the caller's page; a Device-resident duplicate
    // leaves it private to the caller.
    const std::optional<NodeRef> existing = index_->find_child(parent, lookup_hash, tokens, extra);
    const bool adopts      = !existing || index_->node(*existing).device == CopyState::Absent;
    const std::uint32_t id = allocate_block_id(pages);
    if (adopts) {
        try {
            retain(pages, kFullPage, kFullPage);
        } catch (...) {
            free_block_id(id);
            throw;
        }
    }
    InsertResult result;
    try {
        result = index_->insert_block(parent, lookup_hash, tokens, extra, id);
    } catch (...) {
        if (adopts) { release(pages); }
        free_block_id(id);
        throw;
    }
    if (!result.device_attached) {
        if (adopts) { release(pages); }
        free_block_id(id);
        ++counters_.blocks_duplicate;
    } else if (result.inserted) {
        ++counters_.blocks_inserted;
    } else {
        ++counters_.blocks_reattached;
    }
    return result;
}

std::uint32_t HybridPrefixCache::register_tail(HybridBlockPages pages, std::uint32_t text_columns,
                                               std::uint32_t backend_columns) {
    const std::uint32_t id = allocate_block_id(pages);
    try {
        retain(pages, text_columns, backend_columns);
    } catch (...) {
        free_block_id(id);
        throw;
    }
    return id;
}

std::uint32_t HybridPrefixCache::register_tail_copy(HybridBlockPages pages, std::uint32_t columns) {
    adopt_destination(pages, columns);
    try {
        return allocate_block_id(pages);
    } catch (...) {
        release(pages);
        throw;
    }
}

// ---- snapshot images ---------------------------------------------------------------------------

void HybridPrefixCache::attach_image(SnapshotRef snapshot, StateImageHandle image) {
    if (!index_->valid(snapshot) || snapshot.index >= images_.size() ||
        images_[snapshot.index].valid()) {
        throw std::logic_error("hybrid snapshot image binding is invalid");
    }
    images_[snapshot.index] = image;
}

StateImageHandle HybridPrefixCache::image(SnapshotRef snapshot) const {
    if (!index_->valid(snapshot) || !images_[snapshot.index].valid()) {
        throw std::logic_error("hybrid snapshot has no Device StateImage");
    }
    return images_[snapshot.index];
}

// ---- transfers ---------------------------------------------------------------------------------

std::byte* HybridPrefixCache::slab(std::uint32_t id) const noexcept {
    return static_cast<std::byte*>(host_chunks_[id / slabs_per_chunk_].data()) +
           static_cast<std::size_t>(id % slabs_per_chunk_) * host_layout_.slab_bytes;
}

void HybridPrefixCache::add_copies(PageCopies& copies, const HybridBlockPages& pages,
                                   std::uint32_t slab_id) {
    std::byte* record         = slab(slab_id);
    const std::uint32_t chunk = slab_id / slabs_per_chunk_;
    copies.text_pages.push_back(text_pages_->physical(pages.text));
    copies.text_records.push_back(record);
    copies.text_groups.push_back(chunk);
    if (pages.backend) {
        copies.backend_pages.push_back(backend_pages_->physical(*pages.backend));
        copies.backend_records.push_back(record + host_layout_.backend_offset);
        copies.backend_groups.push_back(chunk);
    }
}

void HybridPrefixCache::enqueue_to_host(const PageCopies& copies, cudaStream_t stream) const {
    if (!copies.text_pages.empty()) {
        text_pages_->physical_pool().copy_to_host_records(
            copies.text_pages, copies.text_records, copies.text_groups, host_layout_.text, stream);
    }
    if (!copies.backend_pages.empty()) {
        backend_pages_->physical_pool().copy_to_host_records(
            copies.backend_pages, copies.backend_records, copies.backend_groups,
            *host_layout_.backend, stream);
    }
}

void HybridPrefixCache::enqueue_from_host(const PageCopies& copies, cudaStream_t stream) const {
    if (!copies.text_pages.empty()) {
        text_pages_->physical_pool().copy_from_host_records(const_records(copies.text_records),
                                                            copies.text_groups, copies.text_pages,
                                                            host_layout_.text, stream);
    }
    if (!copies.backend_pages.empty()) {
        backend_pages_->physical_pool().copy_from_host_records(
            const_records(copies.backend_records), copies.backend_groups, copies.backend_pages,
            *host_layout_.backend, stream);
    }
}

cudaEvent_t HybridPrefixCache::take_event() {
    if (!spare_events_.empty()) {
        cudaEvent_t event = spare_events_.back();
        spare_events_.pop_back();
        return event;
    }
    cudaEvent_t event = nullptr;
    CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    return event;
}

void HybridPrefixCache::order_after(cudaStream_t producer, cudaStream_t consumer) {
    cudaEvent_t ready = take_event();
    try {
        CUDA_CHECK(cudaEventRecord(ready, producer));
        CUDA_CHECK(cudaStreamWaitEvent(consumer, ready, 0));
    } catch (...) {
        spare_events_.push_back(ready);
        throw;
    }
    // The wait captured the recorded work; the event may be re-recorded.
    spare_events_.push_back(ready);
}

// ---- Host write-through ------------------------------------------------------------------------

bool HybridPrefixCache::start_snapshot_host_write(SnapshotRef snapshot, cudaStream_t producer,
                                                  cudaStream_t transfer) {
    if (!host_tier() || !index_->valid(snapshot) || !images_[snapshot.index].valid()) {
        return false;
    }
    {
        const SnapshotView view = index_->snapshot(snapshot);
        if (view.host != CopyState::Absent || view.device_slot == kNoId ||
            (view.tail_len != 0 && view.tail_device_copy != CopyState::Resident)) {
            return false;
        }
    }
    index_->pin_snapshot(snapshot);
    if (!index_->begin_snapshot_host_fill(snapshot)) {
        index_->unpin_snapshot(snapshot);
        return false;
    }
    try {
        const SnapshotView view = index_->snapshot(snapshot);
        std::vector<std::byte*> segments;
        segments.reserve(host_layout_.image_slabs);
        for (std::uint32_t i = 0; i < host_layout_.image_slabs; ++i) {
            segments.push_back(slab(view.host_slabs[i]));
        }
        order_after(producer, transfer);
        state_pool_->copy_to_host_segments(states_->physical_slot(images_[snapshot.index]),
                                           segments, host_layout_.slab_bytes, transfer);
        std::uint64_t bytes = host_layout_.image_bytes;
        if (view.tail_len != 0) {
            write_scratch_.clear();
            add_copies(write_scratch_, block(view.tail_device),
                       view.host_slabs[host_layout_.image_slabs]);
            enqueue_to_host(write_scratch_, transfer);
            bytes += host_layout_.block_payload_bytes;
        }
        PendingWrite write;
        write.snapshot = snapshot;
        write.done     = take_event();
        pending_.push_back(std::move(write));
        CUDA_CHECK(cudaEventRecord(pending_.back().done, transfer));
        ++counters_.host_image_writes;
        counters_.host_write_bytes += bytes;
    } catch (...) {
        // Copies may be in flight into the slabs; they return to the index only once idle.
        (void)cudaStreamSynchronize(transfer);
        if (!pending_.empty() && pending_.back().snapshot == snapshot) {
            if (pending_.back().done != nullptr) { spare_events_.push_back(pending_.back().done); }
            pending_.pop_back();
        }
        index_->abort_snapshot_host_fill(snapshot);
        index_->unpin_snapshot(snapshot);
        throw;
    }
    return true;
}

void HybridPrefixCache::start_block_host_writes(std::span<const NodeRef> nodes,
                                                cudaStream_t producer, cudaStream_t transfer) {
    if (!host_tier()) { return; }
    PendingWrite write;
    write_scratch_.clear();
    const auto abort_batch = [&]() noexcept {
        for (const NodeRef node : write.nodes) {
            index_->abort_host_fill(node);
            index_->unpin_node(node);
        }
        write.nodes.clear();
    };
    try {
        for (const NodeRef node : nodes) {
            const NodeView view = index_->node(node);
            if (view.device != CopyState::Resident || view.host != CopyState::Absent) { continue; }
            index_->pin_node(node);
            const std::optional<std::uint32_t> slab_id = index_->begin_host_fill(node);
            if (!slab_id) {
                // Everything left on the Host tier is pinned or filling; retry at a later release.
                index_->unpin_node(node);
                break;
            }
            write.nodes.push_back(node);
            add_copies(write_scratch_, block(view.device_id), *slab_id);
        }
    } catch (...) {
        abort_batch();
        throw;
    }
    if (write.nodes.empty()) { return; }
    try {
        order_after(producer, transfer);
        enqueue_to_host(write_scratch_, transfer);
        write.done = take_event();
        CUDA_CHECK(cudaEventRecord(write.done, transfer));
    } catch (...) {
        (void)cudaStreamSynchronize(transfer);
        if (write.done != nullptr) { spare_events_.push_back(write.done); }
        abort_batch();
        throw;
    }
    counters_.host_block_writes += write.nodes.size();
    counters_.host_write_bytes += write.nodes.size() * host_layout_.block_payload_bytes;
    pending_.push_back(std::move(write));
}

void HybridPrefixCache::publish_write(PendingWrite& write) {
    for (const NodeRef node : write.nodes) {
        index_->complete_host_fill(node);
        index_->unpin_node(node);
    }
    if (write.snapshot) {
        index_->complete_snapshot_host_fill(*write.snapshot);
        index_->unpin_snapshot(*write.snapshot);
    }
    spare_events_.push_back(write.done);
    write.done = nullptr;
}

void HybridPrefixCache::poll() {
    // Every write shares the transfer stream and every restore the restore stream, so each list
    // completes in submission order.
    while (!pending_.empty()) {
        const cudaError_t status = cudaEventQuery(pending_.front().done);
        if (status == cudaErrorNotReady) { break; }
        CUDA_CHECK(status);
        publish_write(pending_.front());
        pending_.pop_front();
    }
    while (!landing_.empty()) {
        const cudaError_t status = cudaEventQuery(landing_.front().layers.back());
        if (status == cudaErrorNotReady) { break; }
        CUDA_CHECK(status);
        finish_landing(landing_.front());
        landing_.pop_front();
    }
}

void HybridPrefixCache::drain() {
    while (!pending_.empty()) {
        CUDA_CHECK(cudaEventSynchronize(pending_.front().done));
        publish_write(pending_.front());
        pending_.pop_front();
    }
    while (!landing_.empty()) {
        CUDA_CHECK(cudaEventSynchronize(landing_.front().layers.back()));
        finish_landing(landing_.front());
        landing_.pop_front();
    }
}

// ---- Host restores -----------------------------------------------------------------------------

void HybridPrefixCache::open_restore(cudaStream_t producer) {
    if (!host_tier() || restore_.open) {
        throw std::logic_error("hybrid restore cannot be opened");
    }
    order_after(producer, restore_stream_);
    restore_.open = true;
}

void HybridPrefixCache::restore_block(NodeRef node, HybridBlockPages destination) {
    if (!restore_.open || restore_.submitted) {
        throw std::logic_error("hybrid restore is not open");
    }
    const NodeView view = index_->node(node);
    if (view.device != CopyState::Absent || view.host != CopyState::Resident) {
        throw std::logic_error("hybrid restore source block is not host-only");
    }
    adopt_destination(destination, kFullPage);
    const std::uint32_t id = allocate_block_id(destination);
    try {
        index_->begin_device_fill(node, id);
    } catch (...) {
        release(destination);
        free_block_id(id);
        throw;
    }
    restore_.nodes.push_back(node);
    add_copies(restore_.copies, destination, view.host_slab);
}

void HybridPrefixCache::restore_tail(SnapshotRef snapshot, HybridBlockPages destination) {
    if (!restore_.open || restore_.submitted || restore_.tail) {
        throw std::logic_error("hybrid restore is not open");
    }
    const SnapshotView view = index_->snapshot(snapshot);
    if (view.tail_len == 0 || view.tail_device_copy != CopyState::Absent ||
        view.host != CopyState::Resident) {
        throw std::logic_error("hybrid restore source tail is not host-only");
    }
    adopt_destination(destination, view.tail_len);
    const std::uint32_t id = allocate_block_id(destination);
    try {
        index_->begin_tail_device_fill(snapshot, id);
    } catch (...) {
        release(destination);
        free_block_id(id);
        throw;
    }
    restore_.tail = snapshot;
    add_copies(restore_.tail_copies, destination, view.host_slabs[host_layout_.image_slabs]);
}

void HybridPrefixCache::restore_image(SnapshotRef snapshot, std::int32_t device_slot) {
    if (!restore_.open || restore_.submitted || restore_.image) {
        throw std::logic_error("hybrid restore is not open");
    }
    if (index_->snapshot(snapshot).host != CopyState::Resident) {
        throw std::logic_error("hybrid restore source image is not on the Host tier");
    }
    if (restore_.tail && *restore_.tail != snapshot) {
        throw std::logic_error("hybrid restore mixes two snapshots");
    }
    restore_.image      = snapshot;
    restore_.image_slot = device_slot;
}

void HybridPrefixCache::submit_restore() {
    if (!restore_.open || restore_.submitted) {
        throw std::logic_error("hybrid restore is not open");
    }
    std::vector<const std::byte*> segments;
    if (restore_.image) {
        const SnapshotView view = index_->snapshot(*restore_.image);
        segments.reserve(host_layout_.image_slabs);
        for (std::uint32_t i = 0; i < host_layout_.image_slabs; ++i) {
            segments.push_back(slab(view.host_slabs[i]));
        }
    }
    const auto record = [&]() {
        cudaEvent_t event = take_event();
        try {
            CUDA_CHECK(cudaEventRecord(event, restore_stream_));
        } catch (...) {
            spare_events_.push_back(event);
            throw;
        }
        return event;
    };
    // Prelude: what activation reads (the snapshot tail) and what a pass reads outside the layer
    // stack (backend KV, the continuation hidden, DFlash local state).
    enqueue_from_host(restore_.tail_copies, restore_stream_);
    if (!restore_.copies.backend_pages.empty()) {
        backend_pages_->physical_pool().copy_from_host_records(
            const_records(restore_.copies.backend_records), restore_.copies.backend_groups,
            restore_.copies.backend_pages, *host_layout_.backend, restore_stream_);
    }
    if (restore_.image) {
        state_pool_->copy_from_host_segments(
            segments, host_layout_.slab_bytes, restore_.image_slot,
            qwen3_5::StateImagePart{.kind = qwen3_5::StateImagePart::Kind::Rest}, restore_stream_);
    }
    restore_.prelude = record();
    // Then each layer's own state, in forward order, so a pass can start on the first layers
    // while later ones are still arriving.
    restore_.layers.reserve(layers_.size());
    for (const HybridRestoreLayer& layer : layers_) {
        if (layer.attention) {
            if (!restore_.copies.text_pages.empty()) {
                const std::size_t begin = layer.index * planes_per_attention_layer_;
                text_pages_->physical_pool().copy_from_host_records(
                    const_records(restore_.copies.text_records), restore_.copies.text_groups,
                    restore_.copies.text_pages, host_layout_.text, begin,
                    begin + planes_per_attention_layer_, restore_stream_);
            }
        } else if (restore_.image) {
            state_pool_->copy_from_host_segments(
                segments, host_layout_.slab_bytes, restore_.image_slot,
                qwen3_5::StateImagePart{.kind  = qwen3_5::StateImagePart::Kind::LinearLayer,
                                        .layer = layer.index},
                restore_stream_);
        }
        restore_.layers.push_back(record());
    }
    restore_.submitted = true;
    restore_.ticket    = next_ticket_++;
    counters_.host_block_restores += restore_.nodes.size();
    counters_.host_tail_restores += restore_.tail ? 1U : 0U;
    counters_.host_image_restores += restore_.image ? 1U : 0U;
    counters_.host_restore_bytes +=
        (restore_.nodes.size() + (restore_.tail ? 1U : 0U)) * host_layout_.block_payload_bytes +
        (restore_.image ? host_layout_.image_bytes : 0U);
}

std::uint64_t HybridPrefixCache::land_restore(cudaStream_t consumer) {
    if (!restore_.submitted) { throw std::logic_error("hybrid restore was not submitted"); }
    CUDA_CHECK(cudaStreamWaitEvent(consumer, restore_.prelude, 0));
    // The admission's own pins end at activation; the copies' sources and destinations stay
    // pinned by the batch until it lands.
    for (const NodeRef node : restore_.nodes) { index_->pin_node(node); }
    const std::optional<SnapshotRef> source = restore_.tail ? restore_.tail : restore_.image;
    if (source) {
        index_->pin_snapshot(*source);
        restore_.pinned_snapshot = source;
    }
    const std::uint64_t ticket = restore_.ticket;
    landing_.push_back(std::move(restore_));
    restore_ = RestoreBatch{};
    return ticket;
}

void HybridPrefixCache::detach_prefetch() {
    if (!restore_.submitted || restore_.image || restore_.tail) {
        throw std::logic_error("hybrid prefetch is not a submitted batch of blocks");
    }
    for (const NodeRef node : restore_.nodes) { index_->pin_node(node); }
    restore_.prefetch = true;
    counters_.prefetched_blocks += restore_.nodes.size();
    landing_.push_back(std::move(restore_));
    restore_ = RestoreBatch{};
}

bool HybridPrefixCache::prefetch_landing() const noexcept {
    return std::any_of(landing_.begin(), landing_.end(),
                       [](const RestoreBatch& batch) { return batch.prefetch; });
}

void HybridPrefixCache::settle_prefetch() {
    // One restore stream completes batches in submission order: the last prefetch batch's final
    // event covers every batch submitted before it.
    for (auto batch = landing_.rbegin(); batch != landing_.rend(); ++batch) {
        if (batch->prefetch) {
            CUDA_CHECK(cudaEventSynchronize(batch->layers.back()));
            break;
        }
    }
    poll();
}

std::span<const cudaEvent_t>
HybridPrefixCache::restore_layer_events(std::uint64_t ticket) const noexcept {
    for (const RestoreBatch& batch : landing_) {
        if (batch.ticket == ticket) { return batch.layers; }
    }
    return {};
}

void HybridPrefixCache::order_after_restore(std::uint64_t ticket, cudaStream_t consumer) const {
    for (const RestoreBatch& batch : landing_) {
        if (batch.ticket == ticket) {
            CUDA_CHECK(cudaStreamWaitEvent(consumer, batch.layers.back(), 0));
            return;
        }
    }
}

void HybridPrefixCache::finish_landing(RestoreBatch& batch) {
    for (const NodeRef node : batch.nodes) {
        index_->complete_device_fill(node);
        index_->unpin_node(node);
    }
    if (batch.tail) { index_->complete_tail_device_fill(*batch.tail); }
    if (batch.pinned_snapshot) { index_->unpin_snapshot(*batch.pinned_snapshot); }
    recycle_events(batch);
}

void HybridPrefixCache::recycle_events(RestoreBatch& batch) noexcept {
    try {
        if (batch.prelude != nullptr) { spare_events_.push_back(batch.prelude); }
        spare_events_.insert(spare_events_.end(), batch.layers.begin(), batch.layers.end());
    } catch (...) {
        // Out of memory while recycling: release the events instead.
        if (batch.prelude != nullptr) { (void)cudaEventDestroy(batch.prelude); }
        for (cudaEvent_t event : batch.layers) { (void)cudaEventDestroy(event); }
    }
    batch.prelude = nullptr;
    batch.layers.clear();
}

void HybridPrefixCache::abort_restore() noexcept {
    if (!restore_.open) { return; }
    if (restore_stream_ != nullptr) { (void)cudaStreamSynchronize(restore_stream_); }
    try {
        for (const NodeRef node : restore_.nodes) {
            if (index_->valid(node) && index_->node(node).device == CopyState::Filling) {
                index_->abort_device_fill(node);
            }
        }
        if (restore_.tail && index_->valid(*restore_.tail) &&
            index_->snapshot(*restore_.tail).tail_device_copy == CopyState::Filling) {
            index_->abort_tail_device_fill(*restore_.tail);
        }
    } catch (...) { std::terminate(); }
    reset_restore();
}

void HybridPrefixCache::reset_restore() noexcept {
    recycle_events(restore_);
    restore_ = RestoreBatch{};
}

// ---- recovery ----------------------------------------------------------------------------------

void HybridPrefixCache::clear() noexcept {
    // Engine-wide recovery: wait for every copy touching cache memory, then release every
    // reference so the stores end empty of cache ownership.
    if (restore_stream_ != nullptr) { (void)cudaStreamSynchronize(restore_stream_); }
    for (PendingWrite& write : pending_) {
        if (write.done != nullptr) {
            (void)cudaEventSynchronize(write.done);
            spare_events_.push_back(write.done);
        }
    }
    pending_.clear();
    // The index is rebuilt below, so landing batches only return their events.
    for (RestoreBatch& batch : landing_) { recycle_events(batch); }
    landing_.clear();
    reset_restore();
    for (std::optional<HybridBlockPages>& pages : blocks_) {
        if (!pages) { continue; }
        (void)text_pages_->release_reference(pages->text, false);
        if (pages->backend && backend_pages_ != nullptr) {
            (void)backend_pages_->release_reference(*pages->backend, false);
        }
        pages.reset();
    }
    blocks_.clear();
    free_blocks_.clear();
    for (StateImageHandle& image : images_) {
        if (image.valid() && states_->valid(image)) { (void)states_->release(image); }
        image = {};
    }
    index_.reset();
    try {
        index_.emplace(config_, *this);
    } catch (...) { std::terminate(); }
}

// ---- PrefixIndexBackend ------------------------------------------------------------------------

void HybridPrefixCache::release_device_block(std::uint32_t device_id) noexcept {
    if (device_id >= blocks_.size() || !blocks_[device_id]) { std::terminate(); }
    release(*blocks_[device_id]);
    free_block_id(device_id);
    ++counters_.evicted_blocks;
}

void HybridPrefixCache::release_snapshot(SnapshotRef snapshot) noexcept {
    if (snapshot.index >= images_.size()) { std::terminate(); }
    StateImageHandle& image = images_[snapshot.index];
    if (image.valid()) {
        if (!states_->release(image)) { std::terminate(); }
        image = {};
    }
}

void HybridPrefixCache::drop_snapshot_device_image(SnapshotRef snapshot) noexcept {
    // The snapshot survives on its Host slabs; its Device StateImage is released outright.
    release_snapshot(snapshot);
}

} // namespace ninfer::models::qwen3_5::detail
