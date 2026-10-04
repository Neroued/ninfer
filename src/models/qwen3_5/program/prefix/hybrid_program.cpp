// Hybrid prefix cache orchestration inside the Qwen3.5 Program
// (docs/maintainer/hybrid-prefix-cache.md). Admission (staging, Host restores, activation),
// block publication, prefill taps and terminal snapshots; the index and physical bindings live in
// hybrid_cache.{h,cpp}.

#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/context.h"
#include "models/qwen3_5/program/context_work.h"
#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/execution/vision.h"
#include "models/qwen3_5/program/vision_control.h"
#include "models/qwen3_5/program/vision_prefill.h"
#include "models/qwen3_5/program/prefix/block_keys.h"
#include "core/device.h"
#include "core/startup.h"
#include "runtime/prefix_cache/block_hash.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdint>
#include <functional>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

namespace pc = runtime::prefix_cache;

namespace {

constexpr std::uint32_t kBlock = pc::kBlockTokens;

// Blocks one prefetch batch copies (§6.6): about 20 ms of PCIe for 27B INT8, the longest a quote
// may wait for blocks it is about to resume over.
constexpr std::uint32_t kPrefetchBatchBlocks = 256;

std::uint64_t all_vision_key(std::span<const VisionTokenRange> ranges) noexcept {
    std::uint64_t cumulative = 0;
    for (const VisionTokenRange& range : ranges) {
        cumulative = accumulate_vision(cumulative, range.key);
    }
    return cumulative;
}

bool inside_vision(std::uint32_t frontier, std::span<const VisionTokenRange> ranges) noexcept {
    return std::any_of(ranges.begin(), ranges.end(), [&](const VisionTokenRange& range) {
        return range.begin < frontier && frontier < range.end;
    });
}

bool inside_exclusion(std::uint32_t frontier,
                      std::span<const pc::TapExclusion> exclusions) noexcept {
    return std::any_of(exclusions.begin(), exclusions.end(), [&](const pc::TapExclusion& span) {
        return span.begin < frontier && frontier < span.end;
    });
}

PrefixReusePath reuse_path_for(pc::SnapshotKind kind) noexcept {
    return kind == pc::SnapshotKind::Endpoint ? PrefixReusePath::Endpoint
                                              : PrefixReusePath::Snapshot;
}

} // namespace

// ---- construction ------------------------------------------------------------------------------

void ProgramImpl::create_hybrid_prefix_cache(const StartupObserver& observer) {
    // Every generation Program owns the cache; a disabled one only admits from the root and
    // retains nothing (no Host tier, snapshot slot or tap, and no lane publishes).
    if (causal_scoring) { return; }
    const DeviceKVPagePool& text_pool = text_kv_pages->physical_pool();
    const KVPageGeometry* backend_geometry =
        backend_kv_pages ? &backend_kv_pages->physical_pool().geometry() : nullptr;
    const HybridHostLayout host = plan_hybrid_host_layout(text_pool.geometry(), backend_geometry,
                                                          state_images->host_layout().image_bytes);
    const std::uint32_t slabs   = hybrid_host_slabs(host, context_cache.host_cache_bytes);
    const ContextCacheOptions& options = context_cache;
    if (!options.device_snapshot_slots || !options.max_new_taps || !options.tap_ladder_tokens ||
        !options.tap_min_gap_tokens || (options.enabled && *options.device_snapshot_slots == 0)) {
        throw std::logic_error("prefix cache options are not normalized");
    }
    const std::uint32_t device_slots = *options.device_snapshot_slots;
    pc::PrefixIndexConfig config;
    // Every node holds a Device page bundle or a Host slab.
    config.max_nodes             = text_pool.capacity_pages() + slabs + 1U;
    config.max_snapshots         = device_slots + slabs / host.image_slabs + 1U;
    config.host_slabs            = slabs;
    config.image_slabs           = host.image_slabs;
    config.device_snapshot_slots = device_slots;
    config.block_bytes           = host.block_payload_bytes;
    config.image_bytes           = host.image_bytes;
    config.host_blocks           = slabs != 0;
    config.cost                  = hybrid_cost_;
    config.cost.chunk_tokens     = prefill_chunk;
    const pc::TapPlannerConfig taps{
        .max_new_taps   = *options.max_new_taps,
        .ladder_tokens  = *options.tap_ladder_tokens,
        .min_gap_tokens = *options.tap_min_gap_tokens,
    };
    const std::uint64_t host_bytes = static_cast<std::uint64_t>(slabs) * host.slab_bytes;
    std::optional<StartupPhaseScope> phase;
    if (host_bytes != 0) {
        phase.emplace(observer, StartupPhase::HostCachePin, StartupProgressUnit::Bytes, host_bytes);
    }
    // Restores land in forward order: each layer's KV planes or recurrent state.
    const TextConfig& text = parameters.model.config().text;
    std::vector<HybridRestoreLayer> layers;
    layers.reserve(text.layer_types.size());
    for (std::size_t layer = 0; layer < text.layer_types.size(); ++layer) {
        layers.push_back(
            HybridRestoreLayer{.attention = text.layer_types[layer] == MixerKind::FullAttention,
                               .index     = text.compact_layer_indices[layer]});
    }
    hybrid_ = std::make_unique<HybridPrefixCache>(
        config, taps, *text_kv_pages, backend_kv_pages.get(), *state_store, *state_images, host,
        std::move(layers), phase ? &*phase : nullptr);
    if (phase) { phase->complete(host_bytes, host_bytes); }
}

void ProgramImpl::set_hybrid_cost(const pc::CacheCostModel& cost) {
    hybrid_cost_              = cost;
    hybrid_cost_.chunk_tokens = prefill_chunk;
    if (hybrid_) { hybrid_->set_cost(hybrid_cost_); }
}

namespace {

HybridCachePersistence public_result(const HybridPersistResult& result) {
    return HybridCachePersistence{
        .ok                  = result.ok,
        .message             = result.message,
        .blocks              = result.blocks,
        .snapshots           = result.snapshots,
        .bytes               = result.bytes,
        .seconds             = result.seconds,
        .saved_blocks        = result.saved_blocks,
        .saved_snapshots     = result.saved_snapshots,
        .required_host_bytes = result.required_host_bytes,
        .host_bytes          = result.host_bytes,
    };
}

} // namespace

HybridCachePersistence ProgramImpl::attach_hybrid_cache_file(const std::filesystem::path& path,
                                                             std::string fingerprint,
                                                             const StartupObserver& observer) {
    if (!hybrid_) { return {.message = "the hybrid prefix cache is not enabled"}; }
    if (path.empty()) { throw std::invalid_argument("hybrid prefix cache file path is empty"); }
    HybridCachePersistence loaded = public_result(hybrid_->load(path, fingerprint, observer));
    hybrid_file_                  = path;
    hybrid_fingerprint_           = std::move(fingerprint);
    return loaded;
}

void ProgramImpl::save_hybrid_cache_for_shutdown() noexcept {
    if (!hybrid_ || hybrid_file_.empty()) { return; }
    try {
        // Only Host-resident entries are saved, so every Host write must land before the slabs
        // are read. Lanes still pinning a path do not matter: the save only reads.
        device.synchronize();
        if (device.transfer_stream != nullptr) {
            CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
        }
        hybrid_->drain();
        hybrid_shutdown_save_ = public_result(hybrid_->save(hybrid_file_, hybrid_fingerprint_));
    } catch (const std::exception& error) {
        hybrid_shutdown_save_ = HybridCachePersistence{.message = error.what()};
    } catch (...) { hybrid_shutdown_save_ = HybridCachePersistence{.message = "unknown error"}; }
}

std::uint32_t ProgramImpl::hybrid_backend_frontier(std::uint32_t frontier) const noexcept {
    return speculative_backend == SpeculativeBackend::Mtp && frontier != 0 ? frontier - 1U
                                                                           : frontier;
}

// ---- admission ---------------------------------------------------------------------------------

HybridAdmissionQuote ProgramImpl::hybrid_quote(const PreparedPromptData& prompt,
                                               const RequestBasePlan& base,
                                               runtime::LaneId destination) {
    if (!hybrid_ || base.impl_ == nullptr) {
        throw std::logic_error("hybrid admission requires the hybrid prefix cache");
    }
    if (destination.value >= max_concurrency) {
        throw std::out_of_range("hybrid admission lane is out of range");
    }
    hybrid_->poll();
    const RequestBasePlanImpl& plan = *base.impl_;
    const auto n                    = static_cast<std::uint32_t>(prompt.token_ids.size());
    if (n == 0 || n != plan.summary.prompt_tokens) {
        throw std::logic_error("hybrid admission prompt does not match its base plan");
    }

    auto quote                         = std::make_shared<HybridQuoteImpl>();
    quote->summary                     = plan.summary;
    quote->sampling                    = plan.sampling;
    quote->text_kv_page_entitlement    = plan.text_kv_page_entitlement;
    quote->backend_kv_page_entitlement = plan.backend_kv_page_entitlement;
    quote->vision_control_plan         = plan.vision_control_plan;
    quote->destination                 = destination.value;
    quote->destination_epoch           = lane_epochs[destination.value];

    pc::PrefixCacheIndex& index = hybrid_->index();
    const bool backend_pool     = backend_kv_pages != nullptr;
    // Every unpinned Device-resident cache entry owns its page bundle exclusively, so evicting
    // one returns exactly one page to each pool. Pages this admission must newly take: the fork
    // entitlement beyond the shared full blocks, plus one per block or tail restored from Host.
    const auto fits = [&](const pc::MatchCandidate* candidate, std::span<const pc::NodeRef> path) {
        const std::uint32_t frontier = candidate != nullptr ? candidate->frontier : 0U;
        const std::uint32_t restored =
            candidate != nullptr ? candidate->host_only_blocks +
                                       (candidate->tail && !candidate->tail_on_device ? 1U : 0U)
                                 : 0U;
        // Entries this admission pins stop being evictable: its unpinned Device-resident path
        // blocks and the snapshot's Device-resident tail.
        std::uint32_t evictable_on_path = 0;
        for (const pc::NodeRef node : path) {
            const pc::NodeView view = index.node(node);
            if (view.pins == 0 && view.device == pc::CopyState::Resident) { ++evictable_on_path; }
        }
        if (candidate != nullptr && candidate->tail && candidate->tail_on_device &&
            index.snapshot(candidate->snapshot).pins == 0) {
            ++evictable_on_path;
        }
        const std::uint64_t evictable = index.device_evictable_blocks() - evictable_on_path;
        const std::uint64_t text_need = static_cast<std::uint64_t>(plan.text_kv_page_entitlement) -
                                        frontier / kBlock + restored;
        if (text_need > text_kv_pages->physical_pool().available_pages() + evictable) {
            return false;
        }
        if (backend_pool) {
            const std::uint64_t backend_need =
                static_cast<std::uint64_t>(plan.backend_kv_page_entitlement) -
                hybrid_backend_frontier(frontier) / kBlock + restored;
            if (backend_need > backend_kv_pages->physical_pool().available_pages() + evictable) {
                return false;
            }
        }
        return true;
    };

    std::optional<pc::MatchCandidate> selected;
    if (plan.allow_prefix_reuse && prompt.identity.reusable) {
        // Preparation computed the lookup keys once; every quote of this prompt reads them.
        if (prompt.block_hashes.size() != prompt.token_ids.size() / kBlock) {
            throw std::logic_error("prepared prompt carries no hybrid block keys");
        }
        pc::MatchResult match =
            index.match(prompt.token_ids, prompt.block_hashes, prompt.block_extras, n);
        const auto filling = [](const pc::MatchResult& result) {
            return std::any_of(
                result.candidates.begin(), result.candidates.end(),
                [](const pc::MatchCandidate& candidate) { return candidate.filling_blocks != 0; });
        };
        if (hybrid_->prefetch_landing() && filling(match)) {
            // A prefetch (§6.6) is copying blocks this prompt resumes over: they land within a
            // batch's copy time, and skipping them would pick a shallower source.
            hybrid_->settle_prefetch();
            match = index.match(prompt.token_ids, prompt.block_hashes, prompt.block_extras, n);
        }
        quote->cached_prefix_tokens = static_cast<std::uint32_t>(match.path.size()) * kBlock;
        const std::vector<VisionTokenRange> ranges =
            prompt.vision_items.empty() ? std::vector<VisionTokenRange>{} : vision_ranges(prompt);
        std::erase_if(match.candidates, [&](const pc::MatchCandidate& candidate) {
            return candidate.filling_blocks != 0 || inside_vision(candidate.frontier, ranges);
        });
        const pc::AdmissionChoice choice = index.choose(match, n);
        std::vector<std::size_t> order;
        if (choice.candidate) { order.push_back(*choice.candidate); }
        for (std::size_t i = 0; i < match.candidates.size(); ++i) {
            if (!choice.candidate || i != *choice.candidate) { order.push_back(i); }
        }
        for (const std::size_t i : order) {
            const pc::MatchCandidate& candidate = match.candidates[i];
            const std::span<const pc::NodeRef> path(match.path.data(), candidate.path_blocks);
            if (fits(&candidate, path)) {
                selected = candidate;
                break;
            }
        }
    }

    HybridAdmissionQuote out;
    out.destination = destination;
    if (selected) {
        quote->snapshot                       = selected->snapshot;
        quote->reuse_frontier                 = selected->frontier;
        quote->summary.reusable_prompt_tokens = selected->frontier;
        quote->summary.prefix_reuse_path = reuse_path_for(index.snapshot(selected->snapshot).kind);
    } else if (fits(nullptr, {})) {
        quote->summary.reusable_prompt_tokens = 0;
        quote->summary.prefix_reuse_path      = PrefixReusePath::Root;
    } else if (hybrid_->transfers_pending()) {
        // Blocks pinned only by their in-flight Host writes become evictable once those land.
        hybrid_->drain();
        return hybrid_quote(prompt, base, destination);
    } else {
        out.readiness = runtime::Readiness::TemporarilyBlocked;
        out.summary   = quote->summary;
        return out;
    }

    // Scheduler service units: one per nominal prefill chunk, one per Vision item use (which may
    // shorten a chunk) and one per further generated token.
    const std::uint32_t reuse  = quote->reuse_frontier;
    const std::uint32_t suffix = n - reuse;
    std::uint64_t units        = suffix == 0 ? 1ULL : 1ULL + (suffix - 1ULL) / prefill_chunk;
    if (plan.vision_control_plan) {
        for (const auto& item : plan.vision_control_plan->items) {
            if (item.token_end > reuse) { ++units; }
        }
    }
    if (quote->summary.effective_output_tokens != 0) {
        units += quote->summary.effective_output_tokens - 1ULL;
    }
    quote->summary.service_work_quanta = units;

    out.readiness = runtime::Readiness::Ready;
    out.summary   = quote->summary;
    out.impl      = std::move(quote);
    return out;
}

std::optional<std::uint32_t> ProgramImpl::hybrid_prefetch(const PreparedPromptData& prompt,
                                                          const RequestBasePlan& base) {
    if (!hybrid_ || !hybrid_->host_tier() || base.impl_ == nullptr ||
        !base.impl_->allow_prefix_reuse || !prompt.identity.reusable) {
        return 0U;
    }
    if (has_context_transaction() || hybrid_->restore_open()) { return std::nullopt; }
    hybrid_->poll();
    if (hybrid_->prefetch_landing()) { return std::nullopt; }
    const auto n = static_cast<std::uint32_t>(prompt.token_ids.size());
    if (n == 0 || prompt.block_hashes.size() != prompt.token_ids.size() / kBlock) { return 0U; }

    // The source the admission would choose now, as in hybrid_quote.
    pc::PrefixCacheIndex& index = hybrid_->index();
    pc::MatchResult match =
        index.match(prompt.token_ids, prompt.block_hashes, prompt.block_extras, n);
    const std::vector<VisionTokenRange> ranges =
        prompt.vision_items.empty() ? std::vector<VisionTokenRange>{} : vision_ranges(prompt);
    std::erase_if(match.candidates, [&](const pc::MatchCandidate& candidate) {
        return candidate.filling_blocks != 0 || inside_vision(candidate.frontier, ranges);
    });
    const pc::AdmissionChoice choice = index.choose(match, n);
    if (!choice.candidate || match.candidates[*choice.candidate].host_only_blocks == 0) {
        return 0U;
    }
    const std::span<const pc::NodeRef> path(match.path.data(),
                                            match.candidates[*choice.candidate].path_blocks);

    // Pinned while room is made, so no block of this path is evicted to hold another of them.
    // Room comes from free pages and host-backed cache, least recently used first: a prefetch
    // never waits for a transfer and never drops a block's last copy.
    index.acquire_path(path);
    std::uint32_t started = 0;
    try {
        DeviceKVPagePool& text_pool = text_kv_pages->physical_pool();
        DeviceKVPagePool* backend_pool =
            backend_kv_pages ? &backend_kv_pages->physical_pool() : nullptr;
        const auto available = [&] {
            const std::uint32_t text = text_pool.available_pages();
            return backend_pool != nullptr ? std::min(text, backend_pool->available_pages()) : text;
        };
        std::uint32_t wanted =
            std::min(match.candidates[*choice.candidate].host_only_blocks, kPrefetchBatchBlocks);
        while (available() < wanted && index.evict_backed_device_blocks(1) != 0) {}
        wanted = std::min(wanted, available());
        std::optional<DeviceKVPageReservation> text_pages;
        std::optional<DeviceKVPageReservation> backend_pages;
        if (wanted != 0) {
            text_pages = text_pool.reserve(wanted);
            if (backend_pool != nullptr) { backend_pages = backend_pool->reserve(wanted); }
        }
        if (text_pages && (backend_pool == nullptr || backend_pages)) {
            hybrid_->open_restore(device.stream);
            for (const pc::NodeRef node : path) {
                if (started == wanted) { break; }
                if (index.node(node).device != pc::CopyState::Absent) { continue; }
                HybridBlockPages pages{
                    .text = text_kv_pages->materialize_transfer_destination(*text_pages, kBlock)};
                if (backend_pool != nullptr) {
                    pages.backend =
                        backend_kv_pages->materialize_transfer_destination(*backend_pages, kBlock);
                }
                hybrid_->restore_block(node, pages);
                ++started;
            }
            hybrid_->submit_restore();
            hybrid_->detach_prefetch();
        }
    } catch (...) {
        hybrid_->abort_restore();
        index.release_path(path);
        throw;
    }
    index.release_path(path);
    if (started != 0) { advance_resource_revision(); }
    return started;
}

std::uint32_t ProgramImpl::hybrid_prefetch_room() const noexcept {
    if (!hybrid_ || !hybrid_->host_tier()) { return 0U; }
    const std::uint32_t cached = hybrid_->index().device_backed_evictable_blocks();
    std::uint32_t room         = text_kv_pages->physical_pool().available_pages() + cached;
    if (backend_kv_pages) {
        room = std::min(room, backend_kv_pages->physical_pool().available_pages() + cached);
    }
    return room;
}

bool ProgramImpl::hybrid_reservable(const HybridAdmissionQuote& quote,
                                    runtime::CancellationFlagView cancellation) const noexcept {
    if (!hybrid_ || has_context_transaction() || pending_transaction_ || !quote.impl ||
        quote.readiness != runtime::Readiness::Ready || cancellation.requested()) {
        return false;
    }
    const std::uint32_t lane = quote.impl->destination;
    return lane < max_concurrency && lane_epochs[lane] == quote.impl->destination_epoch &&
           requests[lane].lifecycle == Lifecycle::Empty &&
           active_continuations[lane] >= continuation_capacity;
}

runtime::ContextTransactionReserveStatus
ProgramImpl::hybrid_reserve_materialization(HybridAdmissionQuote&& quote,
                                            const PreparedPromptData& prompt,
                                            const std::function<PreparedPromptData()>& take_prompt,
                                            runtime::CancellationFlagView cancellation) {
    if (!hybrid_reservable(quote, cancellation)) {
        throw std::logic_error("hybrid admission quote is no longer reservable");
    }
    // Staging runs before the prompt is taken: a quote that went stale since it was computed
    // leaves the waiting request intact, and the Engine plans it again.
    HybridMaterializationTransaction transaction;
    transaction.quote = std::move(quote.impl);
    bool staged       = false;
    try {
        staged = hybrid_stage(transaction, prompt);
    } catch (...) {
        hybrid_abort_materialization(transaction);
        throw;
    }
    if (!staged) {
        hybrid_abort_materialization(transaction);
        return runtime::ContextTransactionReserveStatus::Aborted;
    }
    try {
        transaction.prompt = take_prompt();
        context_transaction_.emplace(std::move(transaction));
    } catch (...) {
        hybrid_abort_materialization(transaction);
        throw;
    }
    advance_resource_revision();
    return runtime::ContextTransactionReserveStatus::Reserved;
}

ContextTransactionProgress
ProgramImpl::progress_context_transaction(runtime::CancellationFlagView cancellation) {
    if (!context_transaction_) {
        throw std::logic_error("Program has no progressable context transaction");
    }
    MaterializationResult result = progress_hybrid_materialization(cancellation);
    if (result.status != runtime::ContextTransactionStatus::Published &&
        result.status != runtime::ContextTransactionStatus::Aborted) {
        throw std::logic_error("context transaction returned an invalid status");
    }
    return result;
}

void ProgramImpl::finalize_context_transaction() noexcept {
    if (context_transaction_ && context_transaction_->terminal) { context_transaction_.reset(); }
}

bool ProgramImpl::has_context_transaction() const noexcept {
    return context_transaction_.has_value();
}

MaterializationResult
ProgramImpl::progress_hybrid_materialization(runtime::CancellationFlagView cancellation) {
    HybridMaterializationTransaction& transaction = *context_transaction_;
    MaterializationResult out;
    const auto abort = [&] {
        hybrid_abort_materialization(transaction);
        context_transaction_.reset();
        advance_resource_revision();
        out.status = runtime::ContextTransactionStatus::Aborted;
    };
    try {
        if (!transaction.staged) {
            throw std::logic_error("hybrid admission was reserved without staging");
        }
        if (cancellation.requested()) {
            abort();
            return out;
        }
        // A Host restore does not hold the admission: the lane starts now and its Device work
        // queues behind the copies, activation behind the restore's prelude and each layer of
        // its first prefill pass behind that layer's copies (§6.5).
        if (hybrid_->restore_pending()) {
            transaction.restore_ticket = hybrid_->land_restore(device.stream);
        }
        out.diagnostics.cached_prefix_tokens = transaction.quote->cached_prefix_tokens;
        out.diagnostics.restored_host_bytes  = transaction.restore_bytes;
        out.published.emplace(hybrid_activate(transaction));
    } catch (...) {
        // The abort returns the state slot a landing restore may still be writing.
        if (transaction.restore_ticket != 0) {
            try {
                hybrid_->order_after_restore(transaction.restore_ticket, device.stream);
            } catch (...) {}
        }
        hybrid_abort_materialization(transaction);
        context_transaction_.reset();
        throw;
    }
    transaction.terminal = true;
    out.status           = runtime::ContextTransactionStatus::Published;
    return out;
}

bool ProgramImpl::hybrid_make_room(std::uint32_t text_pages, std::uint32_t backend_pages) {
    const DeviceKVPagePool& text_pool = text_kv_pages->physical_pool();
    const DeviceKVPagePool* backend_pool =
        backend_kv_pages ? &backend_kv_pages->physical_pool() : nullptr;
    while (text_pool.available_pages() < text_pages ||
           (backend_pool != nullptr && backend_pool->available_pages() < backend_pages)) {
        if (hybrid_->index().evict_device_blocks(1) != 0) { continue; }
        if (!hybrid_->transfers_pending()) { return false; }
        hybrid_->drain();
    }
    return true;
}

bool ProgramImpl::hybrid_stage(HybridMaterializationTransaction& transaction,
                               const PreparedPromptData& prompt) {
    const HybridQuoteImpl& quote = *transaction.quote;
    const std::uint32_t lane     = quote.destination;
    if (lane >= max_concurrency || lane_epochs[lane] != quote.destination_epoch ||
        requests[lane].lifecycle != Lifecycle::Empty ||
        active_continuations[lane] < continuation_capacity || requests[lane].prefill) {
        throw std::logic_error("hybrid admission destination is not free");
    }
    const auto n = static_cast<std::uint32_t>(prompt.token_ids.size());
    if (n != quote.summary.prompt_tokens) {
        throw std::logic_error("hybrid admission prompt changed after its quote");
    }
    hybrid_->poll();
    pc::PrefixCacheIndex& index = hybrid_->index();
    const std::uint32_t reuse   = quote.reuse_frontier;
    const std::uint32_t full    = reuse / kBlock;
    const std::uint32_t tail    = reuse % kBlock;
    if (prompt.block_hashes.size() != prompt.token_ids.size() / kBlock) {
        throw std::logic_error("prepared prompt carries no hybrid block keys");
    }
    transaction.hashes = prompt.block_hashes;
    transaction.extras = prompt.block_extras;

    std::vector<pc::NodeRef> path;
    pc::SnapshotView snapshot;
    if (reuse != 0) {
        if (!index.valid(quote.snapshot)) { return false; }
        snapshot = index.snapshot(quote.snapshot);
        const pc::MatchResult match =
            index.match(prompt.token_ids, transaction.hashes, transaction.extras, n);
        if (snapshot.frontier != reuse || match.path.size() < full ||
            (full != 0 ? !(snapshot.anchor == match.path[full - 1U]) : snapshot.anchor.valid())) {
            return false;
        }
        path.assign(match.path.begin(), match.path.begin() + full);
    }
    index.acquire_path(path);
    transaction.path   = std::move(path);
    transaction.staged = true;
    if (reuse != 0) {
        index.pin_snapshot(quote.snapshot);
        transaction.snapshot_pinned = true;
    }

    std::vector<pc::NodeRef> host_only;
    for (const pc::NodeRef node : transaction.path) {
        const pc::NodeView view = index.node(node);
        if (view.device == pc::CopyState::Absent) {
            host_only.push_back(node);
        } else if (view.device != pc::CopyState::Resident) {
            return false;
        }
    }
    const bool tail_restore  = tail != 0 && snapshot.tail_device_copy != pc::CopyState::Resident;
    const bool image_restore = reuse != 0 && snapshot.device_slot == pc::kNoId;
    const bool backend_pool  = backend_kv_pages != nullptr;
    const auto restored = static_cast<std::uint32_t>(host_only.size()) + (tail_restore ? 1U : 0U);
    const std::uint32_t text_need = quote.text_kv_page_entitlement - full + restored;
    const std::uint32_t backend_need =
        backend_pool
            ? quote.backend_kv_page_entitlement - hybrid_backend_frontier(reuse) / kBlock + restored
            : 0U;
    // The quote counted these pages; failing here means it went stale, and the Engine plans the
    // request again instead of failing it.
    if (!hybrid_make_room(text_need, backend_need)) { return false; }
    transaction.text_pages = text_kv_pages->physical_pool().reserve(text_need);
    if (!transaction.text_pages) { return false; }
    if (backend_pool) {
        transaction.backend_pages = backend_kv_pages->physical_pool().reserve(backend_need);
        if (!transaction.backend_pages) { return false; }
    }
    transaction.state = state_store->reserve_destination();
    if (!transaction.state) { return false; }
    if (restored == 0 && !image_restore) { return true; }

    // Host restores run on the restore stream while other lanes keep executing; the transaction
    // stays in progress until they complete.
    const std::uint64_t block_bytes = hybrid_->host_layout().block_payload_bytes;
    transaction.restore_bytes =
        static_cast<std::uint64_t>(restored) * block_bytes +
        (image_restore ? static_cast<std::uint64_t>(hybrid_->host_layout().image_bytes) : 0U);
    hybrid_->open_restore(device.stream);
    const auto destination = [&](std::uint32_t columns) {
        HybridBlockPages pages{.text = text_kv_pages->materialize_transfer_destination(
                                   *transaction.text_pages, columns)};
        if (backend_pool) {
            pages.backend = backend_kv_pages->materialize_transfer_destination(
                *transaction.backend_pages, columns);
        }
        return pages;
    };
    for (const pc::NodeRef node : host_only) { hybrid_->restore_block(node, destination(kBlock)); }
    if (tail_restore) { hybrid_->restore_tail(quote.snapshot, destination(tail)); }
    if (image_restore) {
        hybrid_->restore_image(quote.snapshot, state_store->physical_slot(*transaction.state));
        transaction.state_restored = true;
    }
    hybrid_->submit_restore();
    return true;
}

void ProgramImpl::hybrid_abort_materialization(
    HybridMaterializationTransaction& transaction) noexcept {
    if (!hybrid_) { return; }
    if (hybrid_->restore_open()) { hybrid_->abort_restore(); }
    transaction.text_pages.reset();
    transaction.backend_pages.reset();
    if (transaction.state) {
        (void)state_store->release(*transaction.state);
        transaction.state.reset();
    }
    try {
        pc::PrefixCacheIndex& index = hybrid_->index();
        if (transaction.snapshot_pinned && index.valid(transaction.quote->snapshot)) {
            index.unpin_snapshot(transaction.quote->snapshot);
        }
        transaction.snapshot_pinned = false;
        if (!transaction.path.empty()) { index.release_path(transaction.path); }
        transaction.path.clear();
    } catch (...) { std::terminate(); }
}

StartResult ProgramImpl::hybrid_activate(HybridMaterializationTransaction& transaction) {
    const HybridQuoteImpl& quote = *transaction.quote;
    PreparedPromptData& prompt   = transaction.prompt;
    const std::uint32_t lane     = quote.destination;
    if (lane >= max_concurrency || lane_epochs[lane] != quote.destination_epoch ||
        requests[lane].lifecycle != Lifecycle::Empty ||
        active_continuations[lane] < continuation_capacity || requests[lane].prefill ||
        !transaction.state) {
        throw std::logic_error("hybrid admission destination is not free");
    }
    const auto started          = Clock::now();
    RequestControl& request     = requests[lane];
    const auto n                = static_cast<std::uint32_t>(prompt.token_ids.size());
    pc::PrefixCacheIndex& index = hybrid_->index();
    const std::uint32_t reuse   = quote.reuse_frontier;
    const std::uint32_t full    = reuse / kBlock;
    const std::uint32_t tail    = reuse % kBlock;
    const std::span<const pc::NodeRef> path(transaction.path);
    const pc::SnapshotView snapshot =
        reuse != 0 ? index.snapshot(quote.snapshot) : pc::SnapshotView{};
    if (reuse != 0 && tail != 0 && snapshot.tail_device_copy == pc::CopyState::Absent) {
        throw std::logic_error("hybrid admission tail is not on the Device");
    }

    bool continuation_bound = false;
    try {
        const bool backend_pool = backend_kv_addresses && quote.backend_kv_page_entitlement != 0;
        const std::uint32_t backend_reuse = hybrid_backend_frontier(reuse);
        const std::uint32_t backend_full  = backend_reuse / kBlock;
        const std::uint32_t backend_tail  = backend_reuse % kBlock;
        // The staged reservations return to the pools here and the fork takes them back at once.
        transaction.text_pages.reset();
        transaction.backend_pages.reset();

        const std::optional<std::uint32_t> continuation = allocate_continuation_slot();
        if (!continuation) { throw std::logic_error("hybrid admission has no continuation slot"); }
        active_continuations[lane] = *continuation;
        continuation_bound         = true;
        SequenceState& sequence    = continuation_states[*continuation];
        sequence.lane              = lane;

        // State: a zero image, a Device snapshot copy, or the Host restore already landed in the
        // destination.
        const StateImageHandle state = *transaction.state;
        transaction.state.reset();
        sequence.state = ActiveStateBinding{.read = state, .write = state};
        if (reuse == 0) {
            state_store->activate_reset(state, device.stream);
        } else if (transaction.state_restored) {
            state_store->activate_copied(state);
        } else {
            const StateImageHandle image = hybrid_->image(quote.snapshot);
            state_images->copy_slot(state_store->physical_slot(image),
                                    state_store->physical_slot(state), device.stream);
            state_store->activate_copied(state);
        }

        // KV: cached full pages are shared; a partial tail is copied into a private page.
        const auto activate = [&](KVAddressSpaceStore& addresses, LogicalKVPageStore& pages,
                                  std::uint32_t frontier, std::uint32_t full_pages,
                                  std::uint32_t tail_columns, std::uint32_t entitlement,
                                  bool backend) -> KVAddressSpaceHandle {
            const std::optional<KVAddressSpaceHandle> address = addresses.create_inactive();
            if (!address) { throw std::logic_error("hybrid admission has no KV address space"); }
            try {
                if (frontier == 0) {
                    addresses.activate(*address, entitlement, static_cast<std::int32_t>(lane));
                    return *address;
                }
                std::vector<LogicalKVPageHandle> shared;
                shared.reserve(full_pages);
                for (std::uint32_t page = 0; page < full_pages; ++page) {
                    const HybridBlockPages& block =
                        hybrid_->block(index.node(path[page]).device_id);
                    shared.push_back(backend ? *block.backend : block.text);
                }
                std::optional<LogicalKVPageHandle> tail_source;
                if (tail_columns != 0) {
                    const HybridBlockPages& block =
                        full_pages < full ? hybrid_->block(index.node(path[full_pages]).device_id)
                                          : hybrid_->block(snapshot.tail_device);
                    tail_source = backend ? *block.backend : block.text;
                }
                KVPagePrefixForkReservation fork = addresses.prepare_page_prefix_fork(
                    *address, shared, tail_source, tail_columns, entitlement,
                    static_cast<std::int32_t>(lane));
                if (fork.needs_tail_copy()) {
                    pages.physical_pool().copy_page(
                        addresses.page_prefix_fork_tail_source(fork),
                        addresses.page_prefix_fork_tail_destination(fork), device.stream);
                }
                addresses.commit_page_prefix_fork(std::move(fork), device.stream);
                return *address;
            } catch (...) {
                if (addresses.active(*address)) { addresses.deactivate(*address); }
                (void)addresses.release(*address);
                throw;
            }
        };
        SequenceKVBundle bundle{.text = activate(*text_kv_addresses, *text_kv_pages, reuse, full,
                                                 tail, quote.text_kv_page_entitlement, false)};
        sequence.kv.emplace(bundle);
        if (backend_pool) {
            sequence.kv->backend =
                activate(*backend_kv_addresses, *backend_kv_pages, backend_reuse, backend_full,
                         backend_tail, quote.backend_kv_page_entitlement, true);
        }

        const bool prepare_mtp = speculative_backend == SpeculativeBackend::Mtp;
        if (reuse == 0) {
            ordered_reset(sequence);
        } else {
            refresh_state_views(sequence);
            sequence.text_kv_valid = reuse;
            sequence.mtp_kv_valid  = prepare_mtp ? reuse - 1U : 0U;
            sequence.dflash_context_frontier =
                is_masked_draft_backend(speculative_backend) ? reuse : 0U;
        }
        sequence.endpoint_valid    = false;
        sequence.tail_hidden_valid = false;
        sequence.ledger.assign(prompt.token_ids.begin(), prompt.token_ids.end());
        bind_sequence_kv(sequence);

        const std::uint32_t initial_mtp_extent =
            prepare_mtp ? std::min({draft_window,
                                    quote.summary.effective_output_tokens > 1
                                        ? quote.summary.effective_output_tokens - 2
                                        : 0U,
                                    capacity - n > 0 ? capacity - n - 1 : 0U})
                        : 0U;
        const std::uint32_t backend_materialized =
            speculative_backend == SpeculativeBackend::Mtp
                ? std::min(capacity, n + (initial_mtp_extent == 0 ? 0U : initial_mtp_extent - 1U))
            : speculative_backend == SpeculativeBackend::DFlash ? n
                                                                : 0U;
        ensure_sequence_kv_mapped(sequence, n, backend_materialized);
        install_sampling(sequence, request, quote.sampling);
        sequence.rope_delta = prompt.rope_delta;
        set_device_i32(io.rope_delta, sequence.rope_delta);

        request.timings                     = {};
        request.pending                     = {};
        request.publish_continuation        = quote.summary.publish_continuation;
        request.text_kv_page_entitlement    = quote.text_kv_page_entitlement;
        request.backend_kv_page_entitlement = quote.backend_kv_page_entitlement;

        std::optional<VisionPrefillPlan> vision;
        if (quote.vision_control_plan) {
            VisionPrefillPlan plan;
            plan.control_plan      = quote.vision_control_plan;
            std::size_t max_merged = 0;
            for (std::size_t index_item = 0; index_item < quote.vision_control_plan->items.size();
                 ++index_item) {
                const qwen3_5::VisionItemControlPlan& item =
                    quote.vision_control_plan->items[index_item];
                if (item.token_end <= reuse) { continue; }
                const std::uint32_t begin =
                    prepare_mtp && item.token_begin != 0 ? item.token_begin - 1 : item.token_begin;
                plan.uses.push_back(VisionUseSpan{
                    .begin               = begin,
                    .end                 = item.token_end,
                    .prepared_item_index = static_cast<std::uint32_t>(index_item),
                });
                max_merged = std::max(max_merged, item.merged_count);
            }
            if (!plan.uses.empty()) {
                plan.max_merged_count = max_merged;
                std::vector<bool> used(prompt.media_payloads.size(), false);
                for (const VisionUseSpan& use : plan.uses) { used[use.prepared_item_index] = true; }
                for (std::size_t item = 0; item < used.size(); ++item) {
                    if (!used[item]) { prompt.media_payloads[item].reset(); }
                }
                const std::uint32_t first_item = plan.uses.front().prepared_item_index;
                auto control                   = std::make_shared<qwen3_5::VisionControl>(
                    qwen3_5::build_vision_control(prompt, *plan.control_plan, first_item));
                for (VisionUseSpan& use : plan.uses) {
                    use.control_index = use.prepared_item_index - first_item;
                }
                plan.control = std::move(control);
                plan.control_plan.reset();
                vision = std::move(plan);
            }
        }
        if (prompt.has_media() && !vision) { prompt.release_all_media_payloads(); }

        // Taps are planned before the prompt moves into the prefill state.
        const std::vector<VisionTokenRange> ranges = vision_ranges(prompt);
        const bool publish = quote.summary.publish_continuation && prompt.identity.reusable;
        std::vector<pc::TapExclusion> exclusions;
        exclusions.reserve(ranges.size());
        for (const VisionTokenRange& range : ranges) {
            exclusions.push_back(pc::TapExclusion{range.begin, range.end});
        }
        std::vector<pc::PlannedTap> taps;
        if (publish) {
            // A request that resumed from a previous generation's endpoint proves its client
            // echoes generated turns token for token (agent loops, preserved reasoning): its own
            // endpoint will serve the next turn, so the generation opener is not worth the prefill
            // split an exact tap costs. Otherwise the history is re-rendered (for example with
            // reasoning stripped) and the next turn diverges at the opener.
            std::vector<pc::TapHint> hints = prompt.tap_hints.hints;
            if (reuse != 0 && snapshot.kind == pc::SnapshotKind::Endpoint) {
                std::erase_if(hints, [](const pc::TapHint& hint) {
                    return hint.kind == pc::TapHintKind::GenerationOpener;
                });
            }
            const std::array<std::uint32_t, 1> existing{reuse};
            taps = pc::plan_taps(n, reuse, hints,
                                 reuse != 0 ? std::span<const std::uint32_t>(existing)
                                            : std::span<const std::uint32_t>(),
                                 exclusions, hybrid_->taps());
        }
        const std::uint64_t path_hash =
            full == 0 ? pc::kRootLookupHash : transaction.hashes[full - 1U];
        const std::uint64_t trailing = all_vision_key(ranges);

        RequestControl::Prefill prefill{
            .prompt             = std::move(prompt),
            .vision_plan        = std::move(vision),
            .vision             = nullptr,
            .base               = reuse,
            .cursor             = reuse,
            .prompt_tokens      = n,
            .initial_mtp_extent = initial_mtp_extent,
            .elapsed_seconds    = 0.0,
            .prepare_mtp        = prepare_mtp,
            .reuse              = quote.summary.prefix_reuse_path,
            .mtp_bridge =
                prepare_mtp && reuse != 0 ? MtpBridgeMode::BeforeSuffix : MtpBridgeMode::None,
        };
        request.prefill.emplace(std::move(prefill));
        if (request.prefill->vision_plan) {
            if (!workspace_plan.vision) {
                throw std::logic_error("Vision prefill has no startup workspace plan");
            }
            request.prefill->vision = std::make_unique<execution::VisionPrefillSession>(
                device, parameters,
                DeviceSpan{workspace_storage.base(), workspace_storage.capacity()},
                *workspace_plan.vision, request.prefill->prompt, *request.prefill->vision_plan,
                vision_handoff_peak_bytes);
        }

        if (is_masked_draft_backend(speculative_backend)) {
            if (!dflash || !io.dflash_decode || (backend_kv_cache() && !sequence.kv->backend)) {
                throw std::logic_error("DFlash prefill state is incomplete");
            }
            *dflash_host_ingress                            = {};
            dflash_host_ingress->active_lanes[0]            = static_cast<std::int32_t>(lane);
            const StateImageSelectors selectors             = state_selectors(sequence);
            dflash_host_ingress->state_source_slots[0]      = selectors.source;
            dflash_host_ingress->state_destination_slots[0] = selectors.destination;
            dflash_host_ingress->dflash_kv_table_rows[0] =
                sequence.kv->backend ? backend_kv_addresses->bound_row(*sequence.kv->backend) : 0;
            CUDA_CHECK(cudaMemcpyAsync(io.dflash_decode->ingress.data, dflash_host_ingress,
                                       sizeof(qwen3_5::DFlashDecodeIngress), cudaMemcpyHostToDevice,
                                       device.stream));
        }

        // The lane takes over the staged path pins; nothing below throws.
        HybridLaneState& lane_state = hybrid_lanes_[lane];
        lane_state                  = HybridLaneState{};
        lane_state.taps.swap(taps);
        lane_state.exclusions.swap(exclusions);
        lane_state.prompt_extras.swap(transaction.extras);
        lane_state.path.swap(transaction.path);
        lane_state.active                 = true;
        lane_state.publish                = publish;
        lane_state.path_hash              = path_hash;
        lane_state.trailing_extra         = trailing;
        lane_state.deepest_snapshot       = reuse;
        lane_state.resume_snapshot        = reuse != 0 ? quote.snapshot : pc::SnapshotRef{};
        lane_state.resume_frontier        = reuse;
        lane_state.last_capture           = reuse;
        lane_state.restore_ticket         = transaction.restore_ticket;
        lane_state.restore_layers_pending = transaction.restore_ticket != 0;

        request.prefill->elapsed_seconds =
            std::chrono::duration<double>(Clock::now() - started).count();
        request.lifecycle = Lifecycle::Prefilling;

        HybridCacheCounters& counters = hybrid_->counters();
        ++counters.admissions;
        if (reuse != 0) {
            index.note_hit(quote.snapshot);
            ++counters.snapshot_hits;
            counters.reused_tokens += reuse;
            index.unpin_snapshot(quote.snapshot);
            transaction.snapshot_pinned = false;
        }
        invalidate_lane(lane);
        advance_resource_revision();
        return StartResult{
            .sequence =
                ContractAccess::make_sequence(this, runtime::LaneId{lane}, lane_epochs[lane]),
        };
    } catch (...) {
        if (continuation_bound) { clear_lane_best_effort(active_sequence(lane), request); }
        invalidate_lane(lane);
        throw;
    }
}

// ---- publication -------------------------------------------------------------------------------

void ProgramImpl::hybrid_publish_blocks(SequenceState& sequence) {
    if (!hybrid_ || sequence.lane >= max_concurrency) { return; }
    HybridLaneState& lane = hybrid_lanes_[sequence.lane];
    if (!lane.active || !lane.publish || !sequence.kv) { return; }
    hybrid_->poll();
    std::uint32_t frontier = sequence.text_kv_valid;
    // Every enabled pool must hold the whole block. The DFlash2 draft ring is StateImage content,
    // not block content, so it does not bound block publication.
    if (sequence.kv->backend) { frontier = std::min(frontier, backend_kv_valid(sequence)); }
    frontier =
        std::min<std::uint32_t>(frontier, static_cast<std::uint32_t>(sequence.ledger.size()));
    const std::uint32_t full = frontier / kBlock;
    while (lane.path.size() < full) {
        const auto block = static_cast<std::uint32_t>(lane.path.size());
        const std::span<const TokenId> tokens(
            sequence.ledger.data() + static_cast<std::size_t>(block) * kBlock, kBlock);
        const std::uint64_t extra =
            block < lane.prompt_extras.size() ? lane.prompt_extras[block] : lane.trailing_extra;
        const std::uint64_t hash = pc::block_lookup_hash(lane.path_hash, tokens, extra);
        HybridBlockPages pages{.text = text_kv_addresses->logical_page(sequence.kv->text, block)};
        if (sequence.kv->backend) {
            pages.backend = backend_kv_addresses->logical_page(*sequence.kv->backend, block);
        }
        const pc::NodeRef parent      = lane.path.empty() ? pc::NodeRef{} : lane.path.back();
        const pc::InsertResult result = hybrid_->insert_block(parent, hash, tokens, extra, pages);
        lane.path.push_back(result.node);
        lane.path_hash = hash;
    }
    if (!lane.pending.empty()) { hybrid_publish_pending(sequence, false); }
}

std::optional<std::uint32_t> ProgramImpl::hybrid_copy_tail(const HybridBlockPages& source,
                                                           std::uint32_t columns) {
    const bool backend_pool = backend_kv_pages != nullptr;
    if (!hybrid_make_room(1U, backend_pool ? 1U : 0U)) { return std::nullopt; }
    std::optional<DeviceKVPageReservation> text_reservation =
        text_kv_pages->physical_pool().reserve(1U);
    std::optional<DeviceKVPageReservation> backend_reservation;
    if (!text_reservation) { return std::nullopt; }
    if (backend_pool) {
        backend_reservation = backend_kv_pages->physical_pool().reserve(1U);
        if (!backend_reservation) { return std::nullopt; }
    }
    HybridBlockPages copy{
        .text = text_kv_pages->materialize_transfer_destination(*text_reservation, columns)};
    if (backend_pool) {
        copy.backend =
            backend_kv_pages->materialize_transfer_destination(*backend_reservation, columns);
    }
    // Each tail owns its bundle, so evicting any cache entry returns exactly one page per pool.
    text_kv_pages->physical_pool().copy_page(text_kv_pages->physical(source.text),
                                             text_kv_pages->physical(copy.text), device.stream);
    if (backend_pool) {
        backend_kv_pages->physical_pool().copy_page(backend_kv_pages->physical(*source.backend),
                                                    backend_kv_pages->physical(*copy.backend),
                                                    device.stream);
    }
    return hybrid_->register_tail_copy(copy, columns);
}

void ProgramImpl::hybrid_publish_pending(SequenceState& sequence, bool finishing) {
    HybridLaneState& lane         = hybrid_lanes_[sequence.lane];
    pc::PrefixCacheIndex& index   = hybrid_->index();
    HybridCacheCounters& counters = hybrid_->counters();
    const auto drop               = [&](const HybridPendingTap& tap) {
        (void)state_store->release(tap.image);
        index.release_device_slot(tap.slot);
        ++counters.taps_skipped;
    };
    while (!lane.pending.empty()) {
        const HybridPendingTap tap  = lane.pending.front();
        const std::uint32_t full    = tap.frontier / kBlock;
        const std::uint32_t tail    = tap.frontier % kBlock;
        const bool anchored         = lane.path.size() >= full;
        const bool tail_block_ready = tail == 0 || lane.path.size() > full;
        std::optional<HybridBlockPages> tail_source;
        if (anchored && tail != 0) {
            if (tail_block_ready) {
                tail_source = hybrid_->block(index.node(lane.path[full]).device_id);
            } else if (finishing && sequence.kv && sequence.text_kv_valid >= tap.frontier &&
                       (!sequence.kv->backend ||
                        backend_kv_valid(sequence) >= hybrid_backend_frontier(tap.frontier))) {
                // The finishing lane's own partial page holds the tail; it is copied before the
                // lane releases it.
                HybridBlockPages own{.text =
                                         text_kv_addresses->logical_page(sequence.kv->text, full)};
                if (sequence.kv->backend) {
                    own.backend = backend_kv_addresses->logical_page(*sequence.kv->backend, full);
                }
                tail_source = own;
            }
        }
        if (!anchored || (tail != 0 && !tail_source)) {
            if (!finishing) { return; }
            lane.pending.erase(lane.pending.begin());
            drop(tap);
            continue;
        }
        lane.pending.erase(lane.pending.begin());
        std::optional<std::uint32_t> tail_id;
        if (tail != 0) {
            try {
                tail_id = hybrid_copy_tail(*tail_source, tail);
            } catch (...) {
                drop(tap);
                throw;
            }
            if (!tail_id) {
                drop(tap);
                continue;
            }
        }
        const pc::NodeRef anchor = full == 0 ? pc::NodeRef{} : lane.path[full - 1U];
        const std::span<const TokenId> tail_tokens(
            sequence.ledger.data() + static_cast<std::size_t>(full) * kBlock, tail);
        // publish_snapshot consumes the staging slot and the tail id on every outcome.
        const pc::PublishResult published = index.publish_snapshot(
            anchor, tap.frontier, tail_tokens, tail_id, tap.slot,
            tap.boundary ? pc::SnapshotKind::Boundary : pc::SnapshotKind::Tap);
        if (!published.created) {
            (void)state_store->release(tap.image);
            ++counters.taps_skipped;
            continue;
        }
        hybrid_->attach_image(published.snapshot, tap.image);
        lane.deepest_snapshot = std::max(lane.deepest_snapshot, tap.frontier);
        ++counters.taps_created;
        if (!tap.boundary) {
            // A previous tap still pending when this one was captured is superseded here, before
            // the Host write below can take its slabs.
            hybrid_supersede_tap(lane, tap.frontier);
            lane.tap_snapshot = published.snapshot;
            lane.tap_frontier = tap.frontier;
        }
        (void)hybrid_->start_snapshot_host_write(published.snapshot, device.stream,
                                                 device.transfer_stream);
    }
}

void ProgramImpl::hybrid_capture_tap(SequenceState& sequence, std::uint32_t frontier,
                                     bool boundary) {
    HybridLaneState& lane         = hybrid_lanes_[sequence.lane];
    pc::PrefixCacheIndex& index   = hybrid_->index();
    HybridCacheCounters& counters = hybrid_->counters();
    if (sequence.state.fork_pending || frontier == 0 || frontier > sequence.text_kv_valid) {
        ++counters.taps_skipped;
        return;
    }
    hybrid_supersede_resume(lane, frontier);
    hybrid_supersede_tap(lane, frontier);
    const std::optional<std::uint32_t> slot = index.acquire_device_slot(
        index.estimate_priority(lane.last_capture, frontier, frontier % kBlock != 0));
    if (!slot) {
        ++counters.taps_skipped;
        return;
    }
    const std::optional<StateImageHandle> image = state_store->reserve_destination();
    if (!image) {
        index.release_device_slot(*slot);
        ++counters.taps_skipped;
        return;
    }
    try {
        state_images->copy_slot(state_store->physical_slot(sequence.state.write),
                                state_store->physical_slot(*image), device.stream);
        state_store->publish_copied_checkpoint(*image);
        lane.pending.push_back(HybridPendingTap{
            .frontier = frontier, .image = *image, .slot = *slot, .boundary = boundary});
    } catch (...) {
        (void)state_store->release(*image);
        index.release_device_slot(*slot);
        throw;
    }
    lane.last_capture = frontier;
    hybrid_publish_pending(sequence, false);
}

void ProgramImpl::hybrid_after_prefill_chunk(SequenceState& sequence, std::uint32_t cursor,
                                             std::uint32_t prompt_tokens) {
    if (!hybrid_ || sequence.lane >= max_concurrency) { return; }
    HybridLaneState& lane = hybrid_lanes_[sequence.lane];
    if (!lane.active || !lane.publish || cursor >= prompt_tokens) { return; }
    // The next chunk reaches the prompt end, so a flexible tap inside it is realized here.
    const bool final_chunk_next = prompt_tokens - cursor <= prefill_chunk;
    bool exact                  = false;
    bool flexible               = false;
    bool boundary               = false;
    while (lane.next_tap < lane.taps.size()) {
        const pc::PlannedTap& tap = lane.taps[lane.next_tap];
        if (tap.placement == pc::TapPlacement::Exact) {
            if (tap.position > cursor) { break; }
            exact    = exact || tap.position == cursor;
            boundary = boundary || (tap.position == cursor && tap.boundary);
        } else if (tap.position > cursor && !final_chunk_next) {
            break;
        } else {
            flexible = true;
        }
        ++lane.next_tap;
    }
    if (!exact && !flexible) { return; }
    if (inside_exclusion(cursor, lane.exclusions) || cursor <= lane.last_capture ||
        (!exact && cursor - lane.last_capture < pc::kMinimumTapSeparation)) {
        ++hybrid_->counters().taps_skipped;
        return;
    }
    hybrid_capture_tap(sequence, cursor, boundary);
}

void ProgramImpl::hybrid_supersede_resume(HybridLaneState& lane, std::uint32_t frontier) {
    pc::PrefixCacheIndex& index = hybrid_->index();
    if (!lane.resume_snapshot.valid() || frontier <= lane.resume_frontier ||
        !index.valid(lane.resume_snapshot)) {
        return;
    }
    index.supersede(lane.resume_snapshot);
    lane.resume_snapshot = {};
}

void ProgramImpl::hybrid_supersede_tap(HybridLaneState& lane, std::uint32_t frontier) {
    pc::PrefixCacheIndex& index = hybrid_->index();
    if (!lane.tap_snapshot.valid() || frontier <= lane.tap_frontier ||
        !index.valid(lane.tap_snapshot)) {
        return;
    }
    index.supersede(lane.tap_snapshot);
    lane.tap_snapshot = {};
}

std::span<const cudaEvent_t> ProgramImpl::hybrid_take_restore_layers(std::uint32_t lane) {
    if (!hybrid_ || lane >= max_concurrency) { return {}; }
    HybridLaneState& state = hybrid_lanes_[lane];
    if (!std::exchange(state.restore_layers_pending, false)) { return {}; }
    return hybrid_->restore_layer_events(state.restore_ticket);
}

void ProgramImpl::hybrid_release_lane(std::uint32_t lane) noexcept {
    if (!hybrid_ || lane >= max_concurrency) { return; }
    HybridLaneState& state = hybrid_lanes_[lane];
    try {
        // The state slot the lane returns may still be a destination of its restore (a lane can
        // end before its first prefill pass waits for the copies), so later Device work waits
        // for whatever of the restore is still landing.
        if (state.restore_ticket != 0) {
            hybrid_->order_after_restore(state.restore_ticket, device.stream);
        }
        pc::PrefixCacheIndex& index = hybrid_->index();
        for (const HybridPendingTap& tap : state.pending) {
            (void)state_store->release(tap.image);
            index.release_device_slot(tap.slot);
        }
        state.pending.clear();
        if (state.active) {
            // The lane's blocks become evictable once its pins go: back them on the Host tier
            // first, in one batch, so eviction drops Device copies instead of losing nodes.
            try {
                hybrid_->start_block_host_writes(state.path, device.stream, device.transfer_stream);
            } catch (...) {}
            index.release_path(state.path);
        }
    } catch (...) { std::terminate(); }
    state = HybridLaneState{};
}

bool ProgramImpl::hybrid_finish_lane(SequenceState& sequence, RequestControl& request,
                                     std::uint32_t lane, bool endpoint) noexcept {
    HybridLaneState& lane_state = hybrid_lanes_[lane];
    bool image_moved            = false;
    try {
        if (lane_state.active && lane_state.publish && sequence.kv) {
            hybrid_publish_blocks(sequence);
            hybrid_publish_pending(sequence, true);
            const std::uint32_t frontier         = sequence.text_kv_valid;
            const std::uint32_t backend_frontier = hybrid_backend_frontier(frontier);
            const bool backend_caught_up =
                (!sequence.kv->backend || backend_kv_valid(sequence) >= backend_frontier) &&
                (!is_masked_draft_backend(speculative_backend) ||
                 sequence.dflash_context_frontier == frontier);
            const std::uint32_t anchor_blocks = frontier / kBlock;
            // Moving the endpoint image into the index commits this lane to the strict KV release
            // below, which cannot report failure; a release that would fail keeps the image here
            // so the ordinary strict clear can refuse it and the Engine recovers.
            const bool kv_releasable =
                text_kv_addresses->can_release_after_deactivate(sequence.kv->text) &&
                (!sequence.kv->backend ||
                 backend_kv_addresses->can_release_after_deactivate(*sequence.kv->backend));
            if (endpoint && kv_releasable && backend_caught_up && frontier != 0 &&
                frontier >= lane_state.deepest_snapshot + pc::kMinimumTapSeparation &&
                frontier <= sequence.ledger.size() && lane_state.path.size() >= anchor_blocks &&
                !sequence.state.fork_pending && sequence.state.read == sequence.state.write &&
                state_store->role(sequence.state.write) == StateImageRole::ActiveMutable) {
                pc::PrefixCacheIndex& index = hybrid_->index();
                // The lineage moves past its source here, so a slot it needs comes from its own
                // lineage before another conversation's snapshot.
                hybrid_supersede_resume(lane_state, frontier);
                const std::optional<std::uint32_t> slot =
                    index.acquire_device_slot(index.estimate_priority(
                        lane_state.deepest_snapshot, frontier, frontier % kBlock != 0));
                if (slot) {
                    const std::uint32_t tail = frontier % kBlock;
                    std::optional<std::uint32_t> tail_id;
                    try {
                        if (tail != 0) {
                            // The finishing lane hands its last partial page to the snapshot.
                            HybridBlockPages pages{.text = text_kv_addresses->logical_page(
                                                       sequence.kv->text, anchor_blocks)};
                            std::uint32_t backend_columns = 0;
                            if (sequence.kv->backend) {
                                pages.backend = backend_kv_addresses->logical_page(
                                    *sequence.kv->backend, anchor_blocks);
                                backend_columns = backend_frontier > anchor_blocks * kBlock
                                                      ? backend_frontier - anchor_blocks * kBlock
                                                      : 0U;
                            }
                            tail_id = hybrid_->register_tail(pages, tail, backend_columns);
                        }
                    } catch (...) {
                        index.release_device_slot(*slot);
                        throw;
                    }
                    const StateImageHandle image = sequence.state.write;
                    state_store->freeze(image);
                    const std::span<const TokenId> tail_tokens(
                        sequence.ledger.data() + static_cast<std::size_t>(anchor_blocks) * kBlock,
                        tail);
                    const pc::NodeRef anchor =
                        anchor_blocks == 0 ? pc::NodeRef{} : lane_state.path[anchor_blocks - 1U];
                    const pc::PublishResult published = index.publish_snapshot(
                        anchor, frontier, tail_tokens, tail_id, *slot, pc::SnapshotKind::Endpoint);
                    if (published.snapshot.valid()) {
                        lane_state.deepest_snapshot =
                            std::max(lane_state.deepest_snapshot, frontier);
                    }
                    if (published.created) {
                        hybrid_->attach_image(published.snapshot, image);
                        image_moved    = true;
                        sequence.state = {};
                        ++hybrid_->counters().endpoints_created;
                        // Superseded first, so the Host write below can take its slabs.
                        hybrid_supersede_resume(lane_state, lane_state.deepest_snapshot);
                        (void)hybrid_->start_snapshot_host_write(published.snapshot, device.stream,
                                                                 device.transfer_stream);
                    } else {
                        state_store->thaw(image);
                    }
                }
            }
            hybrid_supersede_resume(lane_state, lane_state.deepest_snapshot);
        }
    } catch (...) {
        // Terminal publication is optional; a failure keeps whatever was published and releases
        // the lane normally.
    }
    hybrid_release_lane(lane);
    if (!image_moved) { return clear_lane_strict(sequence, request); }
    // The endpoint StateImage now belongs to the index; release the rest of the lane. The release
    // was checked before the image moved, so its strict invariants hold here.
    release_active_sequence_kv_strict(sequence);
    retire_continuation_slot(static_cast<std::uint32_t>(&sequence - continuation_states.data()));
    request.retire();
    return true;
}

// ---- Device pressure and statistics ------------------------------------------------------------

HybridPrefixCacheStats ProgramImpl::hybrid_stats() const noexcept {
    HybridPrefixCacheStats out;
    if (!hybrid_) { return out; }
    const pc::PrefixIndexStats index    = hybrid_->index().stats();
    const HybridCacheCounters& counters = hybrid_->counters();
    const pc::PrefixIndexConfig& config = hybrid_->index().config();
    out.nodes                           = index.nodes;
    out.snapshots                       = index.snapshots;
    out.device_resident_blocks          = index.device_resident_blocks;
    out.device_evictable_blocks         = index.device_evictable_blocks;
    out.host_slabs                      = config.host_slabs;
    out.host_free_slabs                 = index.host_free_slabs;
    out.host_slab_bytes                 = hybrid_->host_layout().slab_bytes;
    out.free_device_snapshot_slots      = index.free_device_slots;
    out.admissions                      = counters.admissions;
    out.snapshot_hits                   = counters.snapshot_hits;
    out.reused_tokens                   = counters.reused_tokens;
    out.blocks_inserted                 = counters.blocks_inserted;
    out.blocks_reattached               = counters.blocks_reattached;
    out.blocks_duplicate                = counters.blocks_duplicate;
    out.taps_created                    = counters.taps_created;
    out.taps_skipped                    = counters.taps_skipped;
    out.endpoints_created               = counters.endpoints_created;
    out.host_image_writes               = counters.host_image_writes;
    out.host_block_writes               = counters.host_block_writes;
    out.host_image_restores             = counters.host_image_restores;
    out.host_block_restores             = counters.host_block_restores;
    out.host_tail_restores              = counters.host_tail_restores;
    out.host_write_bytes                = counters.host_write_bytes;
    out.host_restore_bytes              = counters.host_restore_bytes;
    out.evicted_blocks                  = counters.evicted_blocks;
    out.host_snapshot_evictions         = index.host_snapshot_evictions;
    out.host_dead_reclaims              = index.host_dead_reclaims;
    out.unbacked_node_losses            = index.unbacked_node_losses;
    return out;
}

} // namespace ninfer::models::qwen3_5::detail
