#pragma once

// ExpertCache: device residency of routed experts for one Program, over one or two GPUs.
//
// Every GPU p holds the experts e with e % gpus == p (partition by id) in a slot pool and a slot
// table `int32 [layers, experts]` (slot on this GPU, or -1). The device only ever reads the
// table: a kernel maps routed ids to slots with expert_cache_map_slots (-1 = not here, the
// "skipped" id of the expert kernels) and the expert kernel reads the slots. All decisions run on
// the host between windows:
//
//   begin_window(streams)   apply pending table updates on each GPU's compute stream
//   <launch the window (CUDA graph) on those streams>
//   end_window(streams)     record the window's fence on each compute stream
//   observe(layer, routed)  per layer, once the window's routing is known: resident slots and
//                           missing experts as the window saw them, and the policy's insertions
//   submit_fills()          queue the window's insertions on each GPU's fill stream
//                           (submit_fills_unfenced: during the window, see there)
//
// Fill queue: submit_fills appends the window's insertions to a FIFO per GPU (mid-window demand
// insertions, submit_fills_unfenced, go ahead of the backlog); the queue issues
// them in batches (cudaMemcpyBatchAsync from the pinned host bank into the slot, on a dedicated
// fill stream per GPU) while at most `max_in_flight_bytes` are in flight, and drops an insertion
// whose slot was reassigned before it was issued (a backlog never copies stale experts). The
// queue advances whenever the cache is called (begin_window, submit_fills, poll).
//
// SM fills (NINFER_QWEN4_FILL_SM=<ctas>, off by default; v07/decode-overheads A): the batches are
// copied by a kernel of <ctas> blocks on the fill stream that reads the host bank in place (mapped
// pinned memory) instead of the copy engine. Same bytes, same events, same ordering. A copy engine
// keeps enough reads in flight to fill the link, and every small PCIe read of the same GPU (the TP
// exchange's flag and payload, the host experts' partial) waits behind them: T201 measured the
// decode exchange at 5.3 us without a fill on either link, 16.1 with one on GPU 0's, 27.0 with one
// on both (0.42 ms per round). A kernel of a few blocks bounds the fill's reads in flight (256
// threads x 4 x 16 B = 16 KiB per block) at the price of a slower fill.
//
// Epoch fence: an evicted expert is unpublished (table entry -1) at the next begin_window, and
// every fill batch first waits for the fence of the last window launched before it is issued, so
// no launched window can read a slot while it is rewritten. A filled expert is published at the
// first begin_window after its batch completed (polled, never waited for), so the table only
// names complete slots. Until then the policy counts the expert as resident and the device does
// not (observe reports it missing).
//
// Deterministic publication (publish lag >= 1, opt-in; set_publish_lag or
// NINFER_QWEN4_FILL_PUBLISH_LAG): a batch is instead published at the begin_window fixed when it
// was issued: with lag 1 the next one for a batch issued inside a window (it ran beside the rest of
// the window, as a timed copy usually does), the one after for a batch issued between windows (it
// would be waited for at once); begin_window waits for it if it has not completed. poll() does
// nothing, and nothing else retires a batch except flush() (between windows, and while a prompt
// chunk runs). Every decision then depends on the sequence of calls alone, never on how fast a
// copy ran: with timed publication which experts the GPU holds, and so which ones the CPU computes
// with another rounding, follows copy speed, and the greedy text can change between runs.
//
// Pageable banks (v0.7 pinned fallback): a layer whose bank planes are not page-locked host memory
// (cudaPointerGetAttributes: unregistered) is filled through a ring of pinned staging slots of one
// expert each, per GPU: the execution thread copies the expert into a free staging slot (memcpy)
// and the slot joins the batch's cudaMemcpyBatchAsync; it is free again once that batch's event
// completed. A batch takes no more pageable fills than free staging slots (the rest stays queued).
// Layers whose banks are pinned are copied exactly as before.
//
// Not thread-safe: every call comes from the Program's execution thread. The caller owns the
// device pool and table memory (sizes from pool_slots / table_entries) and the host banks, and
// keeps them alive until the ExpertCache is destroyed.

#include "core/expert_cache_policy.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <span>
#include <vector>

namespace ninfer::expert_cache {

inline constexpr int kMaxPlanes = 8;
inline constexpr int kMaxGpus   = 2;

// Where the bytes of one expert live. An expert has `planes` planes of `plane_bytes[q]` bytes.
// Host bank: plane q of (layer l, expert e) begins at bank_base[l * planes + q] + e *
// bank_stride[q] in pinned host memory. Device pool of GPU p: plane q of slot s begins at
// pool_base[p][q] + s * pool_stride[q] (one contiguous slot of all planes, as ninfer-ext's
// kExpertSlotBytes layout, or one array per plane, as qf_moe's).
//
// Variable expert sizes (capy-flash v0.7, ggml block banks: the bytes of an expert depend on its
// layer's formats): when `slot_offset` is not empty, plane q of (layer l, expert e) has
// layer_plane_bytes[l * planes + q] bytes at bank_base[l * planes + q] + e *
// layer_bank_stride[l * planes + q], and plane q of slot s (holding an expert of layer l) begins
// at pool_base[p][q] + slot_offset[s] + layer_plane_offset[l * planes + q]; plane_bytes,
// bank_stride and pool_stride are then unused.
struct ExpertLayout {
    std::int32_t planes = 0;
    std::array<std::int64_t, kMaxPlanes> plane_bytes{};
    std::array<std::int64_t, kMaxPlanes> bank_stride{};
    std::vector<const std::byte*> bank_base; // [layers * planes]
    std::array<std::int64_t, kMaxPlanes> pool_stride{};
    std::vector<std::int64_t> layer_plane_bytes;  // [layers * planes]
    std::vector<std::int64_t> layer_bank_stride;  // [layers * planes]
    std::vector<std::int64_t> layer_plane_offset; // [layers * planes]
    std::vector<std::int64_t> slot_offset;        // [pool slots of one GPU]

    [[nodiscard]] bool variable() const noexcept { return !slot_offset.empty(); }

    [[nodiscard]] std::int64_t bytes(std::int32_t layer, std::int32_t q) const noexcept {
        return variable() ? layer_plane_bytes[static_cast<std::size_t>(layer * planes + q)]
                          : plane_bytes[static_cast<std::size_t>(q)];
    }

    [[nodiscard]] const std::byte* bank(std::int32_t layer, std::int32_t q,
                                        std::int32_t expert) const noexcept {
        const auto i = static_cast<std::size_t>(layer * planes + q);
        return bank_base[i] +
               static_cast<std::int64_t>(expert) *
                   (variable() ? layer_bank_stride[i] : bank_stride[static_cast<std::size_t>(q)]);
    }

    [[nodiscard]] std::int64_t pool_offset(std::int32_t layer, std::int32_t q,
                                           std::int32_t slot) const noexcept {
        if (!variable()) {
            return static_cast<std::int64_t>(slot) * pool_stride[static_cast<std::size_t>(q)];
        }
        return slot_offset[static_cast<std::size_t>(slot)] +
               layer_plane_offset[static_cast<std::size_t>(layer * planes + q)];
    }
};

struct GpuBinding {
    int device = 0;
    std::array<std::byte*, kMaxPlanes> pool_base{};
    std::int32_t* table = nullptr; // device [layers * experts]
    // Optional: device [layers * experts] residency of every partition (the owner GPU's slot, or
    // -1), updated with the tables at every begin_window on this GPU's compute stream. A host path
    // on GPU 0 that must skip the experts resident on any GPU reads it; GPU 1's copy lets it pick
    // the same miss-split share as GPU 0's publish (T42).
    std::int32_t* mirror = nullptr;
};

struct ResidentExpert {
    std::int32_t expert = 0;
    std::int32_t gpu    = 0;
    std::int32_t slot   = 0;
};

// observe() result for one layer; spans stay valid until the next observe().
struct LayerView {
    std::span<const ResidentExpert> resident; // routed and published when the window began
    std::span<const std::int32_t> missing;    // routed and not published: the CPU path's work
    std::span<const Insert> fills;            // the policy's insertions (partition = gpu)
    std::int64_t accesses    = 0;
    std::int64_t policy_hits = 0; // counts resident in the policy (includes fills in flight)
    std::int64_t device_hits = 0; // counts of `resident`
};

struct CacheCounters {
    std::int64_t windows        = 0;
    std::int64_t accesses       = 0;
    std::int64_t policy_hits    = 0;
    std::int64_t device_hits    = 0;
    std::int64_t missing        = 0; // distinct missing per layer-window, summed
    std::int64_t inserts        = 0;
    std::int64_t fills_issued   = 0; // experts copied
    std::int64_t fills_dropped  = 0; // insertions whose slot was reassigned before their copy
    std::int64_t fill_bytes     = 0;
    std::int64_t published      = 0;
    std::int64_t unpublished    = 0;
    std::int64_t table_launches = 0;
    // Elastic columns (T64): experts dropped and moved by set_active_columns shrinks.
    std::int64_t shrink_evicted = 0;
    std::int64_t moved          = 0;
    // v0.7 pinned fallback: fills from pageable banks, copied through the staging ring.
    std::int64_t staged_fills = 0;
    // Deterministic publication: begin_window calls that had to wait for a copy, and the wait.
    std::int64_t publish_waits = 0;
    double publish_wait_us     = 0;
};

// True when `p` is page-locked host memory registered with CUDA (cudaHostAlloc or
// cudaHostRegister): device copies from it are DMA and kernels may read it when mapped.
[[nodiscard]] bool pinned_host_memory(const void* p) noexcept;

class ExpertCache {
public:
    // Experts per fill batch: one cudaMemcpyBatchAsync and one completion event each, so early
    // layers publish before the whole window's fill has landed.
    static constexpr std::int32_t kFillChunkExperts = 16;

    // Default fill depth: ~19 ms of one PCIe 5.0 x8 link (26.8 GB/s), enough to keep the copy
    // engine busy between two calls of a ~25 ms window.
    static constexpr std::int64_t kDefaultInFlightBytes = std::int64_t(512) << 20;

    // Pinned staging ring of each GPU when some bank is pageable (NINFER_HOST_STAGING_MIB
    // overrides; at least one expert).
    static constexpr std::int64_t kDefaultStagingBytes = std::int64_t(128) << 20;

    ExpertCache(const PolicyConfig& policy, ExpertLayout layout, std::span<const GpuBinding> gpus,
                std::int64_t max_in_flight_bytes = kDefaultInFlightBytes);
    ~ExpertCache();

    ExpertCache(const ExpertCache&)            = delete;
    ExpertCache& operator=(const ExpertCache&) = delete;

    // Slots of GPU p's pool (all layers, base and elastic: ExpertCachePolicy::pool_slot_count)
    // and int32 entries of each table.
    [[nodiscard]] static std::int64_t pool_slots(const PolicyConfig& policy, std::int32_t gpu);

    [[nodiscard]] static std::int64_t table_entries(const PolicyConfig& policy) {
        return static_cast<std::int64_t>(policy.layers) * policy.experts;
    }

    [[nodiscard]] const ExpertCachePolicy& policy() const noexcept { return policy_; }

    // T69: ExpertCachePolicy::set_insert_cap (between windows).
    void set_insert_cap(std::int32_t cap) { policy_.set_insert_cap(cap); }

    [[nodiscard]] const CacheCounters& counters() const noexcept { return counters_; }

    [[nodiscard]] std::int32_t gpus() const noexcept { return gpus_; }

    // v0.7 pinned fallback: whether `layer`'s bank is in pageable host memory (filled through the
    // staging ring), and the staging ring of each GPU (experts; 0 when every bank is pinned).
    [[nodiscard]] bool pageable_layer(std::int32_t layer) const noexcept {
        return layer >= 0 && layer < layers_ && pageable_[static_cast<std::size_t>(layer)] != 0;
    }

    [[nodiscard]] std::int32_t staging_slots() const noexcept { return staging_slots_; }

    /**
     * Initializes every table to -1 on its GPU (synchronously) and, when `profile` is not empty,
     * fills each layer from it (profile[l] = experts of layer l hottest first, see
     * ExpertCachePolicy::prefill), waits for the copies and publishes them at the next
     * begin_window. Call once, before the first window.
     */
    void initialize(std::span<const std::vector<std::int32_t>> profile);

    // `streams[p]` is GPU p's compute stream.
    void begin_window(std::span<const cudaStream_t> streams);
    void end_window(std::span<const cudaStream_t> streams);

    [[nodiscard]] LayerView observe(std::int32_t layer, std::span<const Routed> routed,
                                    std::int32_t tokens);
    // observe() of a prompt window (prefill chunk): ExpertCachePolicy::decide_prompt (with
    // `target_cap` > 0 at most that many targets per partition, T211b), or with `insert` false
    // ExpertCachePolicy::count_prompt (T211: the residency stays as it is).
    [[nodiscard]] LayerView observe_prompt(std::int32_t layer, std::span<const Routed> routed,
                                           std::int32_t tokens, bool insert = true,
                                           std::int32_t target_cap = 0);

    void submit_fills();

    /**
     * T211b: a prompt's routing applied between windows (after the prompt's last window, before
     * the next window begins): ExpertCachePolicy::decide_prompt with at most `target_cap` targets
     * per partition (0 = the slot count; < 0: the frequencies only, ExpertCachePolicy::count_prompt
     * with frequencies), its fills queued behind the backlog, fenced to the last
     * launched window (the victims' slots are unpublished at the next begin_window). Nothing for
     * a lent layer or a frozen cache. The counters count the insertions only.
     */
    void apply_prompt(std::int32_t layer, std::span<const Routed> routed, std::int32_t tokens,
                      std::int32_t target_cap);

    /**
     * Mid-window submission: queues the insertions observed so far without waiting for the
     * launched window's fence. The caller guarantees that every launched window has finished
     * reading the slots of the observed layers (their victims' slots): e.g. the window has
     * already published the routing of a later layer, which it does only after the observed
     * layers' expert kernels completed. Later windows cannot read those slots: the victims are
     * unpublished at the next begin_window, before them on the compute stream. These fills are
     * queued ahead of fills submitted earlier and not yet issued (e.g. a prompt window's backlog).
     */
    void submit_fills_unfenced();

    /**
     * Reorders every GPU's submitted, not yet issued fills by `rank[cell]` ascending (stable;
     * cell = layer * experts + expert; a cell past the span ranks last), e.g. a prompt window's
     * residency hottest first across layers. Fills already issued are not affected; slot
     * reassignment safety does not depend on the queue order (stale fills are dropped by
     * generation, each fill keeps its own fence requirement).
     */
    void prioritize_queue(std::span<const std::int32_t> rank);

    // Non-blocking: retires completed fill batches (they publish at the next begin_window) and
    // issues queued fills up to the in-flight limit. Deterministic publication: does nothing (the
    // other calls issue and retire at fixed points).
    void poll();
    // Issues and waits for every submitted fill.
    void flush();

    /**
     * Lending (prefill staging): the slots of layers [first, first + count) of every GPU go to the
     * caller until reclaim_layers(). Call between windows or inside a window none of whose kernels
     * reads those layers through the table. Their experts are evicted (unpublished at the next
     * begin_window); insertions already decided for those slots never land (dropped before issue,
     * not published if in flight); while lent, observe()/observe_prompt() of those layers classify
     * and count but insert nothing. Returns, per GPU, an event recorded on its fill stream after
     * every fill issued before the call: the caller's first write of a lent slot waits for it.
     * Lent slots are the base slots of the layers, the pool slots
     * [policy().base_offset(p, first), policy().base_offset(p, first + count)) of partition p
     * (one contiguous run); the layers' elastic slots stay empty while lent.
     *
     * T69 `keep_elastic`: only the base slots are lent (ExpertCachePolicy::block_base): the
     * layers keep the experts of their elastic slots (published, their fills not dropped), and
     * observe()/observe_prompt() of those layers decide as for any layer within the active
     * elastic columns (a prompt fills them with its hottest experts while the base slots are
     * lent). The caller must not read a lent layer's base slots through the table.
     */
    [[nodiscard]] std::span<const cudaEvent_t> lend_layers(std::int32_t first, std::int32_t count,
                                                           bool keep_elastic = false);
    /**
     * Ends the lending (between windows; the caller's last use of the slots is ordered before the
     * fence of the last ended window, which every refill waits for). The layers are empty (with
     * keep_elastic: hold their elastic residents) and are completed from `profile` (one list per
     * lent layer, hottest first, as initialize; ExpertCachePolicy::top_up) through the fill
     * queue, published as their copies complete. With a non-empty `rank`, the whole queue
     * (these refills and every fill submitted before) is reordered by `rank` (prioritize_queue)
     * before any refill is issued, e.g. the experts' rank in their layer's profile, so the
     * hottest experts of every reclaimed layer land first (T69).
     */
    void reclaim_layers(std::span<const std::vector<std::int32_t>> profile,
                        std::span<const std::int32_t> rank = {});
    /**
     * T69: reclaim_layers of the first `count` lent layers only (between windows): layers
     * [lent_first(), lent_first() + count) return to the cache and refill from `profile` (one
     * list per reclaimed layer, or empty), the others stay lent (their lend events stand); with
     * count == lent_count() this is reclaim_layers.
     */
    void reclaim_front(std::int32_t count, std::span<const std::vector<std::int32_t>> profile,
                       std::span<const std::int32_t> rank = {});

    [[nodiscard]] bool lending() const noexcept { return lent_count_ > 0; }

    [[nodiscard]] std::int32_t lent_first() const noexcept { return lent_first_; }

    [[nodiscard]] std::int32_t lent_count() const noexcept { return lent_count_; }

    // The events lend_layers() returned for the current lending (one per GPU).
    [[nodiscard]] std::span<const cudaEvent_t> lend_events() const noexcept {
        return {lend_events_.data(), static_cast<std::size_t>(gpus_)};
    }

    [[nodiscard]] bool lent(std::int32_t layer) const noexcept {
        return layer >= lent_first_ && layer < lent_first_ + lent_count_;
    }

    /**
     * Elastic columns (T64, ExpertCachePolicy::set_active_columns): between windows, with every
     * window's device work complete. Growing only raises the capacity. Shrinking waits for every
     * submitted fill, drops the coldest experts beyond the new capacity, copies the survivors of
     * the released columns device-to-device into freed kept slots and applies the table updates on
     * `streams` (one compute stream per GPU), completed before returning: afterwards no table,
     * residency mirror, pending fill or later insertion names a slot of a released column, so its
     * memory may be released.
     */
    void set_active_columns(std::int32_t columns, std::span<const cudaStream_t> streams);

    [[nodiscard]] std::int32_t active_columns() const noexcept { return policy_.active_columns(); }

    // Host mirror of the table entry the device holds (as of the last begin_window).
    [[nodiscard]] std::int32_t published_slot(std::int32_t layer, std::int32_t expert) const {
        return published_[cell(layer, expert)];
    }

    // Insertions observed but not yet copied (staged, queued) and copies in flight.
    [[nodiscard]] std::int64_t fills_pending() const noexcept;

    // T59 measurement aid: a frozen cache classifies routed experts (counters, resident slots,
    // missing list) exactly as before but never asks the policy: no insertion, no eviction, the
    // policy state does not move. Residency stays what it was when frozen.
    void set_frozen(bool frozen) noexcept { frozen_ = frozen; }

    /**
     * Deterministic publication: -1 = timed (a copy is published
     * at the first begin_window after it completed), 1..3 = a copy is published at the begin_window
     * fixed when it was issued (lag 1: the next one for a copy issued inside a window, the one
     * after for a copy issued between windows; each step of lag adds one), waiting for it if
     * needed, so the residency depends on the routing history alone.
     * Before initialize() only. Default -1; NINFER_QWEN4_FILL_PUBLISH_LAG overrides.
     */
    void set_publish_lag(std::int32_t lag);

    [[nodiscard]] std::int32_t publish_lag() const noexcept { return publish_lag_; }

    [[nodiscard]] bool frozen() const noexcept { return frozen_; }

    // T59: replace the whole residency by `profile` (one list per layer, as initialize), between
    // windows: every layer is cleared and refilled, the copies are waited for and published at the
    // next begin_window. Policy frequencies are unchanged.
    void restore(std::span<const std::vector<std::int32_t>> profile);
    // Experts of `layer` the cache holds or is filling (the policy's residency), ascending id.
    [[nodiscard]] std::vector<std::int32_t> resident_experts(std::int32_t layer) const;

private:
    struct PendingFill {
        std::int32_t cell        = 0; // layer * experts + expert
        std::int32_t slot        = 0;
        std::uint32_t generation = 0;
        bool fenced              = true; // waits for the fence of the last launched window
    };

    struct FillBatch {
        cudaEvent_t done = nullptr;
        std::int64_t due = 0; // deterministic publication: begin_window (launched_) it is due at
        std::vector<PendingFill> fills;
        std::int32_t staged = 0; // staging slots it holds (pageable fills)
    };

    struct Gpu {
        GpuBinding binding;
        cudaStream_t fill_stream = nullptr;
        cudaEvent_t fence        = nullptr;    // end of the last launched window
        std::vector<std::uint32_t> generation; // [pool slots] bumped at every reassignment
        std::vector<PendingFill> staged;       // insertions of the current window, not submitted
        std::vector<std::int32_t> staged_at;   // [pool slots] index in `staged` or -1
        std::deque<PendingFill> queue;         // submitted, not issued
        std::deque<FillBatch> in_flight;
        std::int64_t in_flight_bytes = 0;
        std::vector<cudaEvent_t> spare_events;
        // Pinned staging ring (pageable banks): slots [0, staging_slots_), FIFO like in_flight.
        std::byte* staging          = nullptr;
        std::int64_t staging_head   = 0; // slots handed out, ever
        std::int32_t staging_in_use = 0;
    };

    [[nodiscard]] std::size_t cell(std::int32_t layer, std::int32_t expert) const noexcept {
        return static_cast<std::size_t>(layer) * static_cast<std::size_t>(experts_) +
               static_cast<std::size_t>(expert);
    }

    void stage(const Insert& ins, std::int32_t layer);
    void lent_decision(std::span<const Routed> routed);
    void frozen_decision(std::span<const Routed> routed, const LayerView& view);
    [[nodiscard]] LayerView classify(std::int32_t layer, std::span<const Routed> routed);
    [[nodiscard]] LayerView finish_observe(std::int32_t layer, LayerView view);
    void queue_staged(bool fenced, bool issue_now = true);
    // Launches the pending table updates on `streams` and commits them to the host mirror.
    void publish_targets(std::span<const cudaStream_t> streams);
    void set_target(std::size_t cell, std::int32_t value);
    void pump(Gpu& gpu);
    void issue(Gpu& gpu, std::span<const PendingFill> fills);
    void retire(Gpu& gpu, bool wait);
    // The front batch's bookkeeping (its fills publish at the next begin_window).
    void retire_front(Gpu& gpu);

    [[nodiscard]] bool deterministic() const noexcept { return publish_lag_ > 0; }

    // Deterministic publication: the begin_window `batch` is due at.
    void schedule(FillBatch& batch) const noexcept;
    // Deterministic publication, in begin_window: retires every batch due by now (waiting).
    void retire_due();
    void check_streams(std::span<const cudaStream_t> streams) const;

    ExpertCachePolicy policy_;
    ExpertLayout layout_;
    std::int64_t expert_bytes_        = 0;
    std::int64_t max_in_flight_bytes_ = 0;
    std::int32_t gpus_                = 0;
    std::int32_t layers_              = 0;
    std::int32_t experts_             = 0;
    std::array<Gpu, kMaxGpus> gpu_{};
    std::vector<std::uint8_t> pageable_; // [layers] bank in pageable host memory
    std::int32_t staging_slots_ = 0;     // per GPU

    std::vector<std::int32_t> published_; // [cells] table value on the owner GPU
    std::vector<std::int32_t> target_;    // [cells] value to publish at the next begin_window
    std::vector<std::uint8_t> dirty_;     // [cells]
    std::vector<std::int32_t> dirty_list_;

    std::int32_t lent_first_ = 0;
    std::int32_t lent_count_ = 0;
    bool lent_keep_          = false; // T69: the lent layers keep their elastic slots
    std::array<cudaEvent_t, kMaxGpus> lend_events_{};

    std::int64_t launched_ = 0; // windows begun
    std::int64_t fenced_   = 0; // windows whose fence is recorded
    bool initialized_      = false;
    bool frozen_           = false;
    // NINFER_QWEN4_FILL_SM=<ctas> (v07/decode-overheads A): CTAs of the kernel that copies fills
    // from the mapped host bank instead of the copy engine; 0 = copy engine (cudaMemcpyBatchAsync).
    std::int32_t sm_fill_ctas_ = 0;
    std::int32_t publish_lag_  = -1; // set_publish_lag

    LayerDecision decision_;
    std::vector<ResidentExpert> resident_;
    std::vector<std::int32_t> missing_;
    CacheCounters counters_;

    // Scratch of issue().
    std::vector<void*> copy_dst_;
    std::vector<const void*> copy_src_;
    std::vector<std::size_t> copy_bytes_;
};

/**
 * slot_ids[i] = table[layer * experts + ids[i]] for 0 <= ids[i] < experts, otherwise -1, for
 * i < count. `table` is one GPU's slot table; `layer` and `count` are fixed launch parameters, so
 * the launch can be captured in a CUDA graph and replayed while the table changes between
 * replays.
 */
void expert_cache_map_slots(const std::int32_t* ids, std::int32_t count, std::int32_t layer,
                            std::int32_t experts, const std::int32_t* table, std::int32_t* slot_ids,
                            cudaStream_t stream);

} // namespace ninfer::expert_cache
